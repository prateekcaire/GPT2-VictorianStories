import argparse
import math
import subprocess
import sys
import time

import numpy as np
import tiktoken
import torch
import logging
from matplotlib import pyplot as plt
from tqdm import tqdm

from model import LanguageModel
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def log_with_rank(message, rank, level=logging.INFO):
    logger.log(level, f"Rank {rank} - {message}")


def log_gpu_memory_status(rank):
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory
            allocated_memory = torch.cuda.memory_allocated(i)
            cached_memory = torch.cuda.memory_reserved(i)
            free_memory = total_memory - allocated_memory - cached_memory
            log_with_rank(f"GPU {i}: Total: {total_memory / 1e9:.2f}GB, "
                          f"Allocated: {allocated_memory / 1e9:.2f}GB, "
                          f"Cached: {cached_memory / 1e9:.2f}GB, "
                          f"Free: {free_memory / 1e9:.2f}GB", rank)
    else:
        log_with_rank("CUDA is not available", rank)


def check_cuda_error(rank):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        if torch.cuda.is_current_stream_capturing():
            log_with_rank("CUDA graph capture is active", rank, level=logging.WARNING)


def get_device():
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


class DataLoader:
    def __init__(self, b, t, process_rank, num_processes, split):
        logger.info(
            f"Initializing DataLoader: batch_size={b}, n_tokens={t}, rank={process_rank}, world_size={num_processes}, split={split}")

        self.b = b
        self.t = t
        self.process_rank = process_rank
        self.num_processes = num_processes
        with open('../input.txt', 'r', encoding='utf-8') as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = torch.tensor(enc.encode(text))
        self.pos = self.b * self.t * self.process_rank

    @staticmethod
    def load_tokens(filename):
        npt = np.load(filename)
        npt = npt.astype(np.int32)
        return torch.tensor(npt, dtype=torch.long)

    def get_batch(self):
        logger.debug(f"Getting batch, current position: {self.pos}")
        batch_tokens = self.tokens[self.pos:self.pos + self.b * self.t + 1]
        xs = (batch_tokens[:-1]).view(self.b, self.t)
        ys = (batch_tokens[1:]).view((self.b, self.t))
        self.pos = self.pos + self.b * self.t * self.num_processes
        if self.pos + self.b * self.t * self.num_processes + 1 > len(self.tokens):
            self.pos = self.process_rank * self.b * self.t
        logger.debug(f"Batch fetched, new position: {self.pos}")
        return xs, ys


def setup_distributed():
    logger.info("Starting setup_distributed()")
    if 'WORLD_SIZE' in os.environ:
        ddp_world_size = int(os.environ['WORLD_SIZE'])
    else:
        ddp_world_size = 1

    ddp = ddp_world_size > 1
    if ddp:
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'

        if 'RANK' in os.environ:
            ddp_rank = int(os.environ['RANK'])
        else:
            ddp_rank = 0

        if 'LOCAL_RANK' in os.environ:
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
        else:
            ddp_local_rank = ddp_rank % torch.cuda.device_count()

        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, rank=ddp_rank, world_size=ddp_world_size)
        device = f'{get_device()}:{ddp_local_rank}'
        torch.cuda.set_device(ddp_local_rank)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        device = get_device()
        master_process = True

    logger.info(f"Rank: {ddp_rank}, Local Rank: {ddp_local_rank}, World Size: {ddp_world_size}, Device: {device}")
    return ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process


def train_model(args):
    ddp = False
    ddp_rank = 0
    try:
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device, master_process = setup_distributed()

        log_with_rank(
            f"Process rank: {ddp_rank}, local rank: {ddp_local_rank}, world size: {ddp_world_size}, device: {device}",
            ddp_rank)

        log_gpu_memory_status(ddp_rank)

        log_with_rank(f"Creating model with args: {args}", ddp_rank)
        m = LanguageModel(args.n_layers, args.n_channels, args.n_vocab, args.n_tokens, args.n_heads, args.dropout,
                          device=device)
        log_with_rank(f"Model created, moving to device: {device}", ddp_rank)
        m.to(device)

        model_size = sum(p.numel() for p in m.parameters()) / 1e6
        log_with_rank(f"Model size: {model_size:.2f}M parameters", ddp_rank)

        max_steps = 1000  # completes whole data
        max_lr = 1e-3
        min_lr = max_lr * 0.1
        warmup_steps = 100

        def get_lr(cur_step):
            if cur_step < warmup_steps:
                return max_lr * (cur_step + 1) / warmup_steps
            if cur_step > max_steps:
                return min_lr
            decay_ratio = (cur_step - warmup_steps) / (max_steps - warmup_steps)
            decay = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return min_lr + decay * (max_lr - min_lr)

        # def get_lr(cur_step):
        #     return max_lr


        if torch.cuda.is_available():
            log_with_rank(f"CUDA available: {torch.cuda.is_available()}", ddp_rank)
            log_with_rank(f"CUDA device count: {torch.cuda.device_count()}", ddp_rank)
            log_with_rank(f"Current CUDA device: {torch.cuda.current_device()}", ddp_rank)

        log_with_rank("Creating optimizer", ddp_rank)
        optimizer = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-2)

        if torch.cuda.is_available():
            log_with_rank("Compiling model for CUDA", ddp_rank)
            m = torch.compile(m)
        elif device == 'mps':
            log_with_rank("torch.compile() is not fully supported for MPS. Using the model without compilation.",
                          ddp_rank)
        else:
            log_with_rank("Using CPU. Model compilation is not applicable.", ddp_rank)

        if ddp:
            log_with_rank("Wrapping model with DDP", ddp_rank)
            m = DDP(m, device_ids=[ddp_local_rank])

        torch.manual_seed(1337 + ddp_rank)  # MODIFIED: Different seed for each rank
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)

        epochs, b, t = args.epochs, args.batch_size, args.n_tokens
        total_batch_size = 16384
        assert total_batch_size % (b * t) == 0
        grad_accumulation_steps = total_batch_size // (b * t * ddp_world_size)
        log_with_rank("Setting up data loader", ddp_rank)
        data_loader = DataLoader(b, t, ddp_rank, ddp_world_size, 'train')
        log_with_rank(f'1 epoch : {len(data_loader.tokens) // (data_loader.b * data_loader.t)}', ddp_rank)

        steps, losses = [], []

        autocast_device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        if torch.cuda.is_available():
            autocast_dtype = torch.float16  # todo: use torch.bfloat16 if available and preferred
        elif device == 'mps':
            autocast_dtype = torch.float16
        else:
            autocast_dtype = torch.bfloat16  # for CPU
        log_with_rank(f"Using autocast with device_type: {autocast_device_type}, dtype: {autocast_dtype}", ddp_rank)

        use_grad_scaler = torch.cuda.is_available() and autocast_dtype == torch.float16
        scaler = torch.cuda.amp.GradScaler() if use_grad_scaler else None

        for step in tqdm(range(max_steps)):
            if step % 100 == 0:
                log_gpu_memory_status(ddp_rank)
            t0 = time.time() * 1000
            optimizer.zero_grad()
            xs, ys = data_loader.get_batch()
            xs, ys = xs.to(device), ys.to(device)
            loss_accum = torch.zeros(1, device=device)

            for micro_step in range(grad_accumulation_steps):
                with torch.autocast(device_type=autocast_device_type, dtype=autocast_dtype):
                    _, loss = m(xs, ys)
                    loss = loss / grad_accumulation_steps

                if use_grad_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss_accum += loss.detach()

                if ddp:
                    m.require_backward_grad_sync = micro_step == (grad_accumulation_steps - 1)

            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

            if use_grad_scaler:
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                optimizer.step()

            lr = get_lr(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()

            check_cuda_error(ddp_rank)

            t1 = time.time() * 1000
            tokens_processed = grad_accumulation_steps * data_loader.b * data_loader.t * ddp_world_size
            dt = t1 - t0
            tokens_per_s = tokens_processed * 1000 / dt
            if master_process:
                log_with_rank(
                    f'loss = {loss_accum.item():.4f} | step {step} | perf {tokens_per_s:.6f} | time {(t1 - t0):.4f} | norm {norm:.4f} | lr {lr:.4e}',
                    ddp_rank)
                losses.append(loss_accum.item())
                steps.append(step)

        log_with_rank("Training completed", ddp_rank)
        if ddp:
            dist.barrier()
            dist.destroy_process_group()
        if master_process:
            save_model(m, args)
            plot_loss_chart(steps, losses)
        log_with_rank("Exiting train_model()", ddp_rank)
        return losses, steps

    except Exception as e:
        log_with_rank(f"Error in train_model: {str(e)}", ddp_rank, level=logging.ERROR)
        raise
    finally:
        if ddp:
            dist.destroy_process_group()


def save_model(model, args):
    model_dict = {
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }
    torch.save(model_dict, os.path.join(args.model_dir, 'model.pth'))
    print(f"Model saved to {args.model_dir}")


import os


def plot_loss_chart(steps, losses):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)

    # Get the SageMaker output directory
    output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '')

    # Save the figure to the output directory
    output_path = os.path.join(output_dir, 'training_loss_chart.png')
    plt.savefig(output_path)
    plt.close()

    print(f"Loss chart saved to {output_path}")


if __name__ == '__main__':
    try:
        print(f'in train.py')
        parser = argparse.ArgumentParser()
        parser.add_argument('--model-dir', type=str, default='./sp_model/')
        parser.add_argument('--n_layers', type=int, default=6)
        parser.add_argument('--n_channels', type=int, default=384)
        parser.add_argument('--n_vocab', type=int, default=50304)
        parser.add_argument('--n_tokens', type=int, default=512)
        parser.add_argument('--n_heads', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--epochs', type=int, default=2)
        parser.add_argument('--dropout', type=float, default=0.2)
        # parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
        # parser.add_argument('--use_mixed_precision', type=bool, default=True)

        args = parser.parse_args()
        logger.info(f"Parsed arguments: {args}")
        train_model(args)
        logger.info("Script execution completed")
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
