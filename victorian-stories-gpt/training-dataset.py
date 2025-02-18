import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import time
from fsspec.exceptions import FSTimeoutError


def load_dataset_with_retry(dataset_name, max_retries=5, initial_delay=1, backoff_factor=2):
    retries = 0
    while retries < max_retries:
        try:
            print(f"Attempting to load dataset (Attempt {retries + 1}/{max_retries})...")
            dataset = load_dataset(dataset_name, split="train")
            print("Dataset loaded successfully!")
            return dataset
        except FSTimeoutError as e:
            retries += 1
            if retries == max_retries:
                print(f"Failed to load dataset after {max_retries} attempts. Error: {str(e)}")
                raise
            delay = initial_delay * (backoff_factor ** (retries - 1))
            print(f"Timeout error occurred. Retrying in {delay} seconds...")
            time.sleep(delay)


def tokenize(doc, enc, eot):
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2 ** 16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)


def process_data(args):
    fw, enc, eot, shard_size, DATA_CACHE_DIR = args
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None

    for doc in fw:
        tokens = tokenize(doc, enc, eot)

        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count:token_count + len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"pg19_{split}_{shard_index:06d}")
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"pg19_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])


def main():
    local_dir = "../gpt-2/pg19-dataset"
    shard_size = int(1e8)  # 100M tokens per shard

    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    try:
        fw = load_dataset_with_retry("deepmind/pg19")
    except Exception as e:
        print(f"Failed to load dataset: {str(e)}")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']

    nprocs = max(1, os.cpu_count() // 2)

    # Use the default context
    with mp.Pool(nprocs) as pool:
        pool.apply(process_data, [(fw, enc, eot, shard_size, DATA_CACHE_DIR)])


if __name__ == '__main__':
    main()