import torch
from torch import nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_channels, n_head_channels, n_tokens, dropout, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = nn.Linear(n_channels, n_head_channels)
        self.q = nn.Linear(n_channels, n_head_channels)
        self.v = nn.Linear(n_channels, n_head_channels)
        self.register_buffer('tril', torch.tril(torch.ones(n_tokens, n_tokens)).to(device))
        self.dropout = nn.Dropout(dropout)
        self.device = device

    def forward(self, x):
        k = self.k(x)
        q = self.q(x)
        v = self.v(x)
        t = x.shape[1]

        if self.device == 'mps':
            # MPS doesn't support scaled_dot_product_attention, use manual implementation
            w = torch.matmul(q, k.transpose(-2, -1)) * k.shape[-1] ** (-0.5)
            w = w.masked_fill(self.tril[:t, :t] == 0, float('-inf'))
            w = F.softmax(w, dim=-1)
            w = self.dropout(w)
            y = w @ v
        else:
            # Use flash attention for CUDA and CPU
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = self.dropout(y)
        return y


class MultiAttentionHead(nn.Module):
    def __init__(self, n_heads, n_channels, n_tokens, f_dropout, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_head_channels = n_channels // n_heads
        self.attentions = nn.ModuleList(
            [CausalSelfAttention(n_channels, n_head_channels, n_tokens, f_dropout, device) for _ in range(n_heads)])
        self.linear = nn.Linear(n_channels, n_channels)
        self.dropout = nn.Dropout(f_dropout)

    def forward(self, x):
        x = torch.cat([attention(x) for attention in self.attentions], dim=-1)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, n_channels, f_dropout, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.network = nn.Sequential(
            nn.Linear(n_channels, 4 * n_channels),
            nn.GELU(approximate='tanh'),
            nn.Linear(4 * n_channels, n_channels),
            nn.Dropout(f_dropout)
        )

    def forward(self, x):
        return self.network(x)


class Block(nn.Module):
    def __init__(self, n_channels, n_heads, n_tokens, f_dropout, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln1 = nn.LayerNorm(n_channels)
        self.mah = MultiAttentionHead(n_heads, n_channels, n_tokens, f_dropout, device)
        self.ln2 = nn.LayerNorm(n_channels)
        self.ffn = FeedForwardNetwork(n_channels, f_dropout)

    def forward(self, x):
        x = x + self.mah(self.ln1(x))  # residual connection
        x = x + self.ffn(self.ln2(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, n_layers=8, n_channels=384, n_vocab=50304, n_tokens=1024, n_heads=12, f_dropout=0.2,
                 device='cpu', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.n_tokens = n_tokens
        self.token_embeddings = nn.Embedding(n_vocab, n_channels)
        self.positional_embeddings = nn.Embedding(n_tokens, n_channels)
        self.blocks = nn.Sequential(
            *[Block(n_channels, n_heads, n_tokens, f_dropout, device) for _ in range(n_layers)],
            nn.LayerNorm(n_channels)
        )
        self.lm_head = nn.Linear(n_channels, n_vocab)

    def forward(self, idx, y=None):
        idx = idx.to(self.device)
        b, t = idx.shape
        tokens_emb = self.token_embeddings(idx)  # (B,T,C)
        positional_emb = self.positional_embeddings(torch.arange(t, device=self.device))
        x = tokens_emb + positional_emb
        x = self.blocks(x)
        logits = self.lm_head(x)
        if y is not None:
            y = y.to(self.device)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return logits, loss
        return logits  # return logits

    def generate(self, idx, max_tokens, tokenizer, callback=None, temperature=1.3, top_k=50, top_p=0.95,
                 repetition_penalty=1.2):
        self.eval()
        generated = idx.clone()
        past_tokens = set()

        with torch.no_grad():
            for _ in range(max_tokens):
                idx_cond = generated[:, -self.n_tokens:]
                logits = self(idx_cond)
                logits = logits[:, -1, :]

                # Apply repetition penalty
                if len(past_tokens) > 0:
                    for token in past_tokens:
                        logits[:, token] /= repetition_penalty

                # Apply temperature
                logits = logits / temperature

                # Top-k sampling
                if top_k > 0:
                    top_k = min(top_k, logits.size(-1))  # Safety check
                    indices_to_remove = logits < torch.topk(logits, top_k, dim=-1)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')

                # Top-p (nucleus) sampling
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    # Create a boolean mask for the original logits
                    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices,
                                                                                            sorted_indices_to_remove)

                    logits[indices_to_remove] = float('-inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                generated = torch.cat((generated, next_token), dim=1)
                past_tokens.add(next_token.item())

                if callback:
                    token_text = tokenizer.decode([next_token.item()])
                    callback(token_text)

                if next_token.item() == tokenizer.eot_token:
                    break

        generated_text = tokenizer.decode(generated[0].tolist())
        return generated_text
