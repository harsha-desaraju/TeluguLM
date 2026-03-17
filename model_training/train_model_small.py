
import time
import math
import torch
import inspect
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass
from tokenizers import Tokenizer

from tqdm import tqdm
import pyarrow.dataset as ds


torch.set_default_dtype(torch.bfloat16)
torch.set_float32_matmul_precision("high")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # Flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.register_buffer("pos", torch.arange(config.block_size))

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward a sequence of length {T}, max block size if 1024"
        # pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(self.pos[:T])
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer



class TeluguText(IterableDataset):
    def __init__(self, path: str, loading_batch_size: int, columns: list[str] = None):
        self.dataset = ds.dataset(path, format="parquet")
        self.columns = columns
        self.loading_batch_size = loading_batch_size

    def __len__(self):
        return self.dataset.count_rows()

    def __iter__(self):
        scanner = self.dataset.scanner(
            columns=self.columns,
            batch_size=self.loading_batch_size
        )

        for batch in scanner.to_batches():
            # tokens = batch.column("tokens").values.to_numpy()
            # tokens = np.array(batch["tokens"].to_pylist())
            for row in batch["tokens"]:
                row = row.values.to_numpy()
                x, y = row[:-1], row[1:]
                x, y = torch.tensor(x), torch.tensor(y)
                yield x, y


def save_model(model, optimizer, step, loss, output_dir):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss
    }
    path = f"{output_dir}/model_checkpoint_{step}.pt"
    torch.save(checkpoint, path)


def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    if it > max_steps:
        return min_lr
    # Use cosine decay in between
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)



if __name__ == '__main__':
    learning_rate = 6e-4
    weight_decay = 0.1
    warmup_steps = 500
    max_lr = 6e-4
    min_lr = max_lr * 0.1

    B, T = 16, 512

    model_config = GPTConfig(vocab_size=32768, block_size=T, n_layer=8, n_head=8, n_embd=512)

    num_epochs = 2
    print_step = 500
    save_checkpoint_steps = 1000


    # Auto-detecting the device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"Device: {device}")

    # Model initialization
    model = GPT(model_config)
    model = model.to(device, dtype=torch.bfloat16)
    model = torch.compile(model)

    # Print the size of the model
    params = 0
    for layer in model.parameters():
        params += layer.numel()

    print(f"The number of parameters of the model is: {params}")

    train_dataset = TeluguText("./data/train/", loading_batch_size=8192, columns=['tokens'])
    val_dataset = TeluguText("./data/val/", loading_batch_size=8192, columns=['tokens'])

    tokenizer = Tokenizer.from_file("../tokenization/telugu_tokenizer/tokenizer.json")

    train_loader = DataLoader(train_dataset, batch_size=B, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=B, num_workers=0)

    print(f"The length of the training dataset is: ",len(train_dataset))
    print(f"The length of the train loader is: ", len(train_loader))

    max_steps = num_epochs * len(train_loader)
    # print(f"-------No of max steps: {max_steps} -------")


    optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate,
                                           device_type=device,
                                           betas=(0.9, 0.95))

    for epoch in range(num_epochs):
        # Train epoch
        for step, (x, y) in enumerate(tqdm(train_loader)):
            global_step = epoch * len(train_loader) + step
            t0 = time.time()
            optimizer.zero_grad()

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits, loss = model(x, y)

            loss.backward()

            # Clipping the gradients
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Get the learning rate for this iteration and set it
            lr = get_lr(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            optimizer.step()

            t1 = time.time()
            dt = (t1 - t0) * 1000
            tokens_per_sec = x.numel() / (t1 - t0)

            if (step % print_step == 0) or (step <= 5):
                print(
                    f"step {step} | loss: {loss.item():.5f} | norm: {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")

            if step % save_checkpoint_steps == 0:
                save_model(model, optimizer, global_step, loss.item(), "./models")


        # Test epoch
        model.eval()
        val_loss_accum = 0
        with torch.no_grad():
            for (x, y) in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, loss = model(x, y)
                val_loss_accum += loss.detach()
            val_loss = val_loss_accum / len(val_loader)

        print(f"epoch {epoch+1} | val_loss: {val_loss.item():.5f}")
        model.train()

    save_model(model, optimizer, max_steps, loss.item(), "./models")