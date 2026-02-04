# now fix teacher temperature and scan lr and initialization scale
# add lr decay

import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import sys
import time

# ---- hyperparameters ----
task_id = int(sys.argv[1]) # from 0 to 95
num_tasks = int(sys.argv[2]) # should be 96
assert num_tasks == 96

lrs = torch.logspace(-3, 0, steps=12)
#temperatures = torch.logspace(-3, 0, steps=8)
init_ratios = torch.linspace(0, 1, steps=8)
lr = lrs[task_id // 8]
temperature = 6e-3
init_ratio = init_ratios[task_id % 8]
vocab_size = 128

batch_size = 1024
test_num_batch = 10
num_steps = 100_000
log_interval = 100

# ---- Config ----
@dataclass
class HeadConfig:
    vocab_size: int = 128     # vocab size
    n_embd: int = 32         # model width (d_model)

# ---- Utils ----
def rmsnorm(x: torch.Tensor):
    return F.rms_norm(x, (x.size(-1),))

class Head(nn.Module):
    def __init__(self, config: HeadConfig):
        super().__init__()
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # self.head.weight.data.normal_(mean=0.0, std=1 / (config.n_embd ** 0.5))

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        x = rmsnorm(x)  # (B, n_embd)
        logits = self.head(x) / temperature  # (B, vocab_size)
        return logits
    
# lr decay
def lr_scheduler(step, total_steps = 100_000):
    # linear warmup 0.01 + cosine decay to 0.1 max lr
    step = torch.tensor(step)
    if step < total_steps * 0.01:
        return step / (total_steps * 0.01)
    else:
        progress = (step - total_steps * 0.01) / (total_steps * 0.99)
        return 0.5 * (1.0 + torch.cos(torch.pi * progress)) * 0.9 + 0.1

# ---- main ----
#train_losses = torch.zeros(num_steps)
test_losses = torch.zeros(num_steps // log_interval)
weight_norms = torch.zeros(num_steps // log_interval)
logit_stds = torch.zeros(num_steps // log_interval)
cfg = HeadConfig(vocab_size=vocab_size)
t_model = Head(cfg) # teacher model
s_model = Head(cfg) # student model
with torch.no_grad():
    s_model.head.weight.mul_(init_ratio / temperature)
    
optimizer = torch.optim.Adam(s_model.head.parameters())

start_time = time.time()
for step in range(num_steps):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * lr_scheduler(step+1, total_steps=num_steps)
    x = torch.randn(batch_size, cfg.n_embd)  # (B, n_embd)
    with torch.no_grad():
        t_logits = t_model(x, temperature=temperature)  # (B, vocab_size)
        t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
    s_logits = s_model(x)  # (B, vocab_size)
    s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
    loss = F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True)  # KL divergence
    #train_losses[step] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % log_interval == 0:
        with torch.no_grad():
            test_loss = 0.0
            logit_std = 0.0
            for _ in range(test_num_batch):
                test_x = torch.randn(batch_size, cfg.n_embd)  # (B, n_embd)
                t_logits = t_model(test_x, temperature=temperature)  # (B, vocab_size)
                t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
                s_logits = s_model(test_x)  # (B, vocab_size)
                s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
                test_loss += F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True).item()
                logit_std += s_logits.std(dim = -1).mean().item()
            test_loss /= test_num_batch
            test_losses[step // log_interval] = test_loss
            weight_norms[step // log_interval] = s_model.head.weight.norm().item()
            logit_stds[step // log_interval] = logit_std / test_num_batch

print(f'Run time {time.time() - start_time:.2f} seconds for {num_steps} steps and {vocab_size} vocab size.')
torch.save({'test_losses': test_losses,
            'weight_norms': weight_norms,
            'logit_stds': logit_stds}, 
           f'../outputs/exp-2-1-{task_id}.pt')