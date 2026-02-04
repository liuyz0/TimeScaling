
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataclasses import dataclass
import sys
import time
from scipy.sparse.linalg import LinearOperator, eigsh
import numpy as np

# ---- hyperparameters ----
task_id = int(sys.argv[1]) # from 0 to 7
num_tasks = int(sys.argv[2]) # should be 8
assert num_tasks == 8

temperatures = torch.logspace(-2, -1.2, steps=8)
temperature = temperatures[task_id % 8]
vocab_size = 128

batch_size = 1024
num_batch_H = 4
num_steps = 50_000
log_interval = 200
neigs = 3

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

    def forward(self, x: torch.Tensor, temperature: float = 1.0):
        x = rmsnorm(x)  # (B, n_embd)
        logits = self.head(x) / temperature  # (B, vocab_size)
        return logits

# ---- Hessian ----
def prepare_hvp(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = torch.cat([g.reshape(-1) for g in grads])

    def hvp(vec):
        grad_vec = (flat_grad * vec).sum()
        hv = torch.autograd.grad(grad_vec, params, retain_graph=True)
        return torch.cat([h.reshape(-1) for h in hv])

    return hvp

def lanczos(matrix_vector, dim: int, neigs: int):
    """ Invoke the Lanczos algorithm to compute the leading eigenvalues and eigenvectors of a matrix / linear operator
    (which we can access via matrix-vector products). """

    def mv(vec: np.ndarray):
        return matrix_vector(torch.tensor(vec))

    operator = LinearOperator((dim, dim), matvec=mv)
    evals, evecs = eigsh(operator, neigs)
    return torch.from_numpy(np.ascontiguousarray(evals[::-1]).copy()).float(), \
           torch.from_numpy(np.ascontiguousarray(np.flip(evecs, -1)).copy()).float()

def get_hessian_eigenvalues(network, losses, neigs=1, return_trace=False, nprobe=16):
    params = [p for p in network.parameters() if p.requires_grad]
    hvps = [prepare_hvp(loss, params) for loss in losses] # list of hvp functions
    def hvp_delta(vec):
        result = torch.zeros_like(vec)
        for hvp in hvps:
            result += hvp(vec)
        return result / len(hvps)
    nparams = torch.cat([p.reshape(-1) for p in params]).numel()
    evals, _ = lanczos(hvp_delta, nparams, neigs=neigs)
    if return_trace:
        trace = 0.0
        for _ in range(nprobe):
            v = torch.randn(nparams)
            hv = hvp_delta(v)
            trace += torch.dot(v, hv).item()
        trace /= nprobe
        return evals, trace
    else:
        return evals

# ---- main ----
train_losses = torch.zeros(num_steps)
test_losses = torch.zeros(num_steps // log_interval)
eign_vals = torch.zeros((num_steps // log_interval, neigs))
traces = torch.zeros(num_steps // log_interval)
weight_norms = torch.zeros(num_steps // log_interval)
grad_norms = torch.zeros(num_steps // log_interval)
cfg = HeadConfig(vocab_size=vocab_size)
t_model = Head(cfg) # teacher model
s_model = Head(cfg) # student model
with torch.no_grad():
    s_model.head.weight.zero_()
    
optimizer = torch.optim.Adam(s_model.head.parameters(), lr=3e-3)

# save target model
torch.save(t_model.state_dict(), 
           f'../outputs/exp-0-teacher-{task_id}.pt')

start_time = time.time()
for step in range(num_steps):
    x = torch.randn(batch_size, cfg.n_embd)  # (B, n_embd)
    with torch.no_grad():
        t_logits = t_model(x, temperature=temperature)  # (B, vocab_size)
        t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
    s_logits = s_model(x)  # (B, vocab_size)
    s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
    loss = F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True)  # KL divergence
    train_losses[step] = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (step + 1) % log_interval == 0:
        Hlosses = []
        for _ in range(num_batch_H):
            x = torch.randn(batch_size, cfg.n_embd)  # (B, n_embd)
            with torch.no_grad():
                t_logits = t_model(x, temperature=temperature)  # (B, vocab_size)
                t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab_size)
            s_logits = s_model(x)  # (B, vocab_size)
            s_log_probs = F.log_softmax(s_logits, dim=-1)  # (B, vocab_size)
            Hlosses.append(F.kl_div(s_log_probs, t_log_probs, reduction='batchmean', log_target=True))
        grad_norms[(step + 1) // log_interval - 1] = torch.cat([g.reshape(-1) for g in torch.autograd.grad(torch.stack(Hlosses).mean(), tuple(s_model.parameters()), retain_graph=True)]).norm()
        est_evals, trace = get_hessian_eigenvalues(s_model, Hlosses, neigs=neigs, return_trace=True)
        with torch.no_grad():
            test_losses[(step + 1) // log_interval - 1] = torch.stack(Hlosses).mean().item()
            eign_vals[(step + 1) // log_interval - 1] = est_evals
            traces[(step + 1) // log_interval - 1] = trace
            weight_norms[(step + 1) // log_interval - 1] = s_model.head.weight.norm().item()
        del Hlosses
        # save model checkpoint
        torch.save(s_model.state_dict(), 
                   f'../outputs/exp-0-ckp-{task_id}-{(step + 1) // log_interval - 1}.pt')

print(f'Run time {time.time() - start_time:.2f} seconds for {num_steps} steps and {vocab_size} vocab size.')
torch.save({'train_losses': train_losses, 
            'test_losses': test_losses, 
            'eign_vals': eign_vals,
            'traces': traces,
            'weight_norms': weight_norms,
            'grad_norms': grad_norms}, 
           f'../outputs/exp-0-{task_id}.pt')
