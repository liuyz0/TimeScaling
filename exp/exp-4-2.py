# from test-9
# full batch MNIST
# on GPU
# smaller model

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

num_steps = 100_000

class MLP(nn.Module):
    def __init__(self, m ,m_in = 28*28, ell=6, n=10):
        super(MLP, self).__init__()
        # m : width
        # n : number of classes
        # ell : number of layers
        self.fc_layers = nn.ModuleList([])
        for l in range(ell):
            if l == 0:
                self.fc_layers.append(nn.Linear(m_in, m))
            else:
                self.fc_layers.append(nn.Linear(m, m))
        self.fc_out = nn.Linear(m, n, bias=False)
        self.fc_out.weight.data.zero_()

        self.bn_layers = nn.ModuleList([])
        for l in range(ell):
            if l == 0:
                self.bn_layers.append(nn.BatchNorm1d(m_in))
            else:
                self.bn_layers.append(nn.BatchNorm1d(m))
        self.bn_out = nn.BatchNorm1d(m)
    
    def forward(self, x):
        # x : [B, m_in]
        for l in range(len(self.fc_layers)):
            x = self.bn_layers[l](x)
            x = self.fc_layers[l](x) # [B, m]
            x = torch.relu(x)
        x = self.bn_out(x)
        x = self.fc_out(x) # [B, n] logits
        return x

transform=transforms.Compose([
            transforms.ToTensor()
        ])

train_dataset = datasets.MNIST(root='../data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=60000)
full_batch = next(iter(train_loader))
inputs, labels = full_batch
x_in = inputs.view(-1, 28*28).to('cuda') # [60000, 784]
labels = labels.to('cuda')

model = MLP(m=10, ell = 1).to('cuda') # initialize a model
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = []
accuracies = []
logit_stds = []

model.train()
start_time = time.time()
for step in range(num_steps):
    logits = model(x_in) # [60000, 10]
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        logit_stds.append(logits.std(dim = -1).mean().item())
        accuracy = (logits.argmax(dim=-1) == labels).float().mean().item()
        #print(f"Step {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
        losses.append(loss.item())
        accuracies.append(accuracy)

print(f"Training completed in {time.time() - start_time:.2f} seconds.")
torch.save({
    'losses': losses,
    'accuracies': accuracies,
    'logit_stds': logit_stds}, 
    f'../outputs/exp-4-2.pt')