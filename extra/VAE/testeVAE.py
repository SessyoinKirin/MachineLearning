# %%
import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image, make_grid

# %%
dataset_path = '~/datasets'

DEVICE = torch.device( "cpu")
batch_size = 100

x_dim = 784
hidden_dim = 400
latent_dim = 20

lr = 1e-3
num_epochs = 30

# %%
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

mnist_transform = transforms.Compose([
    transforms.ToTensor(),
])

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

train_dataset = MNIST(dataset_path, train=True, transform=mnist_transform, download=True)
test_dataset = MNIST(dataset_path, train=False, transform=mnist_transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **kwargs)

# %%
'''
implementation of Gaussian MLP 
'''

class Encoder(nn.Module):
    def __init__(self, x_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc3_logvar = nn.Linear(hidden_dim, latent_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.training = True
    
    def forward(self, x):
        h_ = self.LeakyReLU(self.fc1(x))
        h_ = self.LeakyReLU(self.fc2(h_))
        mean = self.fc3_mean(h_)
        logvar = self.fc3_logvar(h_)
        return mean, logvar

# %%
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, y_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, y_dim)

        self.LeakyReLU = nn.LeakyReLU(0.2)
        # self.Sigmoid = nn.Sigmoid()
        # qual a diferença por sigmoid aqui e no forward?
    
    def forward(self, z):
        h = self.LeakyReLU(self.fc1(z))
        h = self.LeakyReLU(self.fc2(h))
        y_hat = torch.sigmoid(self.fc3(h))
        return y_hat

# %%
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    
    def reparameterize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(DEVICE) 
        z = mean + logvar * epsilon
        return z
    
    def forward(self, x):
        mean, logvar = self.Encoder(x)
        z = self.reparameterize(mean, torch.exp(0.5 * logvar))
        y_hat = self.Decoder(z)
        return y_hat, mean, logvar

# %%
encoder = Encoder(x_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, x_dim)

model = Model(encoder, decoder).to(DEVICE)

# %%
from torch.optim import Adam

BCE_loss = nn.BCELoss()

def loss_(y, y_hat, mean, logvar):
    reproduction_loss = nn.functional.binary_cross_entropy(y_hat, y, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return reproduction_loss + KLD

optimizer = Adam(model.parameters(), lr=lr)

# %%
print("Starting training VAE...")
model.train()

for epoch in range(num_epochs):
    overall_loss = 0
    for batch_idx, (x, _) in enumerate(train_loader):
        x = x.view(-1, x_dim)

        optimizer.zero_grad()

        y_hat, mean, logvar = model(x)

        # Printando a média e o logvar do primeiro item do batch
        if batch_idx == 0: # Printa apenas no primeiro batch para não inundar a tela
            print(f"Média (primeiros 5 valores do vetor): {mean[0][:5].detach().cpu().numpy()}")
            print(f"LogVar (primeiros 5 valores do vetor): {logvar[0][:5].detach().cpu().numpy()}")

        loss = loss_(x, y_hat, mean, logvar)

        overall_loss += loss.item()
        loss.backward()
        optimizer.step()

        print("\tEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
            epoch, batch_idx * len(x), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), overall_loss / len(train_loader.dataset)))
        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / ((batch_idx+1)*batch_size))
    
print("Finish!!")

# %%

import matplotlib.pyplot as plt

# %%
model.eval()

with torch.no_grad():
    for batch_idx, (x, _) in enumerate(tqdm(test_loader)):
        x = x.view(batch_size, x_dim).to(DEVICE)
        y_hat, _, _ = model(x)


        break

# %%
def show_image(x, idx):
    x = x.view(batch_size, 28, 28)

    grid_size = int(np.sqrt(idx))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten() # Transforma a matriz de eixos em uma lista simples

    for i in range(idx):
        if i < len(x):
            axes[i].imshow(x[i], cmap='gray')
            axes[i].axis('off') # Remove os números dos eixos
        else:
            axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# %%
show_image(x, idx=25)

# %%
show_image(y_hat, idx=25)


