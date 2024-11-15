import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from vae import VAE  # Assuming VAE is defined in vae.py

def vae_loss(reconstruction, x, mu, log_sigma):
    # Binary Cross-Entropy loss for reconstruction error since bernoulli case
    recon_loss = F.binary_cross_entropy(reconstruction, x, reduction='sum')
    # KL Divergence loss for regularization term, using mean and log_sigma
    kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    # Return the sum of reconstruction loss and KL divergence
    return (recon_loss + kl_div) / x.size(0)

def main():

    dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)

    # just some basic values for proof of concept
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    input_dim = 784      
    latent_dim = 20      
    hidden_dim = 200     
    learning_rate = 1e-3
    epochs = 10       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = VAE(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = Adam(vae.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for x_batch, labels in dataloader:
            x_batch = x_batch.view(-1, input_dim).to(device)
            
            optimizer.zero_grad()
            
            reconstruction, mu, log_sigma = vae(x_batch)
            
            # Compute VAE loss for current batch
            loss = vae_loss(reconstruction, x_batch, mu, log_sigma)
            
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

if __name__ == "__main__":
    main()