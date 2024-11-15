from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        # transform the input into a latent space dimension
        self.img_to_hidden = nn.Linear(input_dim, hidden_dim)
        
        # approximate the mean and the log of the standard deviation
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_sigma = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # q_phi(z|x) which is the approximate posterior of the true posterior p_theta(z|x)
        x = self.img_to_hidden(x)
        x = torch.relu(x) # guess we need non-linearity? not mentioned in the paper
        
        # get the mean and the log of the standard deviation
        mu = self.mu(x)
        log_sigma = self.log_sigma(x)
        
        return mu, log_sigma # sample from the distribution
        
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        # p_theta(x|z)
        
        # transform the latent space into a hidden space
        self.hidden_to_img = nn.Linear(latent_dim, hidden_dim)
        
        # transform the hidden space into a output space
        self.img = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.hidden_to_img(x)
        x = torch.relu(x)
        
        x = torch.sigmoid(self.img(x)) # assuming binary data (ex for mnist, pixel value scaled [0,1])
        return x
    
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, log_sigma):
        # since z is originally a random variable, we need to reparameterize it to make it a deterministic variable for backprop
        std = torch.exp(log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.reparameterize(mu, log_sigma)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_sigma
