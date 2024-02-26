import matplotlib.pyplot as plt
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader



def reconstruction_loss(data, reconstructions):

    loss = F.binary_cross_entropy(reconstructions, data, reduction='sum')
    return loss

def kl_normal_loss(mean, logvar, batch_mean=True, component_sum=True):
    """
    Calculates the KL divergence between a normal distribution
    with diagonal covariance and a unit normal distribution.

    Parameters
    ----------
    mean : torch.Tensor
        Mean of the normal distribution. Shape (batch_size, latent_dim) where
        D is dimension of distribution.

    logvar : torch.Tensor
        Diagonal log variance of the normal distribution. Shape (batch_size,
        latent_dim)

    batch_mean: boolean
        If false, returns a separate value for each data point, if true takes
        the mean over data points

    component_sum: boolean
        If false, returns a separate value for each latent dimension, if true
        takes the sum over latent dimensions
    """
    latent_kl = 0.5 * torch.sum(mean.pow(2) + logvar.exp() - 1 - logvar)

    if batch_mean:
        latent_kl = latent_kl.mean(dim=0)

    if component_sum:
        latent_kl = latent_kl.sum(dim=-1)

    return latent_kl

class BetaVAELoss(object):

    def __init__(self, beta):
        self.beta = beta

    def __call__(self, data, reconstructions, stats_qzx):
        # Reconstruction loss
        rec_loss = reconstruction_loss(data, reconstructions)

        latent_dim = stats_qzx.size()[-1] // 2
        mean = stats_qzx[:, :latent_dim]
        logvar = stats_qzx[:, latent_dim:]
        # KL loss
        kl_loss = kl_normal_loss(mean, logvar)

        # Total loss of beta-VAE
        loss = rec_loss + self.beta * kl_loss

        return loss


class Encoder(nn.Module):

    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        # Fully connected layers
        self.lin1 = nn.Linear(64 * 6 * 6, 256)
        self.lin2 = nn.Linear(256, 2 * self.latent_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        x = x.view(batch_size, -1)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)  # no activation
        return x

class Decoder(nn.Module):

    def __init__(self, latent_dim):
        super(Decoder, self).__init__()

        self.lin1 = nn.Linear(latent_dim, 256)
        self.lin2 = nn.Linear(256, 64 * 6 * 6)

        self.convT1 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0)
        self.convT2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0)
        self.convT3 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)

        x = x.view(-1, 64, 6, 6)

        x = self.convT1(x)
        x = F.relu(x)
        x = self.convT2(x)
        x = F.relu(x)
        x = self.convT3(x)
        x = F.sigmoid(x)
        return x

class VAEModel(nn.Module):
    def __init__(self, latent_dim, beta):
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.encoder = Encoder(latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim)
        self.loss = BetaVAELoss(beta=self.beta)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reparameterize(self, mean, logvar, mode='sample'):
        """
        Samples from a normal distribution using the reparameterization trick.

        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim)

        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)

        mode : 'sample' or 'mean'
            Returns either a sample from qzx or the mean of qzx. The former is
            useful at training time. The latter is useful at inference time as
            the mean is usually used for reconstruction, rather than a sample.
        """
        if mode=='sample':
            # Implements the reparametrization trick (slide 43):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps
        elif mode=='mean':
            return mean
        else:
            return ValueError("Unknown mode: {mode}".format(mode))

    def forward(self, x, mode='sample'):
        """
        Forward pass of model, used for training or reconstruction.

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)

        mode : 'sample' or 'mean'
            Reconstructs from either a sample from qzx or the mean of qzx
        """

        # stats_qzx is the output of the encoder
        stats_qzx = self.encoder(x)

        mean = stats_qzx[:, :self.latent_dim]
        logvar = stats_qzx[:, self.latent_dim:]
        # Use the reparametrization trick to sample from q(z|x)
        samples_qzx = self.reparameterize(mean, logvar, mode=mode)

        # Decode the samples to image space
        reconstructions = self.decoder(samples_qzx)

        # Return everything:
        return {
            'reconstructions': reconstructions,
            'stats_qzx': stats_qzx,
            'samples_qzx': samples_qzx}

    def sample_qzx(self, x):
        """
        Returns a sample z from the latent distribution q(z|x).

        Parameters
        ----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        stats_qzx = self.encoder(x)
        samples_qzx = self.reparameterize(*stats_qzx.unbind(-1))
        return samples_qzx

    def sample_pz(self, N):
        samples_pz = torch.randn(N, self.latent_dim, device=self.encoder.conv1.weight.device)
        return samples_pz

    def generate_samples(self, samples_pz=None, N=None):
        if samples_pz is None:
            if N is None:
                return ValueError("samples_pz and N cannot be set to None at the same time. Specify one of the two.")

            # If samples z are not provided, we sample N samples from the prior
            # p(z)=N(0,Id), using sample_pz
            samples_pz = self.sample_pz(N)

        # Decode the z's to obtain samples in image space (here, probability
        # maps which can later be sampled from or thresholded)
        generations = self.decoder(samples_pz)
        return {'generations': generations}

    def train_vae(self, X_train, epochs, learning_rate=1e-3, batch_size=64, 
                    optimizer=None, print_error_every=1, plot_errors=True):
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(self.parameters(), 
                                    weight_decay=learning_rate/10, 
                                    lr=learning_rate)

        if len(X_train.shape) <= 2:
            X_train = X_train.reshape(-1, 1, 28, 28)

        tensor_X = torch.Tensor(X_train) 
        dataset = TensorDataset(tensor_X) 
        dataloader = DataLoader(dataset, batch_size=batch_size)

        train_losses = []
        for epoch in range(epochs):
            train_loss = 0.
            for data in dataloader:
                self.optimizer.zero_grad()
                data = data[0].to(self.device)

                predict = self(data)
                reconstructions = predict['reconstructions']
                stats_qzx = predict['stats_qzx']

                loss = self.loss(data, reconstructions, stats_qzx)
                train_loss += loss.item()

                # Backpropagate
                loss.backward()
                self.optimizer.step()

            train_loss /= dataloader.dataset.tensors[0].shape[0]
            train_losses.append(train_loss)
            if (epoch % print_error_every == 0 or epoch == epochs-1) and print_error_every != -1:
                print(f'Epoch {epoch}: error = {round(train_loss, 4)}')
                
        if plot_errors:
            plt.figure(figsize=(7, 4))
            plt.plot(train_losses)
            plt.grid('on')
            plt.title('Beta-VAE Loss')
            plt.show()
        return train_losses

    def test_vae(self, X_test, plot_=True, nb_to_plot=10, ncols=5):
        if len(X_test.shape) <= 2:
            X_test = X_test.reshape(-1, 1, 28, 28)

        tensor_X = torch.Tensor(X_test) 
        dataset = TensorDataset(tensor_X) 
        dataloader = DataLoader(dataset, batch_size=32)

        test_reconstructions = np.zeros((len(X_test), 28, 28), dtype=np.uint)

        self.eval()
        with torch.no_grad():
            total_loss = 0.
            n = 0
            for data in dataloader:
                data = data[0].to(self.device)

                predict = self(data, mode='mean')
                reconstructions = predict['reconstructions']
                stats_qzx = predict['stats_qzx']

                loss = self.loss(data, reconstructions, stats_qzx)
                total_loss += loss.item()

                batch_size = stats_qzx.shape[0]
                test_reconstructions[n:(n+batch_size), :, :] = np.where(
                    reconstructions.detach().cpu().numpy().reshape(batch_size, 28, 28) >= 0.5, 
                    1, 0)
                #test_reconstructions[n:(n+batch_size), :, :] = reconstructions.detach().cpu().numpy().reshape(batch_size, 28, 28)
                n += batch_size

            total_loss /= len(X_test)
            print(f'Test loss = {round(total_loss, 4)}')

            if plot_:

                def plot_im(X, ax):
                    ax.imshow(X, cmap='gray')
                    ax.axis('off')

                idx_to_plot = np.random.choice(len(X_test), size=nb_to_plot, replace=False)
                nrows = int(np.ceil(nb_to_plot/ncols))
                fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
                if nrows == 1:
                    for i, ax in enumerate(axs):
                        id = idx_to_plot[i]
                        plot_im(test_reconstructions[id], ax=ax)
                else:
                    for i in range(nrows):
                        for j in range(ncols):
                            id = idx_to_plot[nrows*i+j]
                            plot_im(test_reconstructions[id], ax=axs[i][j])
            return loss, test_reconstructions

    def generate_data(self, nb_to_generate=20, ncols=10, plot_=True):
        generated_images = self.generate_samples(N=nb_to_generate)['generations']
        generated_images = np.where(
            generated_images.detach().cpu().numpy().reshape(nb_to_generate, 28, 28) >= 0.5, 
            1, 0)

        if plot_:
            def plot_im(X, ax):
                ax.imshow(X, cmap='gray')
                ax.axis('off')

            nrows = int(np.ceil(nb_to_generate/ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
            if nrows == 1:
                for i, ax in enumerate(axs):
                    plot_im(generated_images[i], ax=ax)
            else:
                for i in range(nrows):
                    for j in range(ncols):
                        plot_im(generated_images[nrows*i+j], ax=axs[i][j])
        return generated_images
