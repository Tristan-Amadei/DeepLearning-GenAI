import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models
import torchvision.transforms as transforms



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

    def __init__(self, latent_dim, alpha_lrelu=0.3):
        super(Encoder, self).__init__()
        self.alpha_lrelu = alpha_lrelu

        self.lin1 = nn.Linear(784, 512)
        self.lin2 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 2*latent_dim)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.lin1(x)
        x = F.leaky_relu(x, self.alpha_lrelu)
        x = self.lin2(x)
        x = F.leaky_relu(x, self.alpha_lrelu)

        x = self.fc(x)  # no activation
        return x

class Decoder(nn.Module):

    def __init__(self, latent_dim, alpha_lrelu=0.3):
        super(Decoder, self).__init__()
        self.alpha_lrelu = alpha_lrelu

        self.lin1 = nn.Linear(latent_dim, 256)
        self.lin2 = nn.Linear(256, 512)
        self.lin3 = nn.Linear(512, 784)

    def forward(self, x):
        batch_size = x.size(0)

        x = self.lin1(x)
        x = F.leaky_relu(x, self.alpha_lrelu)
        x = self.lin2(x)
        x = F.leaky_relu(x, self.alpha_lrelu)
        x = self.lin3(x)
        x = F.sigmoid(x)

        return x

class VAEModel(nn.Module):
    def __init__(self, latent_dim, beta, alpha_lrelu=0.3):
        super(VAEModel, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.alpha_lrelu = alpha_lrelu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = Encoder(latent_dim=latent_dim, alpha_lrelu=alpha_lrelu).to(self.device)
        self.decoder = Decoder(latent_dim=latent_dim, alpha_lrelu=alpha_lrelu).to(self.device)
        self.loss = BetaVAELoss(beta=self.beta)
        
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


    def sample_pz(self, N):
        samples_pz = torch.randn(N, self.latent_dim, device=self.device)
        return samples_pz

    def generate_samples(self, samples_pz=None, N=None):
        if samples_pz is None:
            if N is None:
                return ValueError("samples_pz and N cannot be set to None at the same time. Specify one of the two.")

            # If samples z are not provided, we sample N samples from p(z)=N(0,Id)
            samples_pz = self.sample_pz(N)

        generations = self.decoder(samples_pz)
        return {'generations': generations}

    def train_vae(self, X_train, epochs, learning_rate=1e-3, batch_size=64, 
                    optimizer=None, print_error_every=1, plot_errors=True, patience=50):
        if optimizer is None:
            #self.optimizer = torch.optim.AdamW(self.parameters(), 
            #                        weight_decay=learning_rate/10, 
            #                        lr=learning_rate)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        tensor_X = torch.Tensor(X_train) 
        dataset = TensorDataset(tensor_X) 
        dataloader = DataLoader(dataset, batch_size=batch_size)

        train_losses = []
        best_loss = np.inf
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = 0.
            for data in dataloader:
                self.optimizer.zero_grad()
                data = data[0].to(self.device)

                predict = self(data, mode='mean')
                reconstructions = predict['reconstructions']
                stats_qzx = predict['stats_qzx']

                loss = self.loss(data, reconstructions, stats_qzx)
                train_loss += loss.item()

                # Backpropagate
                loss.backward()
                self.optimizer.step()

            train_loss /= dataloader.dataset.tensors[0].shape[0]
            train_losses.append(train_loss)
            if train_loss <= best_loss:
                best_loss = train_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    print("##### PATIENCE TRIGGERED!")
                    print(f'Epoch {epoch}: error = {round(train_loss, 4)}')
                    break
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

    def generate_data(self, nb_to_generate=20, ncols=10, plot_=True, binary=True):
        generated_images = self.generate_samples(N=nb_to_generate)['generations']
        generated_images = generated_images.detach().cpu().numpy().reshape(nb_to_generate, 28, 28)
        if binary:
            generated_images = np.where(
                generated_images >= 0.5, 1, 0
                )

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

    def compute_fid_score(self, X_test, batch_size=32):
        N = len(X_test)
        generated_images = self.generate_data(nb_to_generate=N, plot_=False)
        return compute_fid_score(X_test, generated_images, self.device, batch_size=batch_size)


def compute_fid_score(X_test, generated_images, device, batch_size=32):
    inception_model = models.inception_v3(weights='IMAGENET1K_V1').to(device)
    inception_model.fc = nn.Identity()  # we cannot directly remove the classification layer from Inception
                                        # we artificially remove it by replacing it by an identity layer

    class GrayscaleToRgb:
        def __call__(self, tensor):
            # tensor is a 1-channel grayscale image
            return tensor.repeat(3, 1, 1)

    transform = transforms.Compose([
        GrayscaleToRgb(),
        transforms.Resize((342, 342), antialias=True),
        transforms.CenterCrop((299, 299)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    def preprocess_images(images):
        return torch.stack([transform(image) for image in images])

    def pass_through_inception(loader):
        inception_embeddings = np.zeros((len(X_test), 2048))  # Inception embeddings contain 2048 elements
                                                              # they correspond to the output of the final convolutional layer, just after the global pooling
        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data[0].to(device)
                inception_images = preprocess_images(data)
                embeddings = inception_model(inception_images).logits.cpu().numpy()
                inception_embeddings[i*batch_size: i*batch_size+data.size(0)] = embeddings
        return inception_embeddings

    if len(X_test.shape) <= 2:
        X_test = X_test.reshape(-1, 1, 28, 28)

    tensor_X = torch.Tensor(X_test) 
    dataset = TensorDataset(tensor_X) 
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Pass real test images through InceptionV3
    real_embeddings = pass_through_inception(dataloader)
    
    torch.cuda.empty_cache()

    gen_dataset = TensorDataset(torch.Tensor(generated_images))
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size, shuffle=False)
    # Pass generated images through InceptionV3
    generated_embeddings = pass_through_inception(gen_loader)

    real_mean, real_cov = np.mean(real_embeddings, axis=0), np.cov(real_embeddings, rowvar=True)
    gen_mean, gen_cov = np.mean(generated_embeddings, axis=0), np.cov(generated_embeddings, rowvar=True)

    # Calculate Fréchet distance
    mean_difference = np.linalg.norm(real_mean - gen_mean, ord=2)
    offset = np.eye(real_cov.shape[0]) * 1e-6
    cov_sqrt, _ = sqrtm((real_cov+offset) @ (gen_cov+offset), disp=False)
    trace = np.trace(real_cov + gen_cov - 2*cov_sqrt.real)

    return mean_difference + trace