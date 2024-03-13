import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = np.expand_dims(sample, axis=1)  # Add channel dimension
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)


def sample_images(generator, z_dim, device, r=5, c=5, binarize=True):
    z_random = torch.randn(r * c, z_dim, dtype=torch.float, device=device)
    gen_imgs = np.transpose(generator(z_random).cpu().detach().numpy(), (0, 2, 3, 1))

    # Rescale images 0 - 1
    if binarize:
        gen_imgs = np.where(gen_imgs > 0.5, 1, 0)

    fig, axs = plt.subplots(r, c, figsize=(2*c, 2*r))
    cnt = 0
    for i in range(r):
        for j in range(c):
            # black and white images
            if (gen_imgs.shape[3] == 1):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            elif (gen_imgs.shape[3] == 3):   # color images
                gen_imgs_temp = gen_imgs.copy()
                gen_imgs_temp = 255.*np.clip(gen_imgs_temp, 0., 1.)
                axs[i, j].imshow(gen_imgs_temp[cnt, :, :, :].astype(np.uint8))
            else:
                print('Error, unsupported channel size.')
            axs[i, j].axis('off')
            cnt += 1
    plt.tight_layout()
    plt.show()
    return gen_imgs


class Generator(nn.Module):
    def __init__(self, z_dim, h_dim_1, h_dim_2, n_rows, n_cols, n_channels, negative_slope=0.2):
        super(Generator, self).__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = (self.n_rows) * (self.n_cols)
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.z_dim = z_dim
        self.negative_slope = negative_slope

        self.fc1 = nn.Linear(z_dim, h_dim_1)
        self.fc2 = nn.Linear(h_dim_1, h_dim_2)
        self.fc3 = nn.Linear(h_dim_2, self.n_pixels)

    def forward(self, z):
        y = F.leaky_relu(self.fc1(z), negative_slope=self.negative_slope)
        y = F.leaky_relu(self.fc2(y), negative_slope=self.negative_slope)
        y = F.tanh(self.fc3(y))

        y = y.view(y.size(0), self.n_channels, self.n_rows, self.n_cols)
        return y


class Discriminator(nn.Module):
    def __init__(self, h_dim_2, h_dim_1, z_dim, n_rows, n_cols, n_channels, negative_slope=0.2):
        super(Discriminator, self).__init__()

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_channels = n_channels
        self.n_pixels = (self.n_rows) * (self.n_cols)
        self.h_dim_1 = h_dim_1
        self.h_dim_2 = h_dim_2
        self.z_dim = z_dim
        self.negative_slope = negative_slope

        self.fc1 = nn.Linear(self.n_pixels, h_dim_2)
        self.fc2 = nn.Linear(h_dim_2, h_dim_1)
        self.fc3 = nn.Linear(h_dim_1, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = F.leaky_relu(self.fc1(x), negative_slope=self.negative_slope)
        y = F.leaky_relu(self.fc2(y), negative_slope=self.negative_slope)
        y = F.sigmoid(self.fc3(y))
        return y


class GAN(nn.Module):
    def __init__(self, latent_dim, h_dim_1=256, h_dim_2=512, negative_slope=0.2):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.h_dim1 = h_dim_1
        self.h_dim2 = h_dim_2
        self.negative_slope = negative_slope
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gen_model = Generator(latent_dim, h_dim_1, h_dim_2, 28, 28, 1,
                                   negative_slope=negative_slope).to(self.device)
        self.disc_model = Discriminator(h_dim_2, h_dim_1, latent_dim, 28, 28, 1,
                                        negative_slope=negative_slope).to(self.device)
        self.bce_criterion = nn.BCELoss()

    def train(self, X_train, epochs, learning_rate, beta_1=0.5,
              sample_interval=1, n_iters_inner=1, batch_size=64,
              plot_every=1):

        X_train = np.array(X_train, dtype=np.float32)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))
                                        ])
        custom_dataset = CustomDataset(X_train, transform=transform)
        dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

        optimizer_disc = optim.Adam(self.disc_model.parameters(),
                                    lr=learning_rate, betas=(beta_1, 0.999))
        optimizer_gen = optim.Adam(self.gen_model.parameters(),
                                   lr=learning_rate, betas=(beta_1, 0.999))

        G_losses, D_losses = [], []

        print("Starting Training")
        for epoch in range(epochs):
            for i, data in enumerate(dataloader, 0):
                d_loss_total = 0.
                for iter_inner in range(n_iters_inner):

                    ############################
                    # Train discriminator
                    ############################

                    # Train with true data batch
                    self.disc_model.zero_grad()

                    true_imgs = data.to(self.device)
                    d_output_true = self.disc_model(true_imgs)
                    true_labels = torch.ones_like(d_output_true, device=self.device)
                    d_loss_true = self.bce_criterion(d_output_true, true_labels)
                    d_loss_true.backward()
                    disc_true_value = d_output_true.mean().item()

                    # Train with fake data batch
                    z_latent_noise = torch.randn(true_imgs.size(0), self.latent_dim,
                                                 device=self.device)
                    fake_imgs = self.gen_model(z_latent_noise)
                    disc_output_fake = self.disc_model(fake_imgs.detach())
                    fake_labels = torch.zeros_like(disc_output_fake, device=self.device)
                    disc_loss_fake = self.bce_criterion(disc_output_fake, fake_labels)
                    disc_loss_fake.backward()
                    disc_fake_value = disc_output_fake.mean().item()
                    optimizer_disc.step()

                    d_loss_total += d_loss_true + disc_loss_fake

                ############################
                # Train generator
                ############################
                self.gen_model.zero_grad()

                z_latent_noise = torch.randn(true_imgs.size(0), self.latent_dim,
                                             device=self.device)
                fake_imgs = self.gen_model(z_latent_noise)
                disc_gen_output_fake = self.disc_model(fake_imgs)
                wrong_labels_gen = torch.ones_like(disc_gen_output_fake, device=self.device)
                g_loss = self.bce_criterion(disc_gen_output_fake, wrong_labels_gen)
                g_loss.backward()
                optimizer_gen.step()

                # Save Losses for plotting later
                G_losses.append(g_loss.item())
                D_losses.append(d_loss_total.item())

            if (epoch % plot_every == 0 or epoch == epochs - 1) and plot_every != -1:
                print(f'Epoch {epoch+1}/{epochs}:: loss_disc: {d_loss_total.item():.4f}, '
                      f'loss_gen: {g_loss.item():.4f}, D(x): {disc_true_value:.4f}, '
                      f'D(G(z)): {disc_fake_value:.4f}')

            if epoch % sample_interval == 0:
                sample_images(self.gen_model, self.latent_dim, self.device,
                              r=2, c=10, binarize=False)
                clear_output(wait=True)

    def generate_data(self, nb_to_generate=20, ncols=10, nrows=2, plot_=True, binarize=True):
        if plot_:
            return sample_images(self.gen_model, self.latent_dim, self.device,
                                 r=nrows, c=ncols, binarize=binarize)

        z_random = torch.randn(nb_to_generate, 1, self.latent_dim, dtype=torch.float,
                               device=self.device)
        gen_imgs = np.transpose(self.gen_model(z_random).cpu().detach().numpy(), (0, 2, 3, 1))
        if binarize:
            gen_imgs = np.where(gen_imgs > 0.5, 1, 0)
        return gen_imgs
