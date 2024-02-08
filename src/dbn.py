import numpy as np
import matplotlib.pyplot as plt

from rbm import RBM, sigmoid
from tqdm import tqdm

class DBN:
    def __init__(self, X, L, qs):
        self.X = X
        self.L = L
        self.qs = qs
        self.init_DBN()
        
    def init_DBN(self):
        self.rbms = []
        self.rbms.append(RBM(self.X, self.qs[0]))
        for i in range(1, len(self.qs)):
            # we initialize the RBMs with random X matrices
            # we will update them later on, iteratively during training
            h = self.rbms[-1].b
            self.rbms.append(
                RBM(h, self.qs[i])
            )
        
    def train_DBN(self, epochs, learning_rate, batch_size):
        h = self.X.copy()
        for rbm in tqdm(self.rbms):
            rbm.update_X(h)
            rbm.train_RBM(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, print_error_every=-1)
            h, _ = rbm.entree_sortie_RBM(h)  # sigmoid(h @ rbm.W + rbm.b)
            
    def generer_image_DBN(self, num_samples, gibbs_steps, ncols=10, image_size=(20, 16)):
        # Start with a random input for the topmost RBM
        top_rbm = self.rbms[-1]
        h_L = (np.random.uniform(size=(num_samples, top_rbm.q)) < np.random.uniform(size=(num_samples, top_rbm.q))).astype(int)

        for _ in range(gibbs_steps):
            _, h = top_rbm.sortie_entree_RBM(h_L)
            _, h_L = top_rbm.entree_sortie_RBM(h)

        for rbm in reversed(self.rbms[:-1]):  
            _, h = rbm.sortie_entree_RBM(h)

        generated_data = h  # The final state of 'h' is the generated visible data
        def plot_im(X, ax):
            ax.imshow(X.reshape(image_size), cmap='gray')
            ax.axis('off')
            
        nrows = num_samples // ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
        if nrows == 1:
            for i, ax in enumerate(axs):
                plot_im(generated_data[i], ax=ax)
        else:
            for i in range(nrows):
                for j in range(ncols):
                    plot_im(generated_data[nrows*i+j], ax=axs[i][j])
        return generated_data        