import numpy as np
import matplotlib.pyplot as plt
import pickle
from IPython.display import clear_output

from rbm import RBM, sigmoid
from tqdm import tqdm

class DBN:
    def __init__(self, X, L, qs, use_adam=False):
        self.X = X
        self.L = L
        self.qs = qs
        self.use_adam = use_adam
        self.init_DBN()
        
    def init_DBN(self):
        self.rbms = []
        self.rbms.append(RBM(self.X, self.qs[0]))
        for i in range(1, len(self.qs)):
            # we initialize the RBMs with random X matrices
            # we will update them later on, iteratively during training
            h = self.rbms[-1].b
            self.rbms.append(
                RBM(h, self.qs[i], use_adam=self.use_adam)
            )
        
    def train_DBN(self, epochs, learning_rate, batch_size, print_error_every=-1):
        h = self.X.copy()
        total_loss = 0.
        with tqdm(self.rbms, unit='rbm') as bar:
            for i, rbm in enumerate(self.rbms):
                bar.set_description(f'RBM {i}')
                rbm.update_X(h)
                loss_rbm = rbm.train_RBM(epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, print_error_every=print_error_every)
                h, _ = rbm.entree_sortie_RBM(h)  # sigmoid(h @ rbm.W + rbm.b)
                total_loss += loss_rbm[-1]
                bar.set_postfix(total_loss=total_loss)
                bar.update(1)
                clear_output(wait=False)
            
    def generer_image_DBN(self, num_samples, gibbs_steps, ncols=10, image_size=(20, 16), plot_=True):
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
            
        if plot_:
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
    
    def save_weights(self, path):
        dict_weights = {}
        for i, rbm in enumerate(self.rbms):
            dict_weights[i] = rbm.save_weights(path=None)
            
        if path is None:
            return dict_weights
            
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dict_weights, f) 
            
    def load_weights(self, path, dict_weights=None):
        if dict_weights is None:
            with open(path, 'rb') as f:
                dict_weights = pickle.load(f)
        for i in range(len(dict_weights)):
            self.rbms[i].load_weights(path=None, dict_weights=dict_weights[i])
             