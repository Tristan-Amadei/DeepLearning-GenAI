from optimizers import AdamOptimizer, SGD

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
import pickle

import warnings
warnings.filterwarnings('ignore')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM:
    def __init__(self, X, q, use_adam=False):
        if len(X.shape) <= 1:
            X = X[np.newaxis, :]
        self.X = X
        self.p = self.X.shape[1]
        self.q = q
        self.init_rbm()
        self.use_adam = use_adam
        self.optimizer = None
        
    def update_X(self, X):
        if len(X.shape) <= 1:
            X = X[np.newaxis, :]
        self.X = X
        self.p = self.X.shape[1]
    
    def init_rbm(self):
        _, p = self.X.shape
        self.a = np.zeros(p)
        self.b = np.zeros(self.q)
        self.W = np.random.normal(loc=0, scale=1e-1, size=(p, self.q))
        
    def entree_sortie_RBM(self, v):
        probabilities = sigmoid(self.b + v @ self.W)
        values = (np.random.uniform(size=probabilities.shape) < probabilities).astype(int)
        return probabilities, values
    
    def sortie_entree_RBM(self, h):
        probabilities = sigmoid(self.a + h @ self.W.T)
        values = (np.random.uniform(size=probabilities.shape) < probabilities).astype(int)
        return probabilities, values
    
    def train_RBM(self, epochs, learning_rate, batch_size, print_error_every=1, plot_errors=False, save_every=None, save_path=None, add_epoch=True):
        n, p = self.X.shape
        
        if print_error_every is None:
            print_error_every = 1 if epochs < 10 else epochs / 10
            
        if self.use_adam:
            self.optimizer = AdamOptimizer(rbm=self, lr=learning_rate)
        else:
            self.optimizer = SGD(rbm=self, lr=learning_rate)
        
        errors = []
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n)
            X_shuffled = self.X[indices]
            
            quadratic_error = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i: i+batch_size]

                # Gibbs sampling
                probabilities_h_given_v0, h = self.entree_sortie_RBM(X_batch)
                _, v = self.sortie_entree_RBM(h)
                probabilities_h_given_v1, _ = self.entree_sortie_RBM(v)
                
                quadratic_error += np.sum((v - X_batch)**2)
                
                # Compute gradients
                grad_a = np.sum(X_batch - v, axis=0)
                grad_b = np.sum(probabilities_h_given_v0 - probabilities_h_given_v1, axis=0)
                grad_W = X_batch.T @ probabilities_h_given_v0 - v.T @ probabilities_h_given_v1
                
                # Update weights
                self.optimizer.step(grad_W=grad_W, grad_b=grad_b, grad_a=grad_a, descent=False)
            quadratic_error /= (n*p)
            errors.append(quadratic_error)
            if (epoch % print_error_every == 0 or epoch == epochs-1) and print_error_every != -1:
                print(f'Epoch {epoch}: error = {round(quadratic_error, 4)}')

            if save_every is not None and epoch % save_every == 0:
                save_path_ = save_path.split('.pkl')[0]
                save_path_ = f"{save_path_}_{epoch}.pkl" if add_epoch else f"{save_path_}.pkl"
                self.save_weights(save_path_)
                
        if plot_errors:
            plt.figure(figsize=(7, 4))
            plt.plot(errors)
            plt.grid('on')
            plt.title('MSE through gradient ascent')
            plt.show()
        return errors
    
    def generer_image_RBM(self, nb_step_gibbs, nb_to_generate, ncols=10, image_size=(20, 16), plot_=True):
        # draw each X_i according to a Bernouilli with randomized parameter
        X = (np.random.uniform(size=(nb_to_generate, self.q)) < np.random.uniform(size=(nb_to_generate, self.q))).astype(int)
        
        for _ in tqdm(range(nb_step_gibbs)):
            _, h = self.sortie_entree_RBM(X)
            _, X = self.entree_sortie_RBM(h)
        clear_output(wait=True)
            
        def plot_im(X, ax):
            ax.imshow(X.reshape(image_size), cmap='gray')
            ax.axis('off')
            
        if plot_:
            nrows = int(np.ceil(nb_to_generate/ncols))
            fig, axs = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
            if nrows == 1:
                for i, ax in enumerate(axs):
                    plot_im(h[i], ax=ax)
            else:
                for i in range(nrows):
                    for j in range(ncols):
                        plot_im(h[nrows*i+j], ax=axs[i][j])
        return X, h
    
    def save_weights(self, path):
        dict_weights = {
            'a': self.a,
            'W': self.W,
            'b': self.b
        }
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
        self.b = dict_weights['b']
        self.W = dict_weights['W']
        self.a = dict_weights['a']