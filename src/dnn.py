import numpy as np
import matplotlib.pyplot as plt
import pickle

from rbm import RBM, sigmoid
from dbn import DBN
from optimizers import AdamOptimizer, SGD

class DNN:
    def __init__(self, X_train, y_train, num_classes, num_hidden_layers, neurons, X_val=None, y_val=None, use_adam=False):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.use_adam = use_adam
        self.init_DNN()
        
    def init_DNN(self):
        self.dbn = DBN(self.X_train, self.num_hidden_layers, self.neurons, use_adam=self.use_adam)
        rbm_classification = RBM(self.dbn.rbms[-1].b, self.num_classes, use_adam=self.use_adam)
        self.rbm_classification = rbm_classification
        
    def pretrain_DNN(self, epochs, learning_rate, batch_size):
        self.dbn.train_DBN(epochs, learning_rate, batch_size)
    
    @staticmethod
    def calcul_softmax(X, rbm):
        logits = rbm.b + X @ rbm.W
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return logits, probabilities
    
    def entree_sortie_reseau(self, X):
        outputs = [X]
        
        h = X
        for rbm in self.dbn.rbms:
            output, _ = rbm.entree_sortie_RBM(h)
            outputs.append(output)
            h = output
        
        output_classif_layer, proba_classif_layer = self.calcul_softmax(h, self.rbm_classification)
        outputs.append(output_classif_layer)
        return outputs, proba_classif_layer
    
    @staticmethod
    def cross_entropy(labels, proba):
        return -(np.array(labels) * np.log(proba)).sum()
    
    def test_DNN(self, X, y, batch_size=32):
        n = X.shape[0]
        # Shuffle the data
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        loss = 0.
        nb_well_classified = 0
        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i: i+batch_size]
            y_batch = y_shuffled[i: i+batch_size]
            
            outputs, classification_probabilities = self.entree_sortie_reseau(X_batch)
            loss_batch = self.cross_entropy(y_batch, classification_probabilities)
            loss += loss_batch
            nb_well_classified += (classification_probabilities.argmax(axis=1) == y_batch.argmax(axis=1)).sum()
        loss /= n
        accuracy = nb_well_classified / n
        return loss, accuracy   
    
    def compute_and_plot_metrics(self, loss, accuracy, val_losses, val_accuracies, min_val_loss, patience_counter, 
                                 patience, epoch, epochs, print_error_every):
        patience_trigger = False 
        
        if self.X_val is not None and self.y_val is not None:
            val_loss, val_acc = self.test_DNN(self.X_val, self.y_val)
            val_losses.append(val_loss)
            val_accuracies.append(val_acc)
            if val_loss <= min_val_loss:
                min_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > patience:
                    patience_trigger = True
            if (epoch % print_error_every == 0 or epoch == epochs-1) and print_error_every != -1:
                print(f'Epoch {epoch}:: loss: {round(loss, 4)}, val_loss: {round(val_loss, 4)} ; '
                      f'accuracy: {round(accuracy*100, 2)}%, val_accuracy: {round(val_acc*100, 2)}%')
        elif (epoch % print_error_every == 0 or epoch == epochs-1) and print_error_every != -1:
            print(f'Epoch {epoch}:: loss: {round(loss, 4)}, accuracy: {round(accuracy*100, 2)}%')
        
        return patience_trigger, min_val_loss, patience_counter
        
    @staticmethod
    def plot_loss_acc(losses, accs, labels, suptitle=''):
        def plot(x, ax, label, title=''):
            ax.plot(x, label=label)
            ax.grid('on')
            ax.legend()
            ax.set_title(title)
        
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for i, l in enumerate(losses):
            plot(l, axs[0], label=labels[i], title='Loss')
        for i, acc in enumerate(accs):
            plot(acc, axs[1], label=labels[i], title='Accuracy')
        plt.suptitle(suptitle)
        plt.show()
        
    def init_optimizers(self, learning_rate):
        for rbm in self.dbn.rbms + [self.rbm_classification]:
            if rbm.optimizer is None:
                if self.use_adam:
                    rbm.optimizer = AdamOptimizer(rbm=rbm, lr=learning_rate)
                else:
                    rbm.optimizer = SGD(rbm=rbm, lr=learning_rate)
    
    def retropropagation(self, epochs, learning_rate, batch_size, print_error_every=None, 
                         plot_=False, patience=np.inf, suptitle=''):
        n, _ = self.X_train.shape
        
        if print_error_every is None:
            print_error_every = 1 if epochs < 10 else epochs / 10
            
        self.init_optimizers(learning_rate=learning_rate)
        
        min_val_loss = np.inf
        patience_counter = 0
        
        losses, accuracies = [], []
        val_losses, val_accuracies = [], []
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            loss = 0.
            nb_well_classified = 0
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i: i+batch_size]
                y_batch = y_shuffled[i: i+batch_size]
                
                outputs, classification_probabilities = self.entree_sortie_reseau(X_batch)
                loss_batch = self.cross_entropy(y_batch, classification_probabilities)
                loss += loss_batch
                nb_well_classified += (classification_probabilities.argmax(axis=1) == y_batch.argmax(axis=1)).sum()
                
                ### Gradient RBM classification ###
                proba_diff_clf = classification_probabilities - y_batch
                grad_W_clf = outputs[-2].T @ proba_diff_clf
                grad_b_clf = np.sum(proba_diff_clf, axis=0)
                W_plus_one = self.rbm_classification.W.copy() # copy it before gradient descent as it is needed later on
                self.rbm_classification.W -= learning_rate * grad_W_clf
                self.rbm_classification.b -= learning_rate * grad_b_clf
                
                ### Gradient hidden RBMs ###
                c_plus_one = proba_diff_clf
                for p in range(len(self.dbn.rbms)-1, -1, -1):
                    rbm = self.dbn.rbms[p]
                    x_p_minus_one = outputs[p] # it is already shifted by 1 by default with the presence of rbm_classification
                    x_p = outputs[p+1]
                    
                    c_p = (c_plus_one @ W_plus_one.T) * (x_p * (1-x_p))
                    W_plus_one = rbm.W.copy()
                    c_plus_one = c_p
                    
                    grad_W = x_p_minus_one.T @ c_p
                    grad_b = np.sum(c_p, axis=0)
                    #rbm.W -= learning_rate * grad_W
                    #rbm.b -= learning_rate * grad_b
                    rbm.optimizer.step(grad_W=grad_W, grad_b=grad_b, grad_a=None, descent=True)
                
            loss /= n
            losses.append(loss)
            accuracy = nb_well_classified / n
            accuracies.append(accuracy)

            patience_trigger, min_val_loss, patience_counter= self.compute_and_plot_metrics(loss, accuracy, val_losses, val_accuracies, 
                                                min_val_loss, patience_counter, patience, epoch, epochs, print_error_every)
            if patience_trigger:
                print(f'##### Patience triggered at epoch {epoch}! #####')
                _, _, _= self.compute_and_plot_metrics(loss, accuracy, val_losses, val_accuracies, 
                            min_val_loss, patience_counter, patience, epoch, epochs, epoch)
                break

        val_losses, val_accuracies = val_losses[:len(losses)], val_accuracies[:len(accuracies)] # this is because if patience is triggered, the last validation values are added twice to the lists
        if plot_:
            self.plot_loss_acc([losses, val_losses], [accuracies, val_accuracies], labels=['Train', 'Test'], suptitle=suptitle)
        return losses, accuracies, val_losses, val_accuracies
    
    def save_weights(self, path):
        dict_weights = self.dbn.save_weights(path=None)
        dict_weights['classif'] = self.rbm_classification.save_weights(path=None)
        
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
        self.rbm_classification.load_weights(path=None, dict_weights=dict_weights.pop('classif'))
        self.dbn.load_weights(path=None, dict_weights=dict_weights)