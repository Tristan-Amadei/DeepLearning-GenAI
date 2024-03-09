import numpy as np
import matplotlib.pyplot as plt
import pickle

from rbm import RBM
from dbn import DBN
from optimizers import AdamOptimizer, SGD


class DNN:
    """Deep Neural Network (DNN) class using stacked Restricted Boltzmann Machines (RBMs)
    for pre-training and a final RBM layer for classification.

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_features)
        Training data.
    y_train : ndarray, shape (n_samples, num_classes)
        Training labels (one-hot encoded class labels).
    num_classes : int
        Number of output classes.
    num_hidden_layers : int
        Number of hidden RBM layers in the DBN (not including the classification layer).
    neurons : list, length num_hidden_layers
        List of hidden unit sizes for each RBM layer (not including the classification layer).
    X_val : ndarray, shape (n_val_samples, n_features), optional
        Validation data for early stopping (default: None).
    y_val : ndarray, shape (n_val_samples, num_classes), optional
        Validation labels (default: None).
    use_adam : bool, default False
        Whether to use Adam optimizer for training RBMs. If False, SGD is used.

    Attributes
    ----------
    X_train : ndarray, shape (n_samples, n_features)
        Training data.
    y_train : ndarray, shape (n_samples, num_classes)
        Training labels.
    X_val : ndarray, shape (n_val_samples, n_features), optional
        Validation data (if provided).
    y_val : ndarray, shape (n_val_samples, num_classes), optional
        Validation labels (if provided).
    num_classes : int
        Number of output classes.
    num_hidden_layers : int
        Number of hidden RBM layers.
    neurons : list, length num_hidden_layers
        Hidden unit sizes for each RBM layer (not including classification).
    use_adam : bool
        Whether Adam optimizer is used.
    dbn : DBN object
        Deep Belief Network used for pre-training.
    rbm_classification : RBM object
        Topmost RBM layer for classification.
    """

    def __init__(self, X_train, y_train, num_classes, num_hidden_layers, neurons,
                 X_val=None, y_val=None, use_adam=False):
        """Initialize a DNN object."""

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
        """Randomly initialize the DNN layers (DBN and classification RBM)."""

        self.dbn = DBN(self.X_train, self.num_hidden_layers, self.neurons, use_adam=False)
        rbm_classification = RBM(self.dbn.rbms[-1].b, self.num_classes, use_adam=False)
        self.rbm_classification = rbm_classification

    def pretrain_DNN(self, epochs, learning_rate, batch_size):
        """Train the DBN for pre-training using contrastive divergence.

        Parameters
        ----------
        epochs : int
            Number of training epochs for each RBM layer in the DBN.
        learning_rate : float
            Learning rate for training RBMs.
        batch_size : int
            Size of data batches for training RBMs.
        """

        self.dbn.train_DBN(epochs, learning_rate, batch_size)

    @staticmethod
    def calcul_softmax(X, rbm):
        """Calculate the softmax probabilities for a given RBM layer.
        Will only be used to compute the classification probabilities for the classification RBM.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data for the RBM layer.
        rbm : RBM object
            The RBM layer for which to calculate softmax.

        Returns
        -------
        logits : ndarray, shape (n_samples, num_classes)
            Logits (unnormalized outputs) of the softmax layer.
        probabilities : ndarray, shape (n_samples, num_classes)
            Softmax probabilities for each class.
        """

        logits = rbm.b + X @ rbm.W
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return logits, probabilities

    def entree_sortie_reseau(self, X):
        """Compute the forward pass through the DNN network.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Input data for the network.

        Returns
        -------
        outputs : list
            List containing the activations at each layer (including input).
        proba_classif_layer : ndarray, shape (n_samples, num_classes)
            Softmax probabilities from the classification layer.
        """

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
        """Calculate the cross-entropy loss for a batch of data.

        Parameters
        ----------
        labels : ndarray, shape (n_samples, num_classes)
            True labels (one-hot encoded) for the data.
        proba : ndarray, shape (n_samples, num_classes)
            Softmax probabilities from the classification layer.

        Returns
        -------
        loss : float
            Cross-entropy loss for the batch.
        """

        return -(np.array(labels) * np.log(proba)).sum()

    def test_DNN(self, X, y, batch_size=32):
        """Evaluate the DNN on a given dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Data for evaluation.
        y : ndarray, shape (n_samples, num_classes)
            True labels (one-hot encoded) for the data.
        batch_size : int, default 32
            Batch size for each pass through the network, for evaluation.

        Returns
        -------
        loss : float
            Average loss over the entire dataset.
        accuracy : float
            Classification accuracy on the dataset.
        """

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
            classes_predicted = classification_probabilities.argmax(axis=1)
            nb_well_classified += (classes_predicted == y_batch.argmax(axis=1)).sum()
        loss /= n
        accuracy = nb_well_classified / n
        return loss, accuracy

    def compute_and_plot_metrics(self, loss, accuracy, val_losses, val_accuracies, min_val_loss,
                                 patience_counter, patience, epoch, epochs, print_error_every):
        """Compute and potentially plot training and validation metrics.

        Parameters
        ----------
        loss : float
            Training loss at current epoch.
        accuracy: float
            Training accuracy at current epoch.
        val_losses: list
            List of validation losses computed at each epoch.
        val_accuracies: list
            List of validation accuracies computed at each epoch.
        min_val_loss : float
            Minimum validation loss observed so far.
        patience_counter : int
            Number of epochs without improvement in validation loss.
        epoch : int
            Current training epoch.
        epochs : int
            Total number of training epochs.
        print_error_every : int
            Frequency (in epochs) for printing training and validation metrics.

        Returns
        -------
        patience_trigger : bool
            Whether early stopping should be triggered based on validation loss.
        min_val_loss : float
            Minimum validation loss observed so far.
        patience_counter : int
            Number of epochs without improvement in validation loss.
        """

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
                print(f'Epoch {epoch}:: loss: {loss:.4f}, val_loss: {val_loss:.4f} ; '
                      f'accuracy: {(100*accuracy):.2f}%, val_accuracy: {(val_acc*100):.2f}%')
        elif (epoch % print_error_every == 0 or epoch == epochs-1) and print_error_every != -1:
            print(f'Epoch {epoch}:: loss: {loss:.4f}, accuracy: {(accuracy*100):.2f}%')

        return patience_trigger, min_val_loss, patience_counter

    @staticmethod
    def plot_loss_acc(losses, accs, labels, suptitle=''):
        """Plot training and validation loss/accuracy curves.

        Parameters
        ----------
        losses : list of lists
            List containing training and validation loss curves.
        accs : list of lists
            List containing training and validation accuracy curves.
        labels : list
            List of labels for the curves (e.g., ['Train', 'Test']).
        suptitle : str, optional
            Title for the overall plot (default: '').
        """

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
        """Initialize optimizers (Adam or SGD) for each RBM layer,
        if the RBM has no optimizer already initialized.

        Parameters
        ----------
        learning_rate : float
            Learning rate for training RBMs.
        """

        for rbm in self.dbn.rbms + [self.rbm_classification]:
            if self.use_adam:
                rbm.optimizer = AdamOptimizer(rbm=rbm, lr=learning_rate)
            else:
                rbm.optimizer = SGD(rbm=rbm, lr=learning_rate) 

    def retropropagation(self, epochs, learning_rate, batch_size, print_error_every=1,
                         plot_=False, patience=np.inf, suptitle=''):
        """Train the DNN using backpropagation for fine-tuning.

        Parameters
        ----------
        epochs : int
            Number of training epochs.
        learning_rate : float
            Learning rate for training.
        batch_size : int
            Size of data batches for training.
        print_error_every : int, optional
            Frequency (in epochs) for printing training and validation metrics.
                If None, prints every epoch for short training (< 10 epochs)
                or every 10th epoch otherwise (default: 1).
        plot_ : bool, optional
            Whether to plot training and validation loss/accuracy curves after training
            (default: False).
        patience : float or np.inf, optional
            Patience for early stopping based on validation loss
            (default: np.inf, no early stopping).
            Training will stop if validation loss does not improve for `patience` epochs.
        suptitle : str, optional
            Title for the loss/accuracy plot if `plot_` is True (default: '').

        Returns
        -------
        losses : list
            List of training loss values per epoch.
        accuracies : list
            List of training accuracy values per epoch.
        val_losses : list, optional
            List of validation loss values per epoch (if validation data is provided).
        val_accuracies : list, optional
            List of validation accuracy values per epoch (if validation data is provided).
        """

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
                classes_predicted = classification_probabilities.argmax(axis=1)
                nb_well_classified += (classes_predicted == y_batch.argmax(axis=1)).sum()

                # Gradient RBM classification ###
                proba_diff_clf = classification_probabilities - y_batch
                grad_W_clf = outputs[-2].T @ proba_diff_clf
                grad_b_clf = np.sum(proba_diff_clf, axis=0)
                W_plus_one = self.rbm_classification.W.copy()  # copy it before gradient descent as it is needed later on = self.jls_extract_def()
                self.rbm_classification.optimizer.step(grad_W=grad_W_clf, grad_b=grad_b_clf,
                                                       grad_a=None, descent=True)

                # Gradient hidden RBMs ###
                c_plus_one = proba_diff_clf
                for p in range(len(self.dbn.rbms)-1, -1, -1):
                    rbm = self.dbn.rbms[p]
                    x_p_minus_one = outputs[p]  # it is already shifted by 1 by default with the presence of rbm_classification
                    x_p = outputs[p+1]

                    c_p = (c_plus_one @ W_plus_one.T) * (x_p * (1-x_p))
                    W_plus_one = rbm.W.copy()
                    c_plus_one = c_p

                    grad_W = x_p_minus_one.T @ c_p
                    grad_b = np.sum(c_p, axis=0)
                    rbm.optimizer.step(grad_W=grad_W, grad_b=grad_b, grad_a=None, descent=True)

            loss /= n
            losses.append(loss)
            accuracy = nb_well_classified / n
            accuracies.append(accuracy)

            patience_trigger, min_val_loss, patience_counter = self.compute_and_plot_metrics(loss, accuracy, val_losses, val_accuracies,
                                                min_val_loss, patience_counter, patience, epoch, epochs, print_error_every)
            if patience_trigger:
                print(f'##### Patience triggered at epoch {epoch}! #####')
                _, _, _ = self.compute_and_plot_metrics(loss, accuracy, val_losses, val_accuracies,
                                                        min_val_loss, patience_counter, patience, epoch, epochs, epoch)
                break

        val_losses, val_accuracies = val_losses[:len(losses)], val_accuracies[:len(accuracies)] # this is because if patience is triggered, the last validation values are added twice to the lists
        if plot_:
            self.plot_loss_acc([losses, val_losses], [accuracies, val_accuracies], labels=['Train', 'Test'], suptitle=suptitle)
        return losses, accuracies, val_losses, val_accuracies

    def save_weights(self, path):
        """Save the weights of all RBMs in the DNN to a pickle file.

        Parameters
        ----------
        path : str
            Path to save the weights file (as a pickle file).

        Returns
        -------
        dict_weights : dict, optional
            Dictionary containing the weights of each RBM layer
            (indexed by layer number or 'classif' for the topmost classification layer).
            Is returned if path is set to None.
        """

        dict_weights = self.dbn.save_weights(path=None)
        dict_weights['classif'] = self.rbm_classification.save_weights(path=None)

        if path is None:
            return dict_weights
        if not path.endswith('.pkl'):
            path += '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dict_weights, f)

    def load_weights(self, path, dict_weights=None):
        """Loads the weights of all RBMs in the DNN from a pickle file.

        Parameters
        ----------
        path : str
            Path to the weights file (as a pickle file).
        dict_weights : dict, default None
            Optional dictionary containing loaded weights.
        """

        if dict_weights is None:
            with open(path, 'rb') as f:
                dict_weights = pickle.load(f)
        self.rbm_classification.load_weights(path=None, dict_weights=dict_weights.pop('classif'))
        self.dbn.load_weights(path=None, dict_weights=dict_weights)
