import numpy as np

from rbm import RBM, sigmoid
from dbn import DBN

class DNN:
    def __init__(self, X_train, y_train, num_classes, num_hidden_layers, neurons):
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = num_classes
        self.num_hidden_layers = num_hidden_layers
        self.neurons = neurons
        self.init_DNN()
        
    def init_DNN(self):
        self.dbn = DBN(self.X_train, self.num_hidden_layers, self.neurons)
        rbm_classification = RBM(self.dbn.rbms[-1].b, self.num_classes)
        self.rbm_classification = rbm_classification
        
    def pretrain_DNN(self, epochs, learning_rate, batch_size):
        self.dbn.train_DBN(epochs, learning_rate, batch_size)
    
    @staticmethod
    def calcul_softmax(X, rbm):
        logits = sigmoid(rbm.b + X @ rbm.W)
        probabilities = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        return logits, probabilities
    
    def entree_sortie_reseau(self, X):
        outputs, probabilities = [], []
        
        h = X
        for rbm in self.dbn.rbms:
            output_layer, proba_layer = self.calcul_softmax(h, rbm)
            outputs.append(output_layer)
            probabilities.append(proba_layer)
            h = output_layer
        
        output_classif_layer, proba_classif_layer = self.calcul_softmax(h, self.rbm_classification)
        outputs.append(output_classif_layer)
        probabilities.append(proba_classif_layer)
        return outputs, probabilities
    
    @staticmethod
    def cross_entropy(labels, proba):
        return -(np.array(labels) * np.log(proba)).sum()
    
    def retropropagation(self, epochs, learning_rate, batch_size, print_error_every=None):
        n, p = self.X_train.shape
        
        if print_error_every is None:
            print_error_every = 1 if epochs < 10 else epochs / 10
        
        errors = []
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n)
            X_shuffled = self.X_train[indices]
            y_shuffled = self.y_train[indices]
            
            loss = 0.
            for i in range(0, n, batch_size):
                X_batch = X_shuffled[i: i+batch_size]
                y_batch = y_shuffled[i: i+batch_size]
                
                outputs, probabilities = self.entree_sortie_reseau(X_batch)
                classification_probabilities = probabilities[-1]
                loss_batch = self.cross_entropy(y_batch, classification_probabilities)
                loss += loss_batch
                
            loss /= n
        