import numpy as np

class AdamOptimizer:
    def __init__(self, rbm, lr=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.rbm = rbm
        self.lr = lr  # Learning rate
        self.beta1 = beta1  # Exponential decay rate for the first moment estimates
        self.beta2 = beta2  # Exponential decay rate for the second moment estimates
        self.epsilon = epsilon  # Small number to prevent division by zero
        self.m_W = np.zeros_like(rbm.W)
        self.v_W = np.zeros_like(rbm.W)
        self.m_b = np.zeros_like(rbm.b)
        self.v_b = np.zeros_like(rbm.b)
        self.m_a = np.zeros_like(rbm.a)
        self.v_a = np.zeros_like(rbm.a)
        self.t = 0
        
        
    def step(self, grad_W, grad_b, grad_a, descent):
        direction = 1 if descent else -1
        self.t += 1 

        ### UPDATE GRADIENT W ###
        if grad_W is not None:
            self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * grad_W
            self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * (grad_W ** 2)
            
            m_hat_W = self.m_W / (1 - self.beta1 ** self.t)
            v_hat_W = self.v_W / (1 - self.beta2 ** self.t)
            self.rbm.W -= direction * self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.epsilon)
        
        ### UPDATE GRADIENT B ###
        if grad_b is not None:
            self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad_b
            self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad_b ** 2)
            
            m_hat_b = self.m_b / (1 - self.beta1 ** self.t)
            v_hat_b = self.v_b / (1 - self.beta2 ** self.t)
            self.rbm.b -= direction * self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
        
        ### UPDATE GRADIENT A ###
        if grad_a is not None:
            self.m_a = self.beta1 * self.m_a + (1 - self.beta1) * grad_a
            self.v_a = self.beta2 * self.v_a + (1 - self.beta2) * (grad_a ** 2)
            
            m_hat_a = self.m_a / (1 - self.beta1 ** self.t)
            v_hat_a = self.v_a / (1 - self.beta2 ** self.t)
            self.rbm.a -= direction * self.lr * m_hat_a / (np.sqrt(v_hat_a) + self.epsilon)
        
class SGD:
    def __init__(self, rbm, lr=1e-3):
        self.rbm = rbm
        self.lr = lr
        
    def step(self, grad_W, grad_b, grad_a, descent):
        direction = 1 if descent else -1
        if grad_W is not None:
            self.rbm.W -= direction * self.lr * grad_W
        if grad_b is not None:
            self.rbm.b -= direction * self.lr * grad_b
        if grad_a is not None:
            self.rbm.a -= direction * self.lr * grad_a
        