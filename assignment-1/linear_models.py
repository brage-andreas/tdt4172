import numpy as np

class LinearRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
        
    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """
        
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1/X.shape[0]) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            self.loss_history.append(loss)
            
        return self
    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        
        return np.dot(X, self.weights) + self.bias
    
class LogisticRegression():
    def __init__(self, lr=0.001, n_iterations=1000):
        self.weights = None
        self.bias = None
        
        self.lr = lr
        self.n_iterations = n_iterations
        
        self.loss_history = []
    
    def fit(self, X, y):
        # Initialize weights and bias
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute gradients
            dw = (1/X.shape[0]) * np.dot(X.T, (y_pred - y))
            db = (1/X.shape[0]) * np.sum(y_pred - y)
            
            # Update weights and bias
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            # Compute loss
            loss = np.mean(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred))
            self.loss_history.append(loss)
            
        return self
    
    def predict_proba(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))