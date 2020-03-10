# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, learning_rate=0.01, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def sigmoid(self, z):
        """
        Compute value of sigmoid function "g", where:
        g(z) = 1 / (1 + e^-z) 
        """
        g = 1 / (1 + np.exp(-z))
        return g
        
    def hypothesis(self, X, theta):
        """
       Compute hypothesis "h" where: h(x) = g(z) with:
            z = X.T * theta
            g, a sigmoid function

		Parameters
    	------------
    	X: numpy array
			Features vector.
    	theta: numpy array 
			Parameters vector.

		Returns
    	------------
		numpy ndarray
		"""
        z = X.dot(theta) 
        h = self.sigmoid(z)
        return h

    def compute_cost(self, h, y):
        """
		Compute value of cost function "J", where:
		J(theta) = -np.average(y * np.log(h) + (1 - y) * np.log(1 - h))

		Parameters
    	------------
    	h: numpy array
			Predictions vector.
		y: numpy array
			Target vector.

		Returns
    	------------
		Value of cost function J
		"""
        m = len(y)
        J = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return J

    def add_intercept(self, X):
        """
        Add intercept to feature vector.
        """
        return np.c_[np.ones((m, 1)), X]


    def fit(self, X, y):
        """
		Fit theta to the training set.

		Parameters
    	------------
    	epochs: int 
			Number of iterations.
    	learning_rate: float 
			Rate at which theta is updated

		Returns
    	------------
		numpy ndarray
		"""
        X_b = self.add_intercept(X)
        costs = []
        self.theta = np.random.rand(X_b.shape[1])
        for epoch in range(self.epochs):
            h = self.hypothesis(X_b, self.theta)
            error = h - y
            gradients = 1/m * X_b.T.dot((error))
            self.theta = self.theta - self.learning_rate * gradients
            cost = self.compute_cost(h, y)
            costs.append(cost)

        self.costs = costs
        self.epochs = epochs

        return self.theta

    def predict(self, X):
        """
        """
        X_b = self.add_intercept(X)
        self.predictions = self.hypothesis(X_b, self.theta)
        return self.predictions

    def plot_cost(self):
        """Plot iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
        plt.figure()
        plt.plot(np.arange(1, self.epochs+1), self.costs)
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Cost function minimization")
        plt.show()

    def plot_model(self, X, y):
        """Plot fitted model.
        
        Returns
        -----------       
        matploblib figure
        """ 
        plt.figure()
        x_values = [np.min(x[:, 0]), np.max(x[:, 1] + 3)]
        y_values = -(self.theta[0] + np.dot(self.theta[1], x_values)) / self.theta[2]
        plt.scatter(x[:,0], x[:,1], c=y)
        plt.plot(x_values, y_values, label='Decision Boundary')
        plt.title("Fitted model")
        plt.show()
  
if __name__ == "__main__":
    iris = datasets.load_iris()
    x = iris.data[:, :2] # select only the first 2 features
    y = (iris.target != 0) * 1 # make dataset into binary classification problem
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(x, y)

    print("theta_hat", model.theta)

    model.plot_cost()
    model.plot_model(x, y)
