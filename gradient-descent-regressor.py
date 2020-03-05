# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

class GradientDescentRegressor(object):
	"""
	GradientDescent

    Parameters
    ------------
	X: numpy array
		Features vector.
	y: numpy array
		Target vector.

	Attributes
	------------
	theta : 1-d numpy array, shape = [polynomial order + 1,] 
		Parameters randomly initialized, with theta[0] corresponding
		to the intercept term
	
	method : str , values = "batch_gradient_descent" | "SGD" | "MBGD"
		Method used for finding optimal values of theta
	
	If gradient descent method is chosen:
	
		costs : 1-d numpy array,
			Cost function values for every iteration of gradient descent
		
		epochs: int
			Number of iterations of gradient descent to be performed
    """

	def __init__(self, X, y):
		self.X = X
		self.y = y
	
	def hypothesis(self, X, theta):
		"""
		Compute hypothesis "h" where:
		h(x) = theta_0 * x + theta_1 

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
		h = X.dot(theta)
		return h
	
	def compute_cost(self, error, y):
		"""
		Compute value of cost function "J", where:
		J(theta) = 1/m * ((X.dot(theta)) - y)**2

		Parameters
    	------------
    	X: numpy array
			Features vector.
		y: numpy array
			Target vector.
		theta: numpy array
			Parameters vector.

		Returns
    	------------
		Value of cost function J
		"""
		m = len(y)
		J = (1/m) * np.sum(error**2)
		return J
    
	def fit(self, epochs=100, learning_rate=0.1):
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
		m = X.shape[0]
		theta = np.random.rand(2, 1)
		X_b = np.c_[np.ones((m, 1)), X]
		costs = []
		
		for epoch in range(epochs):
			h = self.hypothesis(X_b, theta)
			error = h - y
			gradients = 2/m * X_b.T.dot(error) # slope of the derivative of the cost function
			theta = theta - learning_rate * gradients # update parameters after each iteration
			cost = self.compute_cost(error, y)
			costs.append(cost)

		self.costs = costs
		self.epochs = epochs
		self.predictions = X_b.dot(theta)

		return theta

	def plot_cost(self):
		"""Plot number of gradient descent iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
		plt.figure()
		plt.plot(np.arange(1, self.epochs+1), self.costs)
		plt.show()

	def plot_model(self):
		"""Plot number of gradient descent iterations versus cost function.
        
        Returns
        -----------       
        matploblib figure
        """ 
		plt.figure()
		plt.scatter(X, y)
		plt.plot(X, self.predictions, "g-")
		plt.show()

if __name__ == "__main__":
	X = 2 * np.random.rand(100, 1)
	y = 3 + 5 * X + np.random.rand(100, 1)

	gradient_descent = GradientDescentRegressor(X, y)

	print(gradient_descent.fit())
	gradient_descent.plot_cost()
	gradient_descent.plot_model()