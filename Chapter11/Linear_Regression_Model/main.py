from random import randint
from sklearn.linear_model import LinearRegression
import json
import numpy as np


class Main(object):
	def __init__ (self):
		# Initialize the linear regression model
		self.predictor = LinearRegression(n_jobs =-1)

	def predict(self, input):
		# the limit within which random numbers are generated
		TRAIN_SET_LIMIT = 1000
		  
		# to create exactly 100 data items
		TRAIN_SET_COUNT = 100
		  
		# list that contains input and corresponding output
		TRAIN_INPUT = list()
		TRAIN_OUTPUT = list()
		  
		# loop to create 100 data  items with three columns each
		for i in range(TRAIN_SET_COUNT):
		    a = randint(0, TRAIN_SET_LIMIT)
		    b = randint(0, TRAIN_SET_LIMIT)
		    c = randint(0, TRAIN_SET_LIMIT)

		    op = a + (2 * b) + (3 * c)
		    TRAIN_INPUT.append([a, b, c])
		    TRAIN_OUTPUT.append(op)

		# Fill the Model with the Data
		self.predictor.fit(X = TRAIN_INPUT, y = TRAIN_OUTPUT)

		# Random Test data
		X_TEST = [np.array(json.loads(input))]

		# Predict the result of X_TEST which holds testing data
		outcome = self.predictor.predict(X = X_TEST)
		  
		# Predict the coefficients
		coefficients = self.predictor.coef_
		  
		# Print the result obtained for the test data
		return('Outcome : {}\nCoefficients : {}'.format(outcome, coefficients))

if __name__ == '__main__':
	# Test the ML Package locally
	m = Main()