import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from hypothesis import Hypothesis

class LinearSVCHipotesys(Hypothesis):

	def valid_parameters(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		if penalty == 'l1' and loss == 'l1':
			return False
		else:
			return True

	def create_hypothesis(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		hipotesys = LinearSVC(C=c, loss=loss, penalty=penalty, tol=tol, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=False, verbose=1)
		return hipotesys


class LogisticRegressionHipotesys(Hypothesis):

	def valid_parameters(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		if penalty == 'l1' and loss == 'l1':
			return False
		else:
			return True

	def create_hypothesis(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		hypothesis = LogisticRegression(C=c, penalty=penalty, tol=tol, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=False)
		return hypothesis
	

def load_csv(file_name='train.csv'):
	csv_file_object = csv.reader(open(file_name, 'rb')) #Load in the csv file
	header = csv_file_object.next() #Skip the fist line as it is a header
	the_data=[] #Creat a variable called 'data'
	the_ids=[]
	for row in csv_file_object: #Skip through each row in the csv file
	    the_data.append(row[:]) #adding each row to the data variable
	return np.array(the_data)

train_data = load_csv('train.csv')

np.random.shuffle(train_data)
import math
validation_size = math.floor(train_data.shape[0] * 0.2)
validation_data = np.asarray(train_data[:validation_size, :])
train_data = np.asarray(train_data[validation_size:, :])

logistic_regression = LogisticRegressionHipotesys(train_data, validation_data)
logistic_regression.train()

linear_svc_hipotesys = LinearSVCHipotesys(train_data, validation_data)
linear_svc_hipotesys.train()

# tol = 0.01
# c = 0.01
# intercept_scaling = 0.01
# winner = 0.0
# winner_tol = tol
# winner_c = c
# winner_penalty = ''
# for penalty in ['l1', 'l2']:
# 	while tol <= 2:
# 		while c <= 2:
# 			for fit_intercept in [True, False]:
# 				while intercept_scaling <= 2:
# 					print 'variables'
# 					print 'penalty'
# 					print penalty
# 					print 'tol'
# 					print tol
# 					print 'c'
# 					print c
# 					hipotesys = LogisticRegressionHipotesys(penalty, tol, c, fit_intercept, intercept_scaling)
# 					precision = hipotesys.calculatePrecision(train_data, validation_data)
# 					print 'precision'
# 					print precision
# 					tol += 0.01
# 					c += 0.01
# 					intercept_scaling += 0.01
# 					if precision > winner:
# 						winner = precision
# 						winner_tol = tol
# 						winner_c = c
# 						winner_penalty = penalty



