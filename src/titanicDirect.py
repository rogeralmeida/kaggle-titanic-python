import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

def extract_y(data):
	y = np.asarray(data[:, 1], dtype=np.float32)
	return np.reshape(y, -1)

def extract_x(data, columns_2_remove=[0, 1, 3, 8, 10]):
	relevant_features = np.delete(data, columns_2_remove, 1)
	relevant_features = convert_texts(relevant_features)
	relevant_features=np.asarray(relevant_features, dtype=np.float32)
	return relevant_features

def convert_texts(data):
	data[data == 'male']=1.
	data[data == 'female']=2.
	data[data == '']=0.
	data[data == 'C']=1.
	data[data == 'Q']=2.
	data[data == 'S']=3.
	return data


class LinearSVCHipotesys:
	def __init__(self, C, loss, penalty, tol, fit_intercept, intercept_scaling):
		self.linearSVC = LinearSVC(C=C, loss=loss, penalty=penalty, tol=tol, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)

	def calculatePrecision(self, train_data, validation_data):
		y = extract_y(train_data)
		x = extract_x(train_data)
		self.linearSVC.fit(x, y)
		validation_y = extract_y(validation_data)
		validation_x = extract_x(validation_data)
		return self.linearSVC.score(validation_x, validation_y)

	def predict(self, train_data):
		x = extract_x(train_data, [0, 2, 7, 9])
		print "Prestes a prever o resultado para:"
		print x
		self.linearSVC.predict(x)

class LogisticRegressionHipotesys:

	def __init__(self, penalty, tol, c, fit_intercept, intercept_scaling):
		self.logisticRegression = LogisticRegression(penalty=penalty, tol=tol, C=c, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)


	def calculatePrecision(self, train_data, validation_data):
		y = extract_y(train_data)
		x = extract_x(train_data)
		self.logisticRegression.fit(x, y)
		validation_y = extract_y(validation_data)
		validation_x = extract_x(validation_data)
		return self.logisticRegression.score(validation_x, validation_y)


	def predict(self, train_data):
		x = extract_x(train_data, [0, 2, 7, 9])
		print "Prestes a prever o resultado para:"
		print x
		self.logisticRegression.predict(x)

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

tol = 0.01
c = 0.01
intercept_scaling = 0.01
winner = 0.0
winner_tol = tol
winner_c = c
winner_penalty = ''
for penalty in ['l1', 'l2']:
	while tol <= 2:
		while c <= 2:
			for fit_intercept in [True, False]:
				while intercept_scaling <= 2:
					print 'variables'
					print 'penalty'
					print penalty
					print 'tol'
					print tol
					print 'c'
					print c
					hipotesys = LogisticRegressionHipotesys(penalty, tol, c, fit_intercept, intercept_scaling)
					precision = hipotesys.calculatePrecision(train_data, validation_data)
					print 'precision'
					print precision
					tol += 0.01
					c += 0.01
					intercept_scaling += 0.01
					if precision > winner:
						winner = precision
						winner_tol = tol
						winner_c = c
						winner_penalty = penalty

tol = 0.01
c = 0.01
intercept_scaling = 0.01
Bwinner = 0.0
Bwinner_tol = tol
Bwinner_c = c
Bwinner_penalty = ''
for penalty in ['l1', 'l2']:
	while tol <= 2:
		while c <= 2:
			for fit_intercept in [True, False]:
				while intercept_scaling <= 2:
					for loss in ['l1', 'l2']:
						print 'variables'
						print 'penalty'
						print penalty
						print 'tol'
						print tol
						print 'c'
						print c
						hipotesys = LinearSVCHipotesys(c, 'l2', 'l2', tol, fit_intercept, intercept_scaling)
						precision = hipotesys.calculatePrecision(train_data, validation_data)
						print 'precision'
						print precision
						tol += 0.01
						c += 0.01
						intercept_scaling += 0.01
						if precision > Bwinner:
							Bwinner = precision
							Bwinner_tol = tol
							Bwinner_c = c
							Bwinner_penalty = penalty

print "===================================================="
print "===================Resultados======================="
print "===================================================="
print 'Logistic Regression Max precision was'
print winner
print 'parameters'
print winner_penalty
print winner_tol
print winner_c

print 'LinearSVC Max precision was'
print Bwinner
print 'parameters'
print Bwinner_penalty
print Bwinner_tol
print Bwinner_c
