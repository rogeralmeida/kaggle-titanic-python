import csv as csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from hypothesis import Hypothesis

class LinearSVCHipotesys(Hypothesis):

	def valid_parameters(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		if penalty == 'l1' and loss == 'l1':
			return False
		else:
			return True

	def create_hipothesys(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		hipotesys = LinearSVC(C=c, loss=loss, penalty=penalty, tol=tol, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=False, verbose=1)
		return hipotesys


class LogisticRegressionHipotesys(Hypothesis):

	def valid_parameters(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		if penalty == 'l1' and loss == 'l1':
			return False
		else:
			return True

	def create_hipothesys(self, c, loss, penalty, tol, fit_intercept, intercept_scaling):
		hypothesis = LogisticRegression(C=c, penalty=penalty, tol=tol, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, dual=False)
		return hypothesis

class SVCHipotesys(Hypothesis):

	def train(self):
		highest_precision = 0
		hipothesys = SVC(C=1.0, kernel='rbf', degree=3, gamma=0.0, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, random_state=None)
		precision = self.calculatePrecision(hipothesys)
		if precision > highest_precision:
			self.hipothesys = hipothesys 
			self.precision = precision

	def predict(self, test_data):
		x = self.extract_x(test_data, [0, 2, 7, 9])
		predicted = self.hipothesys.predict(x)
		resultado = np.empty([test_data.shape[0], 2], dtype=int)
		resultado[:, 0] = test_data[:, 0]
		resultado[:, 1] = predicted.astype('int')
		return resultado

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

hipotesys_list = []
hipotesys_list.append(LogisticRegressionHipotesys(train_data, validation_data))
hipotesys_list.append(LinearSVCHipotesys(train_data, validation_data))
hipotesys_list.append(SVCHipotesys(train_data, validation_data))
winner_hypothesis = None
highest_score = 0.0
for h in hipotesys_list:
	h.train()
	score = h.score()
	if score > highest_score:
		winner_hypothesis = h
		highest_score = score

print "======== Hipotese Campea ============="
print winner_hypothesis
print "Precisao:"
print highest_score

test_data = load_csv('test.csv')
final_result = winner_hypothesis.predict(test_data)
np.savetxt('final_result.csv', final_result.astype('int'), fmt="%.1d", delimiter=",")#, header="PassengerId,Survived")