import csv as csv
import numpy as np
import matplotlib.pyplot as plt

def load_csv(file_name='train.csv'):
	csv_file_object = csv.reader(open(file_name, 'rb')) #Load in the csv file
	header = csv_file_object.next() #Skip the fist line as it is a header
	the_data=[] #Creat a variable called 'data'
	the_ids=[]
	for row in csv_file_object: #Skip through each row in the csv file
	    the_data.append(row[:]) #adding each row to the data variable
	return np.array(the_data)

def extract_y(data):
	y = np.asarray(data[:, 1], dtype=np.float32)
	return np.reshape(y, -1)

def extract_x(data, columns_2_remove=[0, 1, 3, 6, 7, 8, 9, 10, 11]):
	relevant_features = np.delete(data, columns_2_remove, 1)
	relevant_features = convert_texts(relevant_features)
	relevant_features=np.asarray(relevant_features, dtype=np.float32)
	return relevant_features

def convert_texts(data):
	data[data == 'male']=1.
	data[data == 'female']=0.
	data[data == '']=0.
	return data

from sklearn.linear_model import LogisticRegression
clf_l1_LR = LogisticRegression(penalty='l1', tol=0.01)

train_data = load_csv('train.csv')
print 'train_data'
print train_data
y = extract_y(train_data)
print 'y'
print y
relevant_features = extract_x(train_data)
print 'relevant_features'
print relevant_features
clf_l1_LR.fit_transform(relevant_features, y)

test_data = load_csv('test.csv')
ids = test_data[:, 0]
print 'test_data'
print test_data
print test_data[0]

test_data = extract_x(test_data, [0, 2, 5, 6, 7, 8, 9, 10])
print test_data

predicted = clf_l1_LR.predict(test_data)
print predicted
i = 0
final_result=[]
for survived in predicted:
	final_result.append([ids[i], survived])
	i+=1
final_result = np.asarray(final_result, dtype=np.int32)
print 'final_result'
print final_result
np.savetxt("final_result.csv", final_result, delimiter=",", fmt="%d")
