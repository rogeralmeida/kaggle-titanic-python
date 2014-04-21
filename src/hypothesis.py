import numpy as np
from sklearn.preprocessing import scale
class Hypothesis:

  def __init__(self, train_data, validation_data):
    self.trainY = self.extract_y(train_data)
    self.trainX = self.extract_x(train_data)
    self.validationY = self.extract_y(validation_data)
    self.validationX = self.extract_x(validation_data)
    self.trainX = scale(self.trainX)
    self.validationX = scale(self.validationX)

  def extract_y(self, data):
    y = np.asarray(data[:, 1], dtype=np.float32)
    return np.reshape(y, -1)

  def extract_x(self, data, columns_2_remove=[0, 1, 3, 8, 10]):
    relevant_features = np.delete(data, columns_2_remove, 1)
    relevant_features = self.convert_texts(relevant_features)
    relevant_features=np.asarray(relevant_features, dtype=np.float32)
    return relevant_features

  def convert_texts(self, data):
    data[data == 'male']=1.
    data[data == 'female']=2.
    data[data == '']=0.
    data[data == 'C']=1.
    data[data == 'Q']=2.
    data[data == 'S']=3.
    return data

  def train(self):
    highest_precision = 0
    local_tol = 0.01
    local_c = 0.01
    local_intercept_scaling = 0
    for local_penalty in ['l1', 'l2']:
      while local_tol <= 2:
        while local_c <= 2:
          for local_fit_intercept in [True, False]:
            while local_intercept_scaling <= 2:
              for local_loss in ['l1', 'l2']:
                if self.valid_parameters(local_c, local_loss, local_penalty, local_tol, local_fit_intercept, local_intercept_scaling):
                  hipothesys = self.create_hipothesys(local_c, local_loss, local_penalty, local_tol, local_fit_intercept, local_intercept_scaling)
                  precision = self.calculatePrecision(hipothesys)
                  if precision > highest_precision:
                    self.hipothesys = hipothesys
                    self.precision = precision
                    highest_precision = precision
                local_tol += 0.01
                local_c += 0.01
                local_intercept_scaling += 0.01

  def score(self):
    return self.precision

  def calculatePrecision(self, hipothesys):
    hipothesys.fit(self.trainX, self.trainY)
    return hipothesys.score(self.validationX, self.validationY) 

  def predict(self, test_data):
    x = self.extract_x(test_data, [0, 2, 7, 9])
    predicted = self.hipothesys.predict(x)
    resultado = np.empty([test_data.shape[0], 2], dtype=int)
    resultado[:, 0] = test_data[:, 0]
    resultado[:, 1] = predicted.astype('int')
    return resultado