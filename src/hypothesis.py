import numpy as np
class Hypothesis:

  def __init__(self, train_data, validation_data):
    self.train_data = train_data
    self.validation_data = validation_data

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
                  hypothesis = self.create_hypothesis(local_c, local_loss, local_penalty, local_tol, local_fit_intercept, local_intercept_scaling)
                  self.calculatePrecision(hypothesis)
                else:
                  hypothesis = None
                local_tol += 0.01
                local_c += 0.01
                local_intercept_scaling += 0.01

  def calculatePrecision(self, hypothesis):
    y = self.extract_y(self.train_data)
    x = self.extract_x(self.train_data)
    hypothesis.fit_transform(x, y)

    validation_y = self.extract_y(self.validation_data)
    validation_x = self.extract_x(self.validation_data)

    return hypothesis.score(validation_x, validation_y) 
