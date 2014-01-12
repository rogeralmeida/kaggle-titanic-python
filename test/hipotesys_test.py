import unittest
from unittest.mock import MagicMock
class TestLogisticRegressionHipotesys(unittest.TestCase):
	def test_execute(self):
		function_object=Object()
		function_object.fit = MagicMock()
		function_object.predict = MagicMock()

		hipotesys = LogisticRegressionHipotesys(function_object, train_data, y_train_data, validation_data, y_validation_data)
		hipotesys.execute()
		precision = hipotesys.precision()
		
		function_object.fit.assert_called_with(train_data)
		function_object.predict.assert_called_with(validation_data)


if __name__ == '__main__':
    unittest.main()