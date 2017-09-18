#!/usr/bin/python

__author__ = "Maryam Salehi"

import unittest
import numpy as np
from scipy.optimize import fmin_bfgs
from logisticReg import LogisticReg

class UnitTest(unittest.TestCase):

    def setUp(self):        
        data = np.loadtxt("ex2data2.txt", delimiter = ",")
        self.x = data[:, [0, 1]]
        self.y = data[:, 2]
        self.degree = 6
        self.lambda_param = 1
        self.x_map = LogisticReg.map_feature(self.x[:,0], self.x[:,1], self.degree)
        # Initialize fitting parameters
        self.initial_theta = np.zeros([self.x_map.shape[1], 1])
        # Compute initial cost and gradient for regularized logistic regression

    def test_map_feature(self):
        expected_feature_array = np.array([[1, 1, 2],[1, 1, 2],[1, 1, 2]]);
        actual_feature_array = LogisticReg.map_feature(np.array([1, 1, 1]), np.array([2, 2, 2]), 1)
        self.assertTrue((expected_feature_array == actual_feature_array).all())

    def test_predict(self):
        expected_predictions = np.array([[1],[0]]);
        actual_predictions = LogisticReg.predict(np.array([1, -1]).reshape(2,1), np.array([[2 , 1],[1 , 2]]));
        self.assertTrue((expected_predictions == actual_predictions).all())
          
    def test_cost_function_reg(self):
        expected_cost = 0.693  # value taken from coursera machine leraning (assignment 2)
        cost = LogisticReg.cost_function_reg(self.initial_theta, self.x_map, self.y, self.lambda_param)
        self.assertAlmostEqual(cost.item(0), expected_cost, places=3, msg=None, delta=None)

    def test_prediction_accuracy(self):
        expected_accuracy = 83.05 # value taken from coursera machine leraning (assignment 2)
        theta = fmin_bfgs(LogisticReg.cost_function_reg, self.initial_theta, LogisticReg.grad_function_reg, args=(self.x_map, self.y, self.lambda_param), maxiter=400)
        predictions = LogisticReg.predict(np.array(theta), self.x_map)
        accuracy = (self.y[np.where(predictions == self.y)].size / float(self.y.size)) * 100.0
        self.assertAlmostEqual(accuracy, expected_accuracy, places=2, msg=None, delta=None)

if __name__ == '__main__':
    unittest.main()