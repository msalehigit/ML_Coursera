#!/usr/bin/python

import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs
from logisticReg import LogisticReg
from logisticRegTest import UnitTest

if __name__ == "__main__":
    
    data = np.loadtxt("ex2data2.txt", delimiter = ",")
    x = data[:, [0, 1]]
    y = data[:, 2]
    degree = 6
    # Create Polynomial Features
    x_map = LogisticReg.map_feature(x[:,0], x[:,1], degree)

    # Initialize fitting parameters
    initial_theta = np.zeros([x_map.shape[1], 1])

    # Set regularization parameter lambda to 1
    lambda_param = 1
    
    # train logistic regression model to find optimum parameters theta     
    theta = fmin_bfgs(LogisticReg.cost_function_reg, initial_theta, LogisticReg.grad_function_reg, args=(x_map, y, lambda_param), maxiter=400)

    LogisticReg.plot_decision_boundary(theta, x, y, degree, "Microchip Test 1", "Microchip Test 2")

    predictions = LogisticReg.predict(np.array(theta), x_map)
    accuracy = (y[np.where(predictions == y)].size / float(y.size)) * 100.0
    print ('prediction accuracy is :', accuracy)    

