#!/usr/bin/python
import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
from scipy.optimize import fmin_bfgs

class LogisticReg(object):

    @staticmethod
    def predict(theta, x):
        """
            parameters: 
                theta: An array of optimized parameters for Logistic Regression 
                    x: Two-dimensional array of training examples (size :num of samples x num of features) 
            Returns: 
                A binary array of predicted labels for training examples (size : num of samples x 1)     

            predict Predicts whether the output is 0 or 1 using parameters vector theta. This method "computes the predictions for X using a 
            threshold at 0.5 (i.e., if sigmoid(x*theta) >= 0.5, predict 1" (Coursera)
                 
            >>> print(predict(np.array([1, -1]).reshape(2,1), np.array([[2 , 1],[1 , 2]])))
            [[1]
             [0]]
                
            >>> predict(np.array([1, -1]), np.array([]))
            Traceback (most recent call last):
            ...
            AssertionError: Data sample is empty        
        """
        m = x.shape[0]
        assert m > 0, "Data sample is empty"
        return (expit(x.dot(theta)) >= 0.5)*1

    @staticmethod
    def map_feature(x1, x2, degree):
        """
            (x1, x2, degree)--> out
            Parameters: 
                x1, x2: Are feature arrays of the same size (size : number of samples x 1)
                degree: Is maximum degree of polynomial features
            Returns: 
                out: A new n-dimensional array with more features, consisting of X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, X1^2*X2, etc.
            map_feature Maps the two input features to quadratic features 
            
            >>> print(map_feature(np.array([1, 1, 1]), np.array([2, 2, 2]), 0))
            [[ 1.]
             [ 1.]
             [ 1.]]

            >>> print(map_feature(np.array([1, 1, 1]), np.array([2, 2, 2]), 1))
            [[ 1.  1.  2.]
             [ 1.  1.  2.]
             [ 1.  1.  2.]]
        """
        x1 = x1.reshape(len(x1), 1);
        x2 = x2.reshape(len(x2), 1);

        assert x1.shape == x2.shape, "length of feature arrays x1 & x2 should be equal"
        out = np.ones((x1.shape[0], 1))

        for ind1 in range(1, degree+1):
            for ind2 in range(0, ind1+1):
                new_feature = np.power(x1, ind1-ind2)*np.power(x2, ind2) # create a new feature
                out = np.append(out, new_feature, axis=1)

        return out    
    
    @staticmethod
    def cost_function_reg(theta, x, y, lambda_param):
        """
            (theta, x, y, lambda_param) --> (cost)
            Parameters:
                theta: An array of logistic regression parameters (size: num of features x 1)
                x: Two dimensional array of training examples (size : num of samples x num of features) 
                y: A binary array of labels of training examples (size : number of samples x 1)
                lambda_param: Parameter of regularized logistic regression 
            Returns:
                cost: Cost of using theta as the parameter for regularized logistic regression
                
            cost_function_reg "computes the cost of using theta as the parameter for regularized logistic regression" (Coursera)
        """
        assert x.shape[0] == y.shape[0], "length of x & y should be equal"
        sample_size = x.shape[0] # number of training examples
        g = expit(x.dot(theta))   # sigmoid function

        cost = (1/sample_size)*(-np.transpose(1-y).dot(np.log(1-g))-np.transpose(y).dot(np.log(g))) 
        regularization_factor = (lambda_param/(2*sample_size))*np.transpose(theta[1:]).dot(theta[1:])
        cost = cost + regularization_factor;
        return cost 
            
    @staticmethod
    def grad_function_reg(theta, x, y, lambda_param):
        """
            (theta, x, y, lambda_param) --> (grad)
            Parameters:
                theta: An array of logistic regression parameters (size: num of features x 1)
                x: Two dimensional array of training examples (size : num of samples x num of features) 
                y: A binary array of labels of training examples (size : number of samples x 1)
                lambda_param: Parameter of regularized logistic regression 
            Returns:
                grad: Gradient of the cost with respect to the parameters (size: num of features x 1)

            grad_function_reg "computes gradient of the cost with respect to the parameters" (Coursera)
        """
        sample_size = x.shape[0] # number of training examples
        g = expit(x.dot(theta))
        grad = (1/sample_size)*np.transpose(x).dot(g-y)
        grad[1:] = grad[1:] + (lambda_param/sample_size)*theta[1:]
        return grad

    @staticmethod
    def plot_data(x, y, x_label, y_label):
        """
            parameters:
                x: Two-dimensional array of training examples (size : num of samples x 2) 
                y: Binary array of labels of training examples
                x_label: str
                y_label: str     

            plot_data Plots the data points x with + for the positive examples (y = 1)
            and o for the negative examples (y = 0). 
        """
        # Find Indices of Positive and Negative Examples
        pos = np.where(y == 1)[0]; 
        neg = np.where(y == 0)[0];

        plt.plot(x[pos, 0], x[pos, 1], 'k+',linewidth = 2.0, MarkerSize = 6);
        plt.plot(x[neg, 0], x[neg, 1], 'ko', MarkerFaceColor = 'y', MarkerSize = 6);
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(['Positive', 'Negative'])
    
    @staticmethod
    def plot_decision_boundary(theta, x, y, degree, x_label, y_label):
        """
            parameters:
                theta: An array of logistic regression parameters (size: num of features x 1)
                x: Training example matrix , where the first column is all-ones
                degree: degree of polynomial features
                x_label: str
                y_label: str

            "Plots the data points X and y into a new figure with the decision boundary defined by theta (coursera)"
            plots the data points with + for the positive examples and o for the negative examples.     
        """
        LogisticReg.plot_data(x, y, x_label, y_label)

        xlist = np.linspace(-1, 1.5, 50)
        ylist = np.linspace(-1, 1.5, 50)
        xlen = len(xlist)
        ylen = len(ylist)

        z = np.zeros([xlen, ylen])
        for x_ind in range(xlen):
            for y_ind in range(ylen):
                z[x_ind, y_ind] = (LogisticReg.map_feature(np.array([xlist[x_ind]]), np.array([ylist[y_ind]]), degree).dot(np.array(theta)))

        z = np.transpose(z)
        plt.contour(xlist, ylist, z)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.show()
