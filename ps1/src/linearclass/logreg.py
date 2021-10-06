import numpy as np
import util

import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def grad_l(x,y,theta):
    n,d = x.shape
    cumm = np.zeros(d)
    for i in range(n):
        cumm += (y[i]-sigmoid(np.dot(theta,x[i])))*x[i]
    return cumm

def H_l(x,y,theta):
    n,d = x.shape
    cumm = np.zeros((d,d))
    for i in range(n):
        _ = sigmoid(np.dot(theta,x[i]))
        cumm += _*(1-_)* np.outer(x[i],x[i])
    return -cumm

def newton(x,y,theta0 = None, eps = 1e-5):
    n,d = x.shape
    if theta0 is None:
        theta0 = np.zeros(d)
    diff = 1
    theta = theta0
    while(diff>eps):
        theta_old = theta.copy()
        theta -= np.linalg.inv(H_l(x,y,theta))@grad_l(x,y,theta)
        diff = np.linalg.norm(theta-theta_old,1)
    return theta

def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=True)
    y_pred = clf.predict(x_eval)

    # Plot decision boundary on top of validation set set
    util.plot(x_eval, y_eval, clf.theta, 'output/problem1b{}.png'.format(save_path[-5]))
    
    np.savetxt(save_path, y_pred , fmt='%d')
    
    # Use np.savetxt to save predictions on eval set to save_path

    # print(np.mean((y_pred>0.5)==y_eval))
    
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***=
        n,d = x.shape
        if self.theta is None:
            self.theta = np.zeros(d)
        
        self.theta = newton(x,y,self.theta)
        # *** END CODE HERE ***
        
    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        return 1/(1+np.exp(-self.theta@x.T))

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
