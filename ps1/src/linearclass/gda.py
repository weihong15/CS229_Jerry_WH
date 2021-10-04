import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    clf = GDA()
    clf.fit(x_train, y_train)
    
    x_eval, y_eval = util.load_dataset(valid_path, add_intercept=False)
    y_pred = clf.predict(x_eval)
    
    # Plot decision boundary on validation set
    util.plot(x_eval, y_eval, np.append([clf.theta_0],clf.theta,0), 'output/problem1e{}.png'.format(save_path[-5]))
    # Use np.savetxt to save outputs from validation set to save_path
    np.savetxt(save_path, y_pred > 0.5, fmt='%d')
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        n,d = x.shape
        y1 = np.sum(y)
        y0 = n-y1
        # Find phi, mu_0, mu_1, and sigma
        phi = y1/n
        mu_0= 0
        for i in range(n):
            mu_0 += (1-y[i])*x[i]
        mu_0/= y0
        
        mu_1= 0
        for i in range(n):
            mu_1 += (y[i])*x[i]
        mu_1/= y1
        
        Sigma = np.zeros((d,d))
        for i in range(n):
            if y[i]>0.5:
                Sigma+= np.outer(x[i]-mu_1,x[i]-mu_1)
            else:
                Sigma+= np.outer(x[i]-mu_0,x[i]-mu_0)
        Sigma /= n
        # Write theta in terms of the parameters
        inv = np.linalg.inv(Sigma)
        self.theta = inv@(mu_1-mu_0)
        self.theta_0 = 0.5*(mu_0@inv@mu_0 - mu_1@inv@mu_1)-np.log((1-phi)/phi)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1/(1+np.exp(-self.theta@x.T-self.theta_0))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
