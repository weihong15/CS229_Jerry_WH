import util
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='raise')


factor = 2.0

class LinearModel(object):
    """Base class for linear models."""

    def __init__(self, theta=None):
        """
        Args:
            theta: Weights vector for the model.
        """
        self.theta = theta

    def fit(self, X, y):
        """Run solver to fit linear model. You have to update the value of
        self.theta using the normal equations.

        Args:
            X: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # solve for theta in normal equation X^T * X theta = X^T * Y
        self.theta = np.linalg.solve(np.matmul(np.transpose(X), X), np.matmul(np.transpose(X), y))
        # *** END CODE HERE ***

    def create_poly(self, k, X):
        """
        Generates a polynomial feature map using the data x.
        The polynomial map should have powers from 0 to k
        Output should be a numpy array whose shape is (n_examples, k+1)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        power = range(k + 1)
        output = np.ones((X.size, k + 1))

        for i in range(X.size):
            output[i, :] = X[i] ** power
        return output
        # *** END CODE HERE ***

    def create_sin(self, k, X):
        import math
        """
        Generates a sin with polynomial featuremap to the data x.
        Output should be a numpy array whose shape is (n_examples, k+2)

        Args:
            X: Training example inputs. Shape (n_examples, 2).
        """
        # *** START CODE HERE ***
        power = range(k + 1)
        output_sin = np.ones((X.size, k + 2))
        for i in range(X.size):
            output_sin[i, 0:-1] = X[i] ** power
            output_sin[i][-1] = math.sin(X[i])
        return output_sin
        # *** END CODE HERE ***

    def predict(self, X):
        """
        Make a prediction given new inputs x.
        Returns the numpy array of the predictions.

        Args:
            X: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return np.matmul(X, self.theta)
        # *** END CODE HERE ***


def run_exp(train_path, sine=False, ks=[1, 2, 3, 5, 10, 20], filename='plot.png'):
    train_x,train_y=util.load_dataset(train_path,add_intercept=True)
    plot_x = np.ones([1000, 2])
    plot_x[:, 1] = np.linspace(-factor*np.pi, factor*np.pi, 1000)
    plt.figure()
    plt.scatter(train_x[:, 1], train_y)

    for k in ks:
        '''
        Our objective is to train models and perform predictions on plot_x data
        '''
        # *** START CODE HERE ***
        lm = LinearModel()
        if sine:
            lm.fit(lm.create_sin(k, train_x[:, 1]), train_y)
            plot_y = lm.predict(lm.create_sin(k, plot_x[:, 1]))
        else:
            lm.fit(lm.create_poly(k, train_x[:, 1]), train_y)
            plot_y = lm.predict(lm.create_poly(k, plot_x[:, 1]))
        # *** END CODE HERE ***
        '''
        Here plot_y are the predictions of the linear model on the plot_x data
        '''
        plt.ylim(-2, 2)
        plt.plot(plot_x[:, 1], plot_y, label='k=%d' % k)

    plt.legend()
    plt.savefig(filename)
    plt.clf()


def main(train_path, small_path, eval_path):
    '''
    Run all expetriments
    '''
    # *** START CODE HERE ***
    # 5b k = [3]
    run_exp(train_path = 'train.csv', sine = False, ks = [3], filename = 'ps1_5b.png')
    # 5c k = [3, 5, 10, 20]
    run_exp(train_path = 'train.csv', sine = False, ks = [3, 5, 10, 20], filename = 'ps1_5c')
    # 5d sin featuremap, k = [0, 1, 2, 3, 5, 10, 20]
    run_exp(train_path = 'train.csv', sine = True, ks = [0, 1, 2, 3, 5, 10, 20], filename = 'ps1_5d')
    # 53 k = [1, 2, 5, 10, 20], small dataset
    run_exp(train_path = 'small.csv', sine = False, ks = [1, 2, 5, 10, 20], filename = 'ps1_5e')
    # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='train.csv',
        small_path='small.csv',
        eval_path='test.csv')