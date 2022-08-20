from ast import Pass
import numpy as np
from math import sqrt
from scipy.stats import t
import pandas as pd

class simple_linear_regression():
    """
    Take x and y as arrays of floats.

    Try to predict Y by minimizing B0 and B1 in y = B0 + B1X,
    where X is a feature vector.
    """

    def __init__(self, x, y):
        self.x = np.array(x) 
        self.y = np.array(y)
        self.n = len(y)
        self.x_avg = self.x.mean()
        self.y_avg = self.y.mean()
        self.parameters = self._minimize_rss()
        self.std_errors = self._get_std_error()

    # TODO Modify minimizing rss to matrix form. 
    def _minimize_rss(self):
        '''
        Minimizes the residual sum of squares for a simple linear regression.
        Returns the parameters.
        '''
        
        b_one = ((self.x-self.x_avg)*(self.y-self.y_avg)).sum()/ \
                ((self.x-self.x_avg)**2).sum()
        
        b_zero = self.y_avg-(b_one*self.x_avg)

        return (b_zero, b_one)


    def predict(self, x):
        return self.parameters[0]+(self.parameters[1]*x)  


    def _get_rss(self):
        '''
        Given x,y returns the residual sum of squares (rss), which is the difference of the true y-value compared to
        to the prediction. Square this difference.  
        '''
        return sum(
            [
                (y_value - self.parameters[0] - (self.parameters[1]*x_value))**2
                for y_value, x_value in zip(self.y, self.x)
            ]
        )


    def _get_rse(self):
        '''
        Given the rss, returns the rse or residual standard error. This is the estimation of 
        the standard error for each observation. 
        This is used as the true standard error cannot be known. 

        Assumes that for each observation the errors are uncorrelated to the common variance.

        Interpretation:
        How much on average will the response deviate from the true regression line. 
        Measures the lack of fit of a model.
        '''
        
        # n-2 is due to the degrees of freedom being loss when trying to predict b1 and b0.
        return sqrt(
            self._get_rss()/(self.n-2)
        )
        
    def _get_r_squared(self):
        '''
        Returns the r squared of a linear regression using the residual sum of squares (RSS)
        and total sum of squares (TSS).

        The RSS looks at the total variability that remains after the regression.
        The TSS looks at the total variability that is in the response before regression. 

        R2 looks at the proportion of variability of the response Y that can be explained
        with the predictors X. 

        R2 = TSS - RSS / TSS = 1 - RSS/TSS 

        A high R2 means that the predictor X can explain a large portion of the variability
        of Y. In contrast a low R2 means that X can not explain a large portion of the 
        variability of Y due to either there being inherently a large variablity in Y, the
        model being wrong or both.
        '''
        rss = self._get_rss()

        tss = ((self.y - self.y_avg)**2).sum()

        print(rss, tss)

        return 1 - (rss/tss)


    def _get_std_error(self):
        '''
        Returns the estimation for the standard error of our predicted parameters.
        '''
        se_b0_squared = (self._get_rse()**2)*(
            (1/self.n) + (
                self.x_avg**2 / ((self.x - self.x_avg)**2).sum()
            )
        )

        se_b1_squared = self._get_rse()**2 / ((self.x - self.x_avg)**2).sum()

        return sqrt(se_b0_squared), sqrt(se_b1_squared)


    def get_confidence_intervals(self):
        '''
        Returns the 95% confidence interval for the predicted parameters. 
        Takes the form parameter +- standard error of parameter. 
        [lower bound, upper bound]
        '''

        ci_b1 = [
            round((self.parameters[1] - 2*self.std_errors[1]),3),
            round((self.parameters[1] + 2*self.std_errors[1]),3)
        ]

        ci_b0 = [
            round((self.parameters[0] - 2*self.std_errors[0]),3),
            round((self.parameters[0] + 2*self.std_errors[0]),3)
        ]

        return {
            '95% Confidence Interval For β1': ci_b1,
            '95% Confidence Interval For β0': ci_b0
        }


    def get_significance(self):
        '''
        Uses t-statistic to see how many standard deviations a parameter is from 0. 
        We can use the standard error to run a hypothesis test. 
        Null hypothesis: No relation between X and Y, Ho: B1 = 0 
        Alternative hypothesis: There is some relation between X and Y, Ha: B1 != 0.
        
        For n greater than 30, the t-distribution has a bell shape and is similar
        to the normal distribution. 

        The probability of observing |t| in this distribution is the p-value.

        Small p-value means there is a low probability that such a strong association
        between the predictor and response was due to chance. 
        '''

        # t-statistic for parameter B1
        t1 = (self.parameters[1] - 0) / self.std_errors[1]
        # t-statistic for parameter B0
        t0 = (self.parameters[0] - 0) / self.std_errors[0]
        
        # Lose two degress of freedom by having two parameters.
        degrees_of_freedom = self.n - 2

        # t.sf is the survival function or 1 - the cumulative distribution function.
        # Looks at the integral from -infinity to |t| in this case and the area under the
        # probability density function of the t-distribution gives you the cdf.
        p1 = t.sf(abs(t1),df=degrees_of_freedom)
        p2 = t.sf(abs(t0),df=degrees_of_freedom)

        return pd.DataFrame(
            {
                'Parameter': ['B0', 'B1'],
                'Coefficient': [self.parameters[0],self.parameters[1]],
                'std_error':[self.std_errors[0], self.std_errors[1]],
                't-statistic':[t0, t1],
                'p-value':[p1, p2]
            }
        )


class multi_linear_regression():
    
    def __init__(self, x, y):
        self.n = len(y)
        self.p = x.shape[1]
        self.x = self._add_intercept(np.array(x))
        self.y = np.array(y)
        self.x_avg = self._avg_x()
        self.y_avg = self.y.mean()
        self.parameters = self._minimize_rss()
    

    def _avg_x(self):
        '''
        Takes a matrix of predictors and gets the average for each predictor. 
        '''
        # axis = 0 gets the means across the columns. 
        return np.mean(a = self.x, axis=0)


    def _add_intercept(self, x):
        '''
        Need to add an intercept term to the data matrix to be able to calculate the 
        intercept for the linear model. 
        '''
        x_and_intercept = np.empty(shape=(self.n, self.p+1))
        x_and_intercept[:,0] = 1
        x_and_intercept[:,1:] = x

        return x_and_intercept


    def _get_rss(self):
        '''
        Returns the residual sum of squares for a multi linear regression. 
        '''
        predictions = self.predict(self.x)
        # Need to reshape predictions and so it can subtract element wise from y.
        residuals = (self.y - predictions.reshape(self.n))
       
        # @ = matrix multiplication. 
        # This is equivalent of sum of residuals**2
        # Should be faster than a for loop. 
        return residuals.T @ residuals
    

    def _get_rse(self): 
        '''
        Returns the residual standard errors for each of the parameters. 
        '''
        rss = self._get_rss()
        return sqrt(
            rss/(self.n - self.p)
        )


    def _get_std_errors(self):
        '''
        Get the std errors for all the parameters for a multiple linear regression. 
        The std error is represented by the variance of the error term in the linear model
        multiplied by invs(X'X).

        Can use QR decomposition to give an easier matrix to invert, by representing X'X
        with R'R
        '''
        
        # The variance of the random error can't be known so estimated with rse.
        variance_of_error_term = self._get_rse()**2
        print(variance_of_error_term)
        _, r = np.linalg.qr(self.x)

        print(r.shape)
        print(r)

        variance_of_coefficients = np.linalg.inv(r.T@r)*variance_of_error_term

        return variance_of_coefficients


    def _minimize_rss(self):
        '''
        Will use QR decomposition to minimize rss with multiple predictors. 
        This is a computationally more efficient and more stable than calculating 
        b = inverse(X'X)     *    (X'Y), where b is a column vector of the coefficients.
                    |               |
              covariance and    Covariance of X
                Variance           and y.
                 of X.
        
        What is QR decomposition?
        Factoring a matrix A into two matrices Q and R.
        Specifically:
        A = Q . R , where A is an invertible matrix we want to decompose.
                          Q is an m x m orthogonal matrix (ie. Q_trans = invs(Q)) and 
                          R is an m x n upper triangle matrix.
        
        b = invs(R) . Q_trans . Y
        invs(R) is easier to find than X_trans X.  
        numpy.qr() returns Q,R for a given matrix.

        B is a vector that represents the coefficients for the linear model. 

        '''
        q, r = np.linalg.qr(self.x)
        b = np.linalg.inv(r).dot(q.T).dot(self.y)
        return b.reshape(self.p+1,1)

    
    def predict(self, x):
        '''
        Given a predictor vector of x predict the response y with the coefficient vector b-hat. 
        '''
        return x @ self.parameters