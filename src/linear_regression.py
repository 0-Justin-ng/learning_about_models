import numpy as np
from math import sqrt

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
        '''
        
        # n-2 is due to the degrees of freedom being loss when trying to predict b1 and b0.
        return sqrt(
            self._get_rss()/(self.n-2)
        )
    

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


    def check_significance(self):
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
        t2 = (self.parameters[0] - 0) / self.std_errors[0]
        
        # Lose two degress of freedom by having two parameters.
        degrees_of_freedom = self.n - 2

        print(t1, t2)
