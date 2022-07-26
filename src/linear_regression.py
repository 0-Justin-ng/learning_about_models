class simple_linear_regression():
    """
    Take x and y as arrays of floats.

    Try to predict Y by minimizing B0 and B1 in y = B0 + B1X,
    where X is a feature vector.
    """

    def __init__(self, x, y):
        self.x = x 
        self.y = y 

    def _minimize_rss(self):
        '''
        Minimizes the residual sum of squares for a simple linear regression.
        Returns the parameters.
        '''
        x_avg = self.x.mean()
        y_avg = self.y.mean()

        b_one = 0
        
        top = 0 
        bot = 0 
        for x_value, y_value in zip(self.x, self.y):
            top += (x_value-x_avg)*(y_value-y_avg)
            bot += (x_value-x_avg)**2
    
        b_one = top/bot
        b_zero = y_avg-(b_one*x_avg)

        return(b_zero, b_one)

    def predict(self, x):
        parameters = self._minimize_rss()
        return parameters[0]+(parameters[1]*x)  

