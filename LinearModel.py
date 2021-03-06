import numpy as np
import pandas as pd
import random

class LinearModel: 
    def __init__(self):
        self.line = np.array([(0,0), (1,1)])
        self.weights = np.array([0.0,0.0,0.0])
    
    def evalLine(self, x):
        ''' Function for evaluating the line
            using point-slope form.'''
        x0, y0 = self.line[0]
        x1, y1 = self.line[1]
        m = (y1-y0)/(x1-x0)
        return m*(x-x0)+y0
    
    def aboveLine(self, x, y):
        ''' Returns 1 if (x,y) is above the line. 
            Returns -1 if it is below or on the line.'''
        if self.evalLine(x) > y:
            return 1
        else:
            return -1
        
    def sign(self, x):
        ''' Returns the sign of the x. 
        
            Returns 1 if x > 0, -1 if x < 0, and 0 o/w.'''
        if x > 0:
            return +1
        elif x < 0:
            return -1
        return 0
    
    def create_testcase(self, N = 100, d = 2):
        ''' Creates testcases.

            Picks N points in [-1,1]x[-1,1] and a line 
            determined by two points in the same area.
            Then creates labels for the points.'''
        # Determine Points
        self.points = np.array([[1] + [random.uniform(-1,1) for i in range(d)]
                       for i in range(N)])
        # Line 
        self.line = np.array([(random.uniform(-1,1), random.uniform(-1,1)) 
                              for i in range(d)])
        # Create labels
        self.labels = np.array([self.aboveLine(point[1], point[2]) for point in self.points])
    
    def create_testcase_nonlinear(self, N = 1000):
        ''' Creates nonlinear testcases based on the target
            function f(x,y) = x^2+y^2-0.6. 
            
            Created with 10% noise.'''
        # Determine Points
        self.points = np.array([[1] + [random.uniform(-1,1) for i in range(2)]
                       for i in range(N)])
        # Line 
        self.line = np.array([(random.uniform(-1,1), random.uniform(-1,1)) 
                              for i in range(2)])
        # Create labels
        self.labels = np.array([self.sign(point[1]**2+point[2]**2-0.6) for point in self.points])
        
        # Simulate Noise
        for i in random.sample(range(N), k = N//10):
            self.labels[i] *= -1
    
    def regress(self):
        ''' Performs a linear regression and stores it in weights. 
        
            Uses the formula pseudo-inv(X)*y.'''
        X = self.points
        y = self.labels
        self.weights = np.linalg.pinv(X)@y
        return self.weights 
    
    def regress_nonlinear(self):
        ''' Performs a linear regression and stores it in weights with 
            transformed features.
        
            Uses the formula pseudo-inv(X)*y.'''
        X = np.array([[1, x1, x2, x1*x2, x1**2, x2**2] for x0, x1, x2 in self.points])
        y = self.labels
        self.weights = np.linalg.pinv(X)@y
        return self.weights    
    
    def PLA(self):
        ''' Performs the Perceptron Learning Algorithm (PLA) and stores 
            the resulting weight vector in self.weights. 
            
            Uses the update w(t+1) = w(t)+y(t)x(t) where (x(t),y(t)) is 
            misclassified.'''
        count = 0
        while True: 
            misclassified = []
            for i in range(len(self.points)):
                if self.sign(self.weights.T @ self.points[i]) != self.labels[i]:
                    misclassified.append(i)
            if len(misclassified) > 0:
                count += 1
                ind = misclassified[random.randint(0,len(misclassified)-1)]
                # Use scalar here since we're multiplying by a scalar! (not @)
                self.weights += self.labels[ind] * self.points[ind] 
            else:
                return count
    
    def logistic_regress(self):
        ''' Performs logistic regression using stochastic gradient descent. '''
        # to be implemented 
        raise NotImplementedError
    
    def evaluate_in(self):
        ''' Evaluates the performance of the regression on sample points.
        
            Returns E_in, the fraction of in-sample points incorrectly classified.'''
        count = 0
        for i in range(len(self.points)):
            if self.sign(self.weights.T@self.points[i]) != self.labels[i]:
                count += 1
        return count/len(self.points)
    
    def evaluate_in_nonlinear(self):
        ''' Evaluates the performance of linear regression with transformed 
            features on sample points. 
            
            Returns E_in, the fraction of in-sample points incorrectly classified.'''
        count = 0
        X = np.array([[1, x1, x2, x1*x2, x1**2, x2**2] for x0, x1, x2 in self.points])
        for i in range(len(X)):
            if self.sign(self.weights.T@X[i]) != self.labels[i]:
                count += 1
        return count/len(X)
    
    def evaluate_out(self, N = 1000):
        ''' Evaluates the performance of the regression on outside points.
        
            Returns E_out, the fraction of out-sample points incorrectly classified.'''
        count = 0
        outpoints = np.array([(1, random.uniform(-1,1), random.uniform(-1,1))
                       for i in range(N)])
        for pt in outpoints:
            if self.sign(self.weights.T@pt) != self.aboveLine(pt[1],pt[2]):
                count += 1
        return count/N
    
    def evaluate_out_nonlinear(self, N = 1000):
        ''' Evaluates the performance of the nonlinear regression on outside points.
        
            Returns E_out, the fraction of out-sample points incorrectly classified.'''
        count = 0
        # Create outside points with noisy labels
        outpoints = np.array([[1] + [random.uniform(-1,1) for i in range(2)]
                       for i in range(N)])
        label = np.array([self.sign(point[1]**2+point[2]**2-0.6) for point in outpoints])
        for i in random.sample(range(N), k = N//10):
            label[i] *= -1
            
        X = np.array([[1, x1, x2, x1*x2, x1**2, x2**2] for x0, x1, x2 in outpoints])
        for i in range(len(X)):
            if self.sign(self.weights.T@X[i]) != label[i]:
                count += 1
        return count/N
#Examples of Usage

# Looking at in-sample and out-sample error of Linear Regression
tot_in = 0
tot_out = 0
l = LinearDataSet()
for i in range(1000):
    l.create_testcase()
    l.regress()
    tot_in += l.evaluate_in()
    tot_out += l.evaluate_out()
print(tot_in/1000)
print(tot_out/1000)

# Look at number of iterations for Perceptron Learning Algorithm w/
# initial weights from Linear Regression

tot = 0
l = LinearDataSet()
for i in range(1000):
    l.create_testcase()
    l.regress()
    tot += l.PLA()
print(tot/1000)

