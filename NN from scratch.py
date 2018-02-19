# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 07:35:36 2018

@author: Carl Steyn
"""
import numpy as np

class NeuralNet(object):
    def __init__(self):
        self.input_size = 2
        self.output_size = 1
        self.hiddenlayer_size = 3
    
        self.W1 = np.random.randn(self.input_size,self.hiddenlayer_size)
        self.W2 = np.random.randn(self.hiddenlayer_size,self.output_size)
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def forward(self,X):
        self.z2 = np.dot(X,self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2,self.W2)
        yh = self.sigmoid(self.z3)
        return yh
    
    def sigmoidprime(self,z):
        return np.exp(-z)/((1+np.exp(-z))**2)
    
    def costfunc(self,x,y):
        e = 0.5*sum((y - self.forward(x))**2)
        return e
    
    def costfuncprime(self,x,y):
        self.yh = self.forward(x)
        delta3 = np.multiply(-(y-self.yh),self.sigmoidprime(self.z3))
        dJdW2 = np.dot(self.a2.T,delta3)
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidprime(self.z2)
        dJdW1 = np.dot(x.T,delta2)
        return dJdW1, dJdW2

    def train(self,x,y):
        yh = self.forward(X)
        scalar = 2
        cost = 10
        while cost>0.001:
            dJdW1 , dJdW2 = self.costfuncprime(x,y)
            self.W1 = self.W1 - scalar*dJdW1
            self.W2 = self.W2 - scalar*dJdW2
            yh = self.forward(x)
            cost
        return yh , cost
#%%    
NN = NeuralNet()
X = np.random.randint(0,10,(4,2))
y = np.random.uniform(size=(4,1))
yh , cost = NN.train(X,y)
