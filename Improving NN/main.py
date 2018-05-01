<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:20:15 2018

@author: Ehsan
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from util import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from util import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec



plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#______________________________________________________________________________
train_X, train_Y, test_X, test_Y = load_dataset()


def model(X,Y, learning_rate= 0.01, num_iteration = 15000,  initialization = 'He', print_cost = False):
      
      grads = {}
      costs = []
      layer_dims = [X.shape[0], 10, 5, 1]
      
      if initialization.lower() == 'zeros':
            params =  initialize_parameters_zeros(layer_dims)
      elif initialization.lower() == 'random':
            params = initialize_parameters_random(layer_dims)
      elif initialization.lower() == 'he':
            params = initialize_parameters_he(layer_dims)
            
            
      for i in range(0,num_iteration):      
            a3, cache = forward_propagation(X,params)
            cost = compute_loss(a3, Y)
            grads = backward_propagation(X,Y,cache)
            params = update_parameters(params, grads, learning_rate)
            
            if i % 1000 == 0 and print_cost:
                  print('cost after {} iterations : {}'.format(i, cost))
                  costs.append(cost)
      
      plt.plot(costs)
      plt.ylabel('cost')
      plt.xlabel('iterations')
      plt.title('learning_rate =' + str(learning_rate))
      plt.show()
      
      return params


def initialize_parameters_zeros(layer_dims):
      
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            params['b' + str(l)] = np.zeros((layer_dims[l] , 1))
            
      return params

'''
params = model (train_X, train_Y, initialization='zeros', print_cost=True)

print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)
'''

def initialize_parameters_random(layer_dims):
      np.random.seed(3)
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*10
            params['b'+ str(l)] = np.zeros((layer_dims[l], 1))
      return params

'''
params = model (train_X, train_Y, initialization='random', print_cost=True)
print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
'''


def initialize_parameters_he(layer_dims):
      np.random.seed(3)
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
            params['b'+ str(l)] = np.zeros((layer_dims[l], 1))
      return params

params = model (train_X, train_Y, initialization='he', print_cost=True)
print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
=======
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 22:20:15 2018

@author: Ehsan
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from util import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from util import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec



plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
#______________________________________________________________________________
train_X, train_Y, test_X, test_Y = load_dataset()


def model(X,Y, learning_rate= 0.01, num_iteration = 15000,  initialization = 'He', print_cost = False):
      
      grads = {}
      costs = []
      layer_dims = [X.shape[0], 10, 5, 1]
      
      if initialization.lower() == 'zeros':
            params =  initialize_parameters_zeros(layer_dims)
      elif initialization.lower() == 'random':
            params = initialize_parameters_random(layer_dims)
      elif initialization.lower() == 'he':
            params = initialize_parameters_he(layer_dims)
            
            
      for i in range(0,num_iteration):      
            a3, cache = forward_propagation(X,params)
            cost = compute_loss(a3, Y)
            grads = backward_propagation(X,Y,cache)
            params = update_parameters(params, grads, learning_rate)
            
            if i % 1000 == 0 and print_cost:
                  print('cost after {} iterations : {}'.format(i, cost))
                  costs.append(cost)
      
      plt.plot(costs)
      plt.ylabel('cost')
      plt.xlabel('iterations')
      plt.title('learning_rate =' + str(learning_rate))
      plt.show()
      
      return params


def initialize_parameters_zeros(layer_dims):
      
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))
            params['b' + str(l)] = np.zeros((layer_dims[l] , 1))
            
      return params

'''
params = model (train_X, train_Y, initialization='zeros', print_cost=True)

print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)
'''

def initialize_parameters_random(layer_dims):
      np.random.seed(3)
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*10
            params['b'+ str(l)] = np.zeros((layer_dims[l], 1))
      return params

'''
params = model (train_X, train_Y, initialization='random', print_cost=True)
print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
'''


def initialize_parameters_he(layer_dims):
      np.random.seed(3)
      params = {}
      L = len(layer_dims)
      
      for l in range(1,L):
            params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*np.sqrt(2/layer_dims[l-1])
            params['b'+ str(l)] = np.zeros((layer_dims[l], 1))
      return params

params = model (train_X, train_Y, initialization='he', print_cost=True)
print('on the train set')
predictions_train = predict(train_X, train_Y, params)
print('on the test set')
predictions_train = predict(test_X, test_Y, params)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(params, x.T), train_X, train_Y)
>>>>>>> e094fa8b4ec99deaa5077849cd83980586f2729b
