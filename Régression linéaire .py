#!/usr/bin/env python
# coding: utf-8

import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# # 1 Dataset

#Création de deux variables --> x et y
# x est un tableau de features
# y est un tableau de targets
x, y = make_regression(n_samples=100, n_features=1, noise=10)
#affiche x en fonction de y sur un graphique
plt.scatter(x, y)

# ##Verification des dimensions des matrices x et y
print(x.shape)
print(y.shape)

#x.shape-->(100, 1) --> OK
#y.shape --> (100,--> incomplet, il faut redefinir ses dimensions

#on redimensionne les dimensions de y
y = y.reshape(y.shape[0], 1)
print(y.shape)

#dimensions de y --> (100, 1)--> OK
#création de la matrice X
#on "colle" deux vecteurs numpy --> x et un vecteur rempli de 1 de même dimensions que x
X = np.hstack((x, np.ones(x.shape)))
X.shape
#dimensions de X --> (100, 2)
X
#vecteur theta
theta = np.random.randn(2, 1)
theta
theta.shape
#theta.shape --> (2, 1)--> OK
# # Modèle

#on crée le modèle --> f(x) = ax + b--> x--> X et a et b --> theta
#donc modele--> X.theta
def model(X, theta):
    return X.dot(theta)

#dimension du model -->(100, 1)-->OK
#on trace le modèle (provisoire) en rouge
plt.plot(x, model(X, theta),c="red")
#et par dessus le nuage de points du datatset
plt.scatter(x, y)
#on a du boulot 

# # Fonction Coût

def cost_function(X, y, theta):
    m = len(y)
    return 1/(2*m)*np.sum((model(X, theta) - y)**2)

cost_function(X, y, theta)

# # Gradients et descente de gradient

#GRADIENT
def grad(X, y, theta):
    m = len(y)
    return 1/m * X.T.dot(model(X, theta) - y)



#DESCENTE DE GRADIENT
def gradient_descent(X, y, theta, learning_rate, n_iterations):
    cost_history = np.zeros(n_iterations)
    for i in range(0, n_iterations):
        theta = theta - learning_rate * grad(X, y, theta) 
        cost_history[i] = cost_function(X, y, theta)
        
    return theta, cost_history


# # Entrainement du modele

#calcul du theta final (valeur de a et b pour lesquels la fonction cout est minimale)
theta_final, cost_history = gradient_descent(X, y, theta, learning_rate = 0.001, n_iterations=1000)

theta_final

#on verifie le resultat
predictions = model(X, theta_final)
#on affiche le dataset
plt.scatter(x, y)
#on affiche le modele entrainé en rouge
plt.plot(x, predictions, c="r")

plt.plot(range(1000), cost_history)
# In[42]:

def coef_determination(y, pred):
    u = ((y-pred)**2).sum()
    v = ((y - y.mean())**2).sum()
    return 1-u/v

coef_determination(y, predictions)