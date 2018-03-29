# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import math

LIN = "lineaire"
RBF = "rbf"
SIG = "sigmoidal"
POL = "polynomial"


class MAPkernel:
    def __init__(self, lamb=0.00001, sigma_2=0.51, b=0.09, c=0.1, d=0.09, M=2, kernel='rbf'):
        self.kernel = self.chooseKernel(lamb, sigma_2, b, c, d, M, kernel)
        self.X_train = None
        self.a = None
        self.t_train = None
        self.kgram = None
        print("kernel selected:", type(self.kernel).__name__)
    def entrainement(self, X_train, t_train):
        #Training self.a
        """
        Entraîne une méthode d'apprentissage à noyau de type Maximum a
        posteriori (MAP) avec un terme d'attache aux données de type
        "moindre carrés" et un terme de lissage quadratique (voir
        Eq.(1.67) et Eq.(6.2) du livre de Bishop).  La variable X_train
        contient les entrées (un tableau 2D Numpy, où la n-ième rangée
        correspond à l'entrée x_n) et des cibles t_train (un tableau 1D Numpy
        où le n-ième élément correspond à la cible t_n).

        L'entraînement doit utiliser un noyau de type RBF, lineaire, sigmoidal,
        ou polynomial (spécifié par ''self.kernel'') et dont les parametres
        sont contenus dans les variables self.sigma_2, self.c, self.b, self.d
        et self.M et un poids de régularisation spécifié par ``self.lamb``.

        Cette méthode doit assigner le champs ``self.a`` tel que spécifié à
        l'équation 6.8 du livre de Bishop et garder en mémoire les données
        d'apprentissage dans ``self.X_train``

        .~= À MODIFIER =~.
        """
        #To keep training data
        self.X_train = X_train
        self.t_train = t_train


        #compute K-Gram:
        self.kgram = np.dot(X_train,X_train.T)

        """
        #debug to see what np.dot really calculates - this is identical to calculation performed by np.dot()
        n,m = X_train.shape
        kgrammanual = np.zeros((n,n))
        for i in range(0,n):
            for j in range(0,n):
                kgrammanual[i][j] = np.dot(X_train[i],X_train[j])
        """

        #identity matrix to build the regulation factor
        identity = np.identity(len(t_train))
        #regulation_matrix = identity*self.kernel.b
        #THis should be self.kernel.lamb.... no?
        regulation_matrix = identity*self.kernel.lamb

        #compute a
        invA = np.linalg.inv(self.kgram + regulation_matrix)
        self.a = np.dot(invA,t_train)


    def prediction(self, x):
        """
        Retourne la prédiction pour une entrée representée par un tableau
        1D Numpy ``x``.

        Cette méthode suppose que la méthode ``entrainement()`` a préalablement
        été appelée. Elle doit utiliser le champs ``self.a`` afin de calculer
        la prédiction y(x) (équation 6.9 de Bishop).

        NOTE : Puisque nous utilisons cette classe pour faire de la
        classification binaire, la prediction est +1 lorsque y(x)>0.5 et 0
        sinon
        .~= À MODIFIER =~.
        """
        #t_train just for debugging, useful to see values
        pred = self.kernel.calculatePrediction(a=self.a, X_train=self.X_train, x=x, t_train=self.t_train)
        sol = 1.0 if pred>0.5 else 0
        return sol

    def erreur(self, t, prediction):
        """
        Retourne l'erreur de la différence au carré entre
        la cible ``t`` et la prédiction ``prediction``.
        .~= À MODIFIER =~.
        """
        return (t-prediction)**2

    def grid_search(self, X_train, t_train, X_val, t_val):
        """
        Cette fonction trouve les meilleurs hyperparametres ``self.sigma_2``,
        ``self.c`` et ``self.M`` (tout dépendant du noyau selectionné) et
        ``self.lamb`` en entraînant sur (X_train, t_train) et en validant sur
        (X_val,t_val).

        NOTE: Les valeurs de ``self.sigma_2`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6
        
        .~= À MODIFIER =~.
        """
        self.kernel.gridSearch(self, X_train, t_train, X_val, t_val)

        return -1
    def chooseKernel(self,lamb, sigma_2, b, c, d, M, kernel):
        if kernel == LIN:
            return Lineaire(lamb)
        elif kernel == RBF:
            return RBFK(lamb, sigma_2)
        elif kernel == SIG:
            return Sigmoidal(lamb, b, d)
        elif kernel == POL:
            return Polynomial(lamb, c, M)
        else:
            print("Invalid kernel selection - ")

"""
######### Classes for easy Kernel management#############

Classes below used so that the main algo above can work with any kernel. Could easily add more

"""


class Kernel:
    "Parent class for different kernels"
    #rbf, lineaire, polynomial, sigmoidal
    def __init__(self, lamb=0.00001, sigma_2=0.6, b=1.0, c=0.1, d=1.0, M=2, kernel='rbf'):
        self.lamb = lamb
        self.sigma_2 = sigma_2
        self.M = M
        self.b = b
        self.c = c
        self.d = d
        self.gridSearchIter = 500
        #dictionary for each kernel. then [param, minvalue, maxvalue]
        self.gridSearchParams = {RBF:[[self.d, 0.000000001, 2], [self.b, 0.000000001, 2]], POL:[[self.c, 0, 5], [self, M, 2, 6]], LIN:[], SIG:[[self.b, 0.00001, 0.01], [self.d, 0.00001, 0.01]]}


    def calculatePrediction(self, a, X_train,x, t_train):
        """each kernel implements its calculation"""
        pass

    def gridSearch(self, mp, X_train, t_train, X_val, t_val):
        """
        NOTE: Les valeurs de ``self.sigma_2`` et ``self.lamb`` à explorer vont
        de 0.000000001 à 2, les valeurs de ``self.c`` de 0 à 5, les valeurs
        de ''self.b'' et ''self.d'' de 0.00001 à 0.01 et ``self.M`` de 2 à 6

        Without having read all the paper, it's an interesting take on it so I decided to give this a try...
        #http://jmlr.csail.mit.edu/papers/volume13/bergstra12a/bergstra12a.pdf
        """

        pass



class RBFK(Kernel):
    def __init__(self, lamb, sigma_2):
        super().__init__(lamb, sigma_2)
    def calculatePrediction(self, a, X_train, x, t_train):

        #Squared distances
        pred = np.sum((X_train - x) ** 2,axis=1)
        #sigma_2. I checked and the above seem to work

        pred *= (-1/(self.sigma_2 / 2))
        #expotentional in-place
        np.exp(pred,pred)

        pred = np.inner(pred.T, a)

        return pred

    def gridSearch(self, mp, X_train, t_train, X_val, t_val):
        bestValidation = 100.0
        #struc: [bestValue, minValue, maxValue]
        lamb = [0.0, 0.000000001, 2.0 ]
        sigma = [0.0, 0.000000001, 2.0 ]

        #storing values to see which ones are generated
        lamb_values = []
        sigma_values = []


        for i in range(0,self.gridSearchIter):

            self.lamb = generateRandomParams(lamb)
            self.sigma_2 = generateRandomParams(sigma)
            mp.entrainement(X_train, t_train)

            #get results
            predictions_validate= np.array([mp.prediction(x) for x in X_val])
            predictions_entrainement = np.array([mp.prediction(x) for x in X_train])

            #Now check error rates
            err_train = 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train)
            err_Val= 100 * np.sum(np.abs(predictions_validate - t_val)) / len(t_val)
            if bestValidation>err_Val:
                print("PB - validation:",err_Val, " training: ", err_train, " lamb: ", self.lamb, " sigma:", self.sigma_2)
                bestValidation = err_Val
                lamb[0] = self.lamb
                sigma[0] = self.sigma_2

            lamb_values.append(self.lamb)
            sigma_values.append(self.sigma_2)

        self.sigma_2 = sigma[0]
        self.lamb = lamb[0]

        print("Final hypers - lamb:", self.lamb, " sigma:", self.sigma_2)

        #####################################################################
        ############Additionals stuff - checking out the generate hyparam
        #####################################################################


        # Affichage

        plt.scatter(lamb_values, sigma_values, edgecolors='y')
        plt.show()

class Lineaire(Kernel):
    def __init__(self,lamb):
        super().__init__(lamb)

    def calculatePrediction(self, a, X_train, x, t_train):
        #we want to x (1 instance of m params datapoint) with X_train, [NxN] containing all training data

         #scalar product of x, and each individual rows in X_train, the multiply by a[i]
        calc = np.array([np.dot(x, X_train[i].T) for i in range(0,len(X_train))])
        pred = np.inner(calc.T, a)
        return  pred

    def gridSearch(self, mp, X_train, t_train, X_val, t_val):
        bestValidation = 100.0
        #struc: [bestValue, minValue, maxValue]
        lamb = [0.0, 0.000000001, 2.0 ]
        for i in range(0,self.gridSearchIter):

            self.lamb = generateRandomParams(lamb)
            mp.entrainement(X_train, t_train)

            #get results
            predictions_validate= np.array([mp.prediction(x) for x in X_val])
            predictions_entrainement = np.array([mp.prediction(x) for x in X_train])
            #Now check error rates
            err_train = 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train)
            err_Val= 100 * np.sum(np.abs(predictions_validate - t_val)) / len(t_val)
            if bestValidation>err_Val:
                print("PB - validation:", err_Val, " training: ", err_train, " lamb: ", self.lamb, " sigma:", self.d, " iter #:", i)
                bestValidation = err_Val
                lamb[0] = self.lamb



        self.lamb = lamb[0]

        print("Final hypers - lamb:", self.lamb)





class Polynomial(Kernel):
    def __init__(self,lamb, c, M):
        super().__init__(lamb, c, M)


class Sigmoidal(Kernel):
    def __init__(self,lamb, b, d):
        super().__init__(lamb, b, d)

    def zcalculatePrediction(self, a, X_train, x, t_train):

        calc = np.array([np.dot(x, X_train[i].T) for i in range(0,len(X_train))])
        pred = np.inner(calc.T, a)
        return  pred

    def calculatePrediction(self, a, X_train, x, t_train):

        #as with linear one
        calc1 = np.array([np.dot(x, X_train[i].T) for i in range(0,len(X_train))])
        calc2 = np.multiply(calc1, self.b)

        calc3 = np.add(calc2, self.d)

        #tanh
        #  in-place
        calc4 = np.tanh(calc3)

        pred = np.inner(calc4.T, a)
        return pred


    def zgridSearch(self, mp, X_train, t_train, X_val, t_val):
        bestValidation = 100.0
        #struc: [bestValue, minValue, maxValue]
        b = [0.00001, 0.00001, 0.01 ]
        d = [0.00001, 0.00001, 0.01]
        lamb = [0.000000001, 0.000000001, 2.0]
        numSteps = 1000



        #for i in np.arange(b[1], b[2], (b[2] - b[1])/numSteps):
            #for j in np.arange(d[1], d[2], (d[2] - d[1]) / numSteps):
        for k in np.arange(lamb[1], lamb[2], (lamb[2] - lamb[1]) / numSteps):

            self.b = k
            mp.entrainement(X_train, t_train)

            #get results
            predictions_validate= np.array([mp.prediction(x) for x in X_val])
            predictions_entrainement = np.array([mp.prediction(x) for x in X_train])

            #Now check error rates
            err_train = 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train)
            err_Val= 100 * np.sum(np.abs(predictions_validate - t_val)) / len(t_val)
            #print("PB - validation:", err_Val, " training: ", err_train, " b: ", self.b, " d:", self.d)

            if bestValidation>err_Val:
                bestValidation = err_Val
                lamb[0] = self.b
            #print("err_val: ", err_Val, " lamb: ", self.lamb)

        self.b = lamb[0]

        print("Final hypers - lamb:", self.b)


    def gridSearch(self, mp, X_train, t_train, X_val, t_val):
        bestValidation = 100.0
        #struc: [bestValue, minValue, maxValue]
        lamb = [0.0, 0.000000001, 2.0 ]
        #b = [0.0, 0.00001, 0.01 ]
        b = [0.0, 0.00001, 0.01 ]
        d = [0.0, 0.00001, 0.01 ]


        #storing values to see which ones are generated
        b_vals = []
        d_vals = []


        for i in range(0,self.gridSearchIter):
            self.lamb = generateRandomParams(lamb)
            self.b = generateRandomParams2(b)
            self.d = generateRandomParams2(d)
            mp.entrainement(X_train, t_train)

            #get results
            predictions_validate= np.array([mp.prediction(x) for x in X_val])
            predictions_entrainement = np.array([mp.prediction(x) for x in X_train])

            #Now check error rates
            err_train = 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train)
            err_Val= 100 * np.sum(np.abs(predictions_validate - t_val)) / len(t_val)
            #print("PB - validation:", err_Val, " training: ", err_train, " b: ", self.b, " d:", " lamb ", self.lamb)

            if bestValidation>err_Val:
                bestValidation = err_Val
                lamb[0] = self.lamb
                b[0] = self.b
                d[0] = self.d

            b_vals.append(self.b)
            d_vals.append(self.d)

        self.lamb = lamb[0]
        self.b = b[0]
        self.d = d[0]

        print("Final hypers - b:", self.b, " d:", self.d, " lamb ", self.lamb)

        #####################################################################
        ############Additionals stuff - checking out the generate hyparam
        #####################################################################


        # Affichage

        plt.scatter(b_vals, d_vals, edgecolors='y')
        plt.show()

def generateRandomParams(args):
    return (rd.random()**2)*args[2]+args[1]

def generateRandomParams2(args):
    return (rd.random())*args[2]+args[1]