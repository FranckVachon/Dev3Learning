# -*- coding: utf-8 -*-

#####
# Vos Noms (Vos Matricules) .~= À MODIFIER =~.
###

import numpy as np

LIN = "lineaire"
RBF = "rbf"
SIG = "sigmoidal"
POL = "polynomial"


class MAPkernel:
    def __init__(self, lamb=0.001, sigma_2=0.06, b=1.0, c=0.1, d=1.0, M=2, kernel='rbf'):
        self.kernel = self.chooseKernel(lamb, sigma_2, b, c, d, M, kernel)
        self.lamb = lamb
        self.X_train = None
        print("kernel selected:", type(self.kernel).__name__)
    def entrainement(self, X_train, t_train):
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

        #compute K-Gram:
        kgram = np.dot(X_train,X_train.T)

        #build identity matrix to build the regulation factor
        identity = np.identity(len(t_train))
        regulation_matrix = identity*self.lamb

        invA = np.linalg.inv(kgram + regulation_matrix)
        self.X_train = np.dot(invA,t_train)
        pass




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
        return 1

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
    def __init__(self, lamb=0.001, sigma_2=0.06, b=1.0, c=0.1, d=1.0, M=2, kernel='rbf'):
        self.lamb = lamb
        self.a = None
        self.sigma_2 = sigma_2
        self.M = M
        self.b = b
        self.c = c
        self.d = d

    def whichKernel(self):
        print("name: ", self.kernelName )

class RBFK(Kernel):
    def __init__(self, lamb, sigma_2):
        super().__init__(lamb, sigma_2)

class Lineaire(Kernel):
    def __init__(self,lamb):
        super().__init__(lamb)


class Polynomial(Kernel):
    def __init__(self,lamb, c, M):
        super().__init__(lamb, c, M)


class Sigmoidal(Kernel):
    def __init__(self,lamb, b, d):
        super().__init__(lamb, b, d)

