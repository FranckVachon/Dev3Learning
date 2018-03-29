# -*- coding: utf-8 -*-
"""
Execution dans un terminal

Exemple:
   python non_linear_classification.py rbf 0

Vos Noms (Vos Matricules) .~= À MODIFIER =~.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from MAPkernel import MAPkernel


def generate_data():
    X_1_1 = np.random.randn(30, 2) + np.array([[5, 1]]) # Gaussienne centrée en mu_1_1=[5,1]
    X_1_2 = np.random.randn(20, 2) + np.array([[0, 4]]) # Gaussienne centrée en mu_1_1=[0,4]
    #X_1_2 = np.random.randn(30, 2) + np.array([[4, 4]]) # Gaussienne centrée en mu_1_1=[5,1]

    X_1 = np.vstack([X_1_1, X_1_2])
    t_1 = np.ones(X_1.shape[0])
    X_2 = np.random.randn(40, 2) + np.array([[2, 3]]) # Gaussienne centrée en mu_2=[2,3]
    t_2 = np.zeros(X_2.shape[0])

    # Fusionne toutes les données dans un seul ensemble
    X = np.vstack([X_1, X_2])
    t = np.hstack([t_1, t_2])
    return X, t


def main():

    if len(sys.argv) < 3:
        usage = "\n Usage: python non_linear_classification.py kernel_type gridSearch\
        \n\n\t kernel_type: rbf, lineaire, polynomial, sigmoidal\n\t gridSearch: 0: pas de grid search,  1: execution du grid search\n"
        print(usage)
        return

    kernel_type = sys.argv[1]
    grid_search = int(sys.argv[2])

    # On génère les données d'entrainement et de validation
    X_train, t_train = generate_data()
    X_val, t_val = generate_data()
    X_test, t_test = generate_data()

    # On entraine le modèle
    mp = MAPkernel(kernel=kernel_type)
    predictions_entrainement = None

    if grid_search == 1:
        mp.grid_search(X_train, t_train, X_val, t_val)

    mp.entrainement(X_train, t_train)
    predictions_entrainement = np.array([mp.prediction(x) for x in X_train])
    predictions_test = np.array([mp.prediction(x) for x in X_test])
    # ~= À MODIFIER =~. 
    # AJOUTER CODE AFIN DE CALCULER L'ERREUR D'APPRENTISSAGE
    # ET DE VALIDATION EN % DU NOMBRE DE POINTS MAL CLASSES
    err_train = 100 * np.sum(np.abs(predictions_entrainement - t_train)) / len(t_train)
    print("Erreur d'entrainement = ", err_train, "%")

    err_test = 100*np.sum(np.abs(predictions_test-t_test))/len(t_test)
    print("Erreur de test = ", err_test, "%")



    # Analyse des erreurs
    if (err_train < 1.0 and err_test > 8.0) or err_test-err_train > 20.0:
        print('WARNING!!! Sur-entrainement possible!')
    elif err_train > 20.0:
        print('WARNING!!! Sous-entrainement possible!')
    if err_test > 20:
        print('WARNING!!! Erreur de test anormalement élevée!')
    # Affichage
    ix = np.arange(X_test[:, 0].min(), X_test[:, 0].max(), 0.1)
    iy = np.arange(X_test[:, 1].min(), X_test[:, 1].max(), 0.1)
    iX, iY = np.meshgrid(ix, iy)
    X_vis = np.hstack([iX.reshape((-1, 1)), iY.reshape((-1, 1))])
    contour_out = np.array([mp.prediction(x) for x in X_vis])
    contour_out = contour_out.reshape(iX.shape)

    plt.contourf(iX, iY, contour_out > 0.5)
    plt.scatter(X_test[:, 0], X_test[:,1], s=(t_test+0.5)*100, c=t_val, edgecolors='y')
    plt.show()

if __name__ == "__main__":
    main()
