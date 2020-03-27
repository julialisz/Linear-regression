# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# ----------------------- DO NOT MODIFY THIS FILE --------------------------
# --------------------------------------------------------------------------

import numpy as np
import pickle
import matplotlib.pyplot as plt
from content import least_squares
from content import model_selection
from content import regularized_model_selection
from utils import polynomial
from test import TestRunner
from time import sleep
import warnings


def target_output(x):
    return np.sin(2 * np.pi * x)

def plot_model(subplot, x_train, y_train, x, y_obj, y_model, x_val=None, y_val=None, train_err=None, val_err=None):
    x_min = np.min([np.min(x_train), np.min(x)])
    x_max = np.max([np.max(x_train), np.max(x)])
    y_min = -1.5
    y_max = 1.5
    int_x = x_max - x_min
    x_beg = x_min - int_x / 14.0
    x_end = x_max + int_x / 14.0
    x_ticks = [x_min,x_max]
    int_y = y_max - y_min
    y_ticks = [y_min, y_min+0.5*int_y, y_max]

    sub.set_xlim(x_beg, x_end)
    sub.set_ylim(1.1*y_min, 1.1*y_max)
    sub.set_xticks(x_ticks)
    sub.set_yticks(y_ticks)
    sub.plot(x_train,y_train,'o',markerfacecolor='none',markeredgecolor='blue',markersize=8,markeredgewidth=2)
    sub.plot(x, y_obj,'-g',linewidth=2)
    sub.plot(x, y_model, '-r', linewidth=2)
    if x_val is not None and y_val is not None:
        sub.plot(x_val, y_val, 'o', markerfacecolor='none', markeredgecolor='red', markersize=8, markeredgewidth=2)
    if train_err is not None and val_err is not None:
        sub.text(0, -1.3,'Train error: {0:.5f}\nVal error:    {1:.5f}'.format(train_err,val_err),bbox={'facecolor': 'none', 'pad': 10})

if __name__ == "__main__":
    # Ignorowanie warningow
    warnings.filterwarnings("ignore")

    # Odpalenie testow
    test_runner = TestRunner()
    results = test_runner.run()

    if results.failures:
        exit()
    sleep(0.1)

    # Ladowanie danych
    data = pickle.load(open('data.pkl', mode='rb'))
    x_plot = np.arange(0,1.01,0.01)
    y_obj = target_output(x_plot)

    # Dopasowanie wielomianow metoda najmniejszych kwadratow
    print('\n--- Least squares fitting ---')
    print('-------------- Number of training points N=8. --------------')
    fig = plt.figure(figsize=(12,6),num='Least squares fit N=8')

    for i in range(8):
        w,err = least_squares(data['x_train_8'],data['y_train_8'],i)
        y_model = polynomial(x_plot,w)
        sub = fig.add_subplot(2,4,i+1)
        plot_model(sub,data['x_train_8'],data['y_train_8'],x_plot,y_obj,y_model)
        sub.set_title("M = {}".format(i))

    plt.tight_layout()
    plt.draw()
    print('\n--- Press a key to continue ---')
    plt.waitforbuttonpress(0)

    print('\n--- Least squares fitting ---')
    print('-------------- Number of training points N=50. --------------')
    fig = plt.figure(figsize=(12, 6),num='Least squares fit for N=50')

    for i in range(8):
        w, err = least_squares(data['x_train_50'], data['y_train_50'], i)
        y_model = polynomial(x_plot, w)
        sub = fig.add_subplot(2, 4, i + 1)
        plot_model(sub, data['x_train_50'], data['y_train_50'], x_plot, y_obj, y_model)
        sub.set_title("M = {}".format(i))

    plt.tight_layout()
    plt.draw()
    print('\n--- Press a key to continue ---')
    plt.waitforbuttonpress(0)

    # Selekcja modelu
    print('\n--- Model selection for linear regression with no regularization ---')
    print('---------------- Polynomial degrees M=0,...,7 ----------------')
    print('- Number of training poitns N=50. Number of validation points N=20 -')

    M_values = range(0,7)
    w,train_err,val_err = model_selection(data['x_train_50'], data['y_train_50'],data['x_val_20'], data['y_val_20'], M_values)
    M = np.shape(w)[0]-1
    y_model = polynomial(x_plot, w)

    fig = plt.figure(figsize=(6, 5),num='Model selelection for M')
    sub = fig.add_subplot(1,1,1)
    sub.set_title('Best M={}'.format(M))
    plot_model(sub, data['x_train_50'], data['y_train_50'], x_plot, y_obj, y_model,data['x_val_20'], data['y_val_20'],train_err,val_err)

    plt.tight_layout()
    plt.draw()
    print('\n--- Press a key to continue ---')
    plt.waitforbuttonpress(0)

    print('\n--- Model selection for linear regression with regularization ---')
    print('-- Polynomial degree M=7. Number of training points N=50. Number of validation points N=20 --')

    M = 7
    lambdas = [0.001,0.003,0.01,0.03,0.1,0.3,1,3,10,30,100,300]
    w, train_err, val_err, best_lambda = regularized_model_selection(data['x_train_50'], data['y_train_50'], data['x_val_20'], data['y_val_20'],
                                            M, lambdas)
    y_model = polynomial(x_plot, w)

    fig = plt.figure(figsize=(6, 5), num='Model selection for regularization coefficient')
    sub = fig.add_subplot(1, 1, 1)
    sub.set_title('M={}    Best $\lambda$={}'.format(M,best_lambda))
    plot_model(sub, data['x_train_50'], data['y_train_50'], x_plot, y_obj, y_model, data['x_val_20'], data['y_val_20'],
               train_err, val_err)

    plt.tight_layout()
    plt.draw()
    print('\n--- Press a key to continue ---')
    plt.waitforbuttonpress(0)