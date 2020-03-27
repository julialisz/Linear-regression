# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial


def mean_squared_error(x, y, w):
    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''
    matrix = design_matrix(x, w.shape[0]-1)
    Q = 0
    for n in range(y.shape[0]):
        #trans = matrix.transpose()
        Q = Q + (y[n] - np.dot(matrix[n], w))**2
    err = float (Q / x.shape[0])
    return err


def design_matrix(x_train,M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''
    matrix = []
    for x in x_train:
        matrix.append([x[0]**i for i in range(M+1)])
    matrix = np.array(matrix)
    return matrix


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    mat = design_matrix(x_train, M)
    trans = mat.transpose()
    first = np.dot(trans, mat)
    inv = np.linalg.inv(first)
    second = np.dot(inv, trans)
    w = np.dot(second, y_train)
    err = mean_squared_error(x_train, y_train, w)
    return (w, err)

def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    mat = design_matrix(x_train, M)
    trans = mat.transpose()
    first = np.dot(trans, mat)
    sec = regularization_lambda * np.identity(M+1)
    inv = np.linalg.inv(first+sec)
    third = np.dot(inv, trans)
    w_reg = np.dot(third, y_train)
    err = mean_squared_error(x_train, y_train, w_reg)
    return (w_reg, err)

def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''
    model_train_arr = []
    err_mod_train_arr = []
    for i in M_values:
        w, err = least_squares(x_train, y_train, i)
        model_train_arr.append(w)
        err_mod_train_arr.append(err)
    err_mod_val_arr = []
    for i in M_values:
        mean = mean_squared_error(x_val, y_val, model_train_arr[i])
        err_mod_val_arr.append(mean)
    min_err_mod_val_index = err_mod_val_arr.index(min(err_mod_val_arr))
    train_err = err_mod_train_arr[min_err_mod_val_index]
    min_w = model_train_arr[min_err_mod_val_index]
    val_err = min(err_mod_val_arr)
    return(min_w, train_err, val_err)


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedure
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''
    train_arr = []
    for lam in lambda_values:
        w, err = regularized_least_squares(x_train, y_train, M, lam)
        train_arr.append((w,err,lam))
    val_arr = []
    for w in train_arr:
        val_arr.append(mean_squared_error(x_val, y_val, w[0]))
    val_err = min(val_arr)
    min_val_index = val_arr.index(val_err)
    w, train_err, lam = train_arr[min_val_index]
    return (w, train_err, val_err, lam)
