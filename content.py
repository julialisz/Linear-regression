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
    pass


def design_matrix(x_train,M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''
    pass


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    pass


def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    pass


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
    pass


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''
    pass