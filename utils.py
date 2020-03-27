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

def polynomial(x, w):
    '''
    :param x: wektor argumentow Nx1
    :param w: wektor parametrow (M+1)x1
    :return: wektor wartosci wielomianu w punktach x, Nx1
    '''
    dm = [w[i] * x**i for i in range(np.shape(w)[0])]
    return np.sum(dm,axis=0)

