import word2number
from word2number import w2n
import numpy as np
from cmath import sqrt
import math
import pandas as pd
from matplotlib import pyplot as plt
from importlib import reload
import seaborn as sb


def regression_equ(x_array, y_array, linear=True):
    x_fourth = x_array**4
    x_third = x_array**3
    x_squared = x_array**2
    y1x1 = y_array * x_array
    y1x2 = y_array * x_squared
    N = len(x_array)
    if linear:
        left_matrix = [[sum(x_squared), sum(x_array)], [sum(x_array), N]]
        right_matrix = [[sum(y1x1)], [sum(y_array)]]
        inv_mat_left = np.linalg.inv(left_matrix)
        solution = np.dot(inv_mat_left, right_matrix)
        slope, y_int = solution[0], solution[1]
        return (slope, y_int)
    else:
        left_matrix = [[sum(x_fourth), sum(x_third), sum(x_squared)], [sum(x_third), sum(x_squared), sum(x_array)], [sum(x_squared), sum(x_array), N]]
        right_matrix = [[sum(y1x2)], [sum(y1x1)], [sum(y_array)]]
        inv_mat_left = np.linalg.inv(left_matrix)
        solution = np.dot(inv_mat_left, right_matrix)
        x2_coeff, x_coeff, con_coeff = solution[0], solution[1], solution[2]
        return(x2_coeff, x_coeff, con_coeff)


def scatter_quadratic(x_list, y_list, x2_coeff, x_coeff, con_coeff, n_dev, r_std, name):
    new_y = []
    l_bound = []
    u_bound = []
    for val in x_list:
        new_y.append((x2_coeff * val**2) + (x_coeff * val) + con_coeff)
        u_bound.append((x2_coeff * val**2) + (x_coeff * val) + con_coeff + (n_dev * r_std))
        l_bound.append((x2_coeff * val**2) + (x_coeff * val) + con_coeff - (n_dev * r_std))

    plt.plot(x_list, new_y, '-r')
    plt.plot(x_list, l_bound, '--r')
    plt.plot(x_list, u_bound, '--r')
    plt.scatter(x_list, y_list)
    plt.title(f'{name}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def mean(data):
    total = sum(data)
    m = total / len(data)
    return m

def variance(data):
    new_list = [(val - mean(data)) ** 2 for val in data]
    v = mean(new_list)
    return v

def stand_dev(data):
    v = variance(data)
    s = math.sqrt(v)
    return s

def comp_residual(x, y, x2_c, x_c, c_c):
    val_list = []
    r_list = []
    for val in x:
        val_list.append(x2_c * val**2 + x_c * val + c_c)
    for i in range(len(y)):
        r_list.append(y[i] - val_list[i])

    r_mean = mean(r_list)
    r_std = stand_dev(r_list)
    return(r_list, r_mean, r_std)

x = np.array([-1.0,0.0,1.0,2.0,3.0,4.0,5.0,6.0])
y = np.array([10.0,6.0,2.0,1.0,0.0,2.0,4.0,7.0])

x2_coeff, x_coeff, con_coeff = regression_equ(x, y, False)

r_list, r_mean, r_std = comp_residual(x, y, x2_coeff, x_coeff, con_coeff)

scatter_quadratic(x, y, x2_coeff, x_coeff, con_coeff, 2, r_std, 'QUADRATIC GRAPH')






