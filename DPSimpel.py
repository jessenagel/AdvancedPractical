import numpy as np
import pandas as pd
import DiamondSquare as DS
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
import time

def f(i,data,new,m):
    U = 45
    F = 30
    HL = 255
    minimum = 0
    if i <10:
        for k in range(0,3):
            a = 30-k*75/m
            if a > 0:
                if a <= U and np.abs(data[i,2]-new[i-1,2])<= HL:
                    if c(a) + f(i+1,data,new,m) < minimum:
                        minimum = c(a) + f(i + 1, data, new, m)
                        new[i,2] =data[i,2] +a

            if a < 0:
                if -a <= F and np.abs(data[i,2]-new[i-1,2])<= HL:
                    if c(a) + f(i+1,data,new,m) < minimum:
                        minimum = c(a) + f(i + 1, data, new, m)
                        new[i,2] =data[i,2] + a
        return c(a) + f(i+1,data,new,m)
    else:
        return 0

def c(a):

    cost=1
    if (a<0):
        return a*2*cost
    if (a>0):
        return a*cost
    else:
        return 0

def minimaliseer(f):


    return 0

def main():
    m = 3
    i = 0
    data = np.genfromtxt('datasimpel.csv',delimiter=',')
    new = np.copy(data)

    print(f(i,data,new,m))

if __name__ == '__main__':
    main()
