import numpy as np
import pandas as pd
import DiamondSquare as DS
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
import time

def main():

    map2 = np.loadtxt('map2.csv',delimiter=',')
    print(map2)
    map3= np.zeros((33*6,4))
    print(map3)
    c= 0
    for i in range(0, 33):
        for j in range (0,6):
            map3[c][0] = j +1
            map3[c][1] = i + 1
            map3[c][2] = 0
            map3[c][3] = map2[i][j]
            c=c+1

    print(map3)
    np.savetxt('map3.csv',map3, delimiter=',',fmt='%2.8f')
if __name__ == '__main__':
    main()
