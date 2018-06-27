import numpy as np
import pandas as pd
import DiamondSquare as DS
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf
import time

def f(i,data,new,result,m):
    HL = 255
    BL = 10
    k=2
    n = np.int(np.floor(m/2-1))
    print("n=")
    print(n)
    result[:,:,:] = np.inf
    result[n,n,:] = 0
    while k <= 7:
        for i0 in range (0,m):
            for imin1 in range (0,m):
                for imin2 in range (0,m):
                        a = 30 - i0 * 75 / m
                        if k == 2:
                            if np.abs(data[k, 2] + a - data[k - 1, 2]) <= HL and np.abs((data[k - 1, 2] - data[k - 2, 2]) - (data[k, 2] + a - data[k - 1, 2])) <= BL:
                                if (c(a) + result[imin2, imin1, i0]) <= result[imin2, imin1, i0]:
                                    result[imin2, imin1, i0] = c(a)
                            else:
                                result[imin2, imin1, i0] = np.inf
                        else:
                            if np.abs(data[k,2] + a - data[k-1,2]) <= HL and np.abs((data[k-1,2]-data[k-2,2])-(data[k,2]+ a -data[k-1,2]))<=BL:
                                if(c(a) + result[imin2,imin1,i0]) <= result[imin2,imin1,i0] :
                                    result[imin2,imin1,i0] = c(a) + np.min(result[imin2,:,:])
                            else:
                                result[imin2, imin1, i0] = np.inf


        index = np.unravel_index(np.argmin(result[:,:,:],axis=None),result[:,:,:].shape)
        print(index)
        print("------------------------------------------------")
        data[k - 2, 2] = data[k - 2, 2] + 30 - (index[2] ) *75 /m
        for l in range (0,m):
            result[l,:,:]= result[np.unravel_index(np.argmin(result[:,:,l],axis=None),result[:,:,l].shape)[0],:,:]
            result[:,l,:]=result[np.unravel_index(np.argmin(result[:,:,l],axis=None),result[:,:,l].shape)[0],:,l]
        k = k+1
        # print(result[0])
    np.savetxt("argmin.csv",data[:,2],delimiter=',')
    np.savetxt("output.csv",result[9],delimiter=',')
    return result


def c(a):

    cost=1
    if a < 0:
        return -a*2*cost
    if a > 0:
        return a*cost
    else:
        return 0

def minimaliseer(f):


    return 0

def main():
    m = 20
    i = 12
    data = np.genfromtxt('datasimpel.csv',delimiter=',')
    new = np.copy(data)
    result = np.zeros((m,m,m))
    # print(result)
    print(f(i,data,new,result,m))

if __name__ == '__main__':
    main()
