import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def f(i,data,new,m):
    xi_2result = np.zeros((m,1))
    xi_1result = np.zeros((m,m))
    xi_0result = np.zeros((m,m,m))
    path = np.zeros((i,m,m))
    finalpath = np.zeros((i,1),dtype=int)
    HL = 255
    BL = 10
    k=2
    n = np.int((30*m/75))
    print("n=")
    print(n)
    finalpath[i-1] = n
    finalpath[i-2] = n
    xi_2result[:] = np.inf
    xi_2result[n] = 0
    xi_1result[:,:] = np.inf
    xi_1result[n,n] =0
    xi_0result[:,:,:] = np.inf
    while k <= i-1:
        xi_0result[:, :, :] = np.inf
        for imin2 in range (0,m):
            for imin1 in range (0,m):
                for i0 in range (0,m):
                        a = 30 - i0 * 75 / m
                        if k == 2:
                            if np.abs(data[k, 2] + a - (data[k - 1, 2] + 30 - imin1 * 75 / m)) <= HL:
                                if np.abs((data[k - 1, 2] + 30 - imin1 * 75 / m - (data[k - 2, 2] + 30 - imin2 * 75 / m)) - (data[k, 2] + a - (data[k - 1, 2] + 30 - imin1 * 75 / m))) <= BL and imin2== imin1 == n:
                                    if (c(a) + xi_0result[imin2,imin1,i0] ) <= xi_0result[imin2, imin1, i0] :
                                        xi_0result[imin2, imin1, i0] = c(a)
                                else:
                                    xi_0result[imin2, imin1, i0] = np.inf
                            else:
                                xi_0result[imin2, imin1, i0] = np.inf
                        if k == 3:
                            if np.abs(data[k, 2] + a - (data[k - 1, 2] + 30 - imin1 * 75 / m)) <= HL:
                                if np.abs((data[k - 1, 2] + 30 - imin1 * 75 / m - (data[k - 2, 2] + 30 - imin2 * 75 / m)) - (data[k, 2] + a - (data[k - 1, 2] + 30 - imin1 * 75 / m))) <= BL and imin2 == n:
                                    if (c(a) + xi_0result[imin2,imin1,i0] ) <= xi_0result[imin2, imin1, i0] :
                                        xi_0result[imin2, imin1, i0] = c(a) + xi_1result[imin2,imin1]
                                else:
                                    xi_0result[imin2, imin1, i0] = np.inf
                            else:
                                xi_0result[imin2, imin1, i0] = np.inf
                        else:
                            if np.abs(data[k,2] + a - (data[k-1,2]+ 30 - imin1 * 75 / m)) <= HL:
                                if np.abs(((data[k - 1, 2] + 30 - imin1 * 75 / m) - (data[k - 2, 2] + 30 - imin1 * 75 / m)) - (data[k, 2] + a - (data[k - 1, 2] + 30 - imin1 * 75 / m)))<=BL:
                                    if(c(a) + xi_0result[imin2,imin1,i0]) <= xi_0result[imin2,imin1,i0] :
                                        xi_0result[imin2,imin1,i0] = c(a) + xi_1result[imin2,imin1]
                                else:
                                    xi_0result[imin2, imin1, i0] = np.inf
                            else:
                                xi_0result[imin2, imin1, i0] = np.inf


        print("------------------------------------------------")
        for l in range (0,m-1):
            xi_1index = np.unravel_index(np.argmin(xi_1result[:, l], axis=None),xi_1result[:,l].shape)
            xi_2result[l]= xi_1result[xi_1index[0],l]
        for l in range(0, m - 1):
            for o in range(0, m - 1):
                xi_2index = np.unravel_index(np.argmin(xi_0result[:, l, o], axis=None), xi_0result[:, l, o].shape)
                xi_1result[l, o] = np.min(xi_0result[:, l, o])
                path[k - 2, l, o ] = xi_2index[0]

        k = k+1
    index = np.unravel_index(np.argmin(xi_0result[:, n, n], axis=None), xi_0result[:, n, n].shape)
    print(index)
    finalpath[i-3] = index[0]
    for alpha in range (3,i+1):
        print(finalpath[i-alpha+1])
        finalpath[i-alpha] = path[(i-alpha),finalpath[i-alpha+2],finalpath[i-alpha+1]]
    finalheights = np.zeros((i,1))
    for beta in range(0,i):
        finalheights[beta] = data[beta,2] + 30 - finalpath[beta] *75 /m
    return np.min(xi_0result[:,n,n],axis=None),path,finalpath,finalheights


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
    m = 40
    i = 12
    data = np.genfromtxt('datasimpel.csv',delimiter=',')
    new = np.copy(data)
    # print(result)
    print("result =", f(i,data,new,m))

if __name__ == '__main__':
    main()
