
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
    map3 = np.loadtxt('map3.csv',delimiter=',')

    timestr = time.strftime("%Y%m%d-%H%M%S")
    x_index = [i for i in range(0, 6)]
    y_index = [i *10 for i in range(0, 33)]
    x_vals, y_vals = np.meshgrid(x_index, y_index)
    # print(x_vals)
    fig = plt.figure()
    p2 = fig.add_subplot(111, projection="3d")
    p2.set_title("Diamond Square 3D Surface Plot")
    p2.plot_surface(y_vals, x_vals, map2,cmap=cm.jet)
    # print(map3[:,1])
    rbfi = Rbf(map3[:,0],map3[:,1],map3[:,2],map3[:,3],function='cubic')
    # plt.savefig("3D_dS%s.png" % timestr, bbox_inches="tight")
    p2.plot_surface(rbfi(map3[:,0],map3[:,1],map3[:,2]))
    plt.show()
if __name__ == '__main__':
    main()
