import numpy as np
import pandas as pd
import DiamondSquare as DS


def main():
    # Inputs here. Recommend ds_steps = 3 up to 11 for 2d and 3 - 5 for 3d.
    ds_steps = 5  # Number of levels. Grid points = ((ds_steps^2)+1)^2.
    max_rnd = 0.2  # Min & Max random value.
    plot_type = "2d"  # "3d" for 3d.  Makes 2d plot for any other input.
    # Inputs end.
    max_indexx = 2 ** ds_steps
    max_indexy = 6
    seeded_map = DS.f_seed_grid(2 ** ds_steps + 1, max_rnd)
    # Final_height_map = DS.f_dsmain(seeded_map, ds_steps, max_index, max_rnd)  # Calcs.
    map = np.loadtxt('Final_height_map.csv',delimiter=',')
    map2 = np.loadtxt('map2.csv',delimiter=',')

    print(map)
    print(map2)
    # print(Final_height_map)
    DS.f_plotting(map, max_indexx,max_indexy, plot_type)  # Plotting.
    DS.f_plotting(map2, max_indexx,max_indexy, plot_type)  # Plotting.



if __name__ == '__main__':
    main()
