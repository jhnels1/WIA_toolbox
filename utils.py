import numpy as np
import pandas as pd

from scipy.signal import welch, savgol_filter, find_peaks, csd, coherence

def lace_area(x, y):
    # Calculate area of polygon with nodes (x,y) via the shoelace formula
    return -0.5*np.sum(x*np.roll(y, 1, axis=-1)-np.roll(x, 1, axis=-1)*y, axis=-1)

def get_offset_dA( P, Q, range_bool ):

    # Heuristic offset estimation to allign pressure and flow traces taken at different locations

    offset_grid = np.arange(-100, 100)
    offset_stack = np.vstack( [ np.roll( Q, offset )[range_bool] for offset in offset_grid ] )
    A_comp = lace_area( offset_stack, P[np.newaxis,range_bool] )
    opt_off = offset_grid[ np.argmin( np.gradient(A_comp) ) ]

    return opt_off

def norm_trace(data):
    return (data-np.mean(data))/np.std(data)


