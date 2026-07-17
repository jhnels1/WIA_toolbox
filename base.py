import numpy as np
import pandas as pd

import utils

from scipy.signal import welch, savgol_filter, find_peaks, csd, coherence

class Experiment:

    def __init__(self, t_data, p_data, q_data, fs=1000, slice_len=20):

        self.t_data = t_data
        self.p_data = p_data
        self.q_data = q_data
        self.fs = fs
        self.slice_len = slice_len

        self.p0 = savgol_filter(p_data, int(50e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.p1 = savgol_filter(p_data, int(50e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.p2 = savgol_filter(p_data, int(50e-3*fs)+1, 3, deriv=2, delta=1./fs)

        self.q0 = savgol_filter(q_data, int(50e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.q1 = savgol_filter(q_data, int(50e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.q2 = savgol_filter(q_data, int(50e-3*fs)+1, 3, deriv=2, delta=1./fs)


    def get_slice_bools(self):
        # Record list of bools which will break the data into chunks of the desired duration
        tmin = np.amin(self.t_data)
        tmax = np.amax(self.t_data)
        window_edges = np.arange(tmin, tmax, self.slice_len)

        slice_bool_list = list()
        slice_live_list = list()
        for i in range(len(window_edges)-1):
            slice_bool = (self.t_data>=window_edges[i])&(self.t_data<window_edges[i])
            slice_bool_list.append( slice_bool )
            slice_live_list.append( np.sum(slice_bool)/self.fs )
        self.slice_bools = slice_bool_list
        self.slice_lives = slice_live_list


    def offset_proc(self):
        offset_grid = np.zeros(len(self.slice_bools))
        for i, slice_bool in enumerate(self.slice_bools):
            offset_grid[i] = utils.get_offset_dA( self.p0, self.q0, slice_bool )
        self.slice_offsets = offset_grid

    def coher_proc(self): 
        coher_grid = np.zeros(len(self.slice_bools))
        N = int(0.5*self.slice_len*self.fs)
        for i, slice_bool in enumerate(self.slice_bools):
            frq1, coher = coherence( self.p0[slice_bool], self.q0[slice_bool], fs=self.fs, nperseg=N )
            frq2, Ppp = welch( self.p0[slice_bool], fs=self.fs, nperseg=N )
            
            frq_range = frq2<25
            
            coher_grid[i] = coher[frq_range][ np.argmax(Ppp[frq_range]) ]
        
        self.slice_coher = coher_grid
