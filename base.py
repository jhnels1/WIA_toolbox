import numpy as np
import pandas as pd

import utils

from scipy.signal import welch, savgol_filter, find_peaks, csd, coherence

class Pulse:
    def __init__(self, t, p0, p1, p2, q0, q1, q2, fs=1000):
        self.t=t
        self.p0=p0
        self.p1=p1
        self.p2=p2
        self.q0=q0
        self.q1=q1
        self.q2=q2
        self.fs=fs

        self.MAP = np.mean(p0)
        self.PP = np.amax(p0)-p0[0]
        self.MAQ = np.mean(q0)
        self.PQ = np.amax(q0)-q0[0]
        self.RI = self.PQ/np.amax(q0[0])
        self.PI = (np.amax(q0)-np.amin(q0))/np.mean(q0)
        self.RR = np.amax(t)-np.amin(t)
        self.t0 = np.amin(t)

class Experiment:

    def __init__(self, t_data, p_data, q_data, fs=1000, slice_len=20):

        # Store the data
        self.t_data = np.array(t_data)
        self.p_data = np.array(p_data)
        self.q_data = np.array(q_data)
        self.fs = fs
        self.slice_len = slice_len

        # also store filters+derivatives of pressure and flow
        self.p0 = savgol_filter(self.p_data, int(50e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.p1 = savgol_filter(self.p_data, int(50e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.p2 = savgol_filter(self.p_data, int(50e-3*fs)+1, 3, deriv=2, delta=1./fs)

        self.q0 = savgol_filter(self.q_data, int(50e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.q1 = savgol_filter(self.q_data, int(50e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.q2 = savgol_filter(self.q_data, int(50e-3*fs)+1, 3, deriv=2, delta=1./fs)


    def get_slice_bools(self):
        # Record list of bools which will break the data into chunks of the desired duration
        # These define a "slice"
        tmin = np.amin(self.t_data)
        tmax = np.amax(self.t_data)
        window_edges = np.arange(tmin, tmax, self.slice_len)

        slice_bool_list = list()
        slice_live_list = list()
        slice_time_list = list()
        for i in range(len(window_edges)-1):
            
            # This defines the slice 
            slice_bool = (self.t_data>=window_edges[i])&(self.t_data<window_edges[i+1])
            slice_bool_list.append( slice_bool )
            
            # Count how many samples (rather livetime) are actually in the slice, for later QC
            slice_live_list.append( np.sum(slice_bool)/self.fs )
            
            # Also save when the slice starts. Helpful for "macro" plotting
            slice_time_list.append( np.amin(self.t_data[slice_bool]) )

        self.slice_bools = np.array(slice_bool_list)
        self.slice_lives = np.array(slice_live_list) 
        self.slice_times = np.array(slice_time_list)


    def offset_proc(self):
        # Go through each slice and get heuristic offsets
        offset_grid = np.zeros(len(self.slice_bools))
        for i, slice_bool in enumerate(self.slice_bools):
            offset_grid[i] = utils.get_offset_dA( self.p0, self.q0, slice_bool )
        self.slice_offsets = np.array(offset_grid)

    def coher_proc(self): 
        # Go through each slice and get some "macro" properties: coherence, MAP, mean flow (MAQ) and HR
        coher_grid = np.zeros(len(self.slice_bools))
        MAP_grid = np.zeros(len(self.slice_bools))
        MAQ_grid = np.zeros(len(self.slice_bools))
        f0_grid = np.zeros(len(self.slice_bools))

        # For welch's algorithm, let the sub-segment be half the length of the slice
        N = int(0.5*self.slice_len*self.fs)
        for i, slice_bool in enumerate(self.slice_bools):
            frq1, coher = coherence( self.p0[slice_bool], self.q0[slice_bool], fs=self.fs, nperseg=N )
            frq2, Ppp = welch( self.p0[slice_bool], fs=self.fs, nperseg=N )
            
            # Only consider the spectrum<25 Hz. Arbirtary but reasonable
            frq_range = frq2<25
            
            # Store averages
            MAP_grid[i] = np.mean(self.p0[slice_bool])
            MAQ_grid[i] = np.mean(self.q0[slice_bool])
            
            # Evaluate coherence and estimate HR at the peak of the pressure spectrum
            coher_grid[i] = coher[frq_range][ np.argmax(Ppp[frq_range]) ]
            f0_grid[i] = frq2[frq_range][ np.argmax(Ppp[frq_range]) ]
        
        self.slice_coher = np.array(coher_grid)
        self.slice_MAP = np.array(MAP_grid)
        self.slice_MAQ = np.array(MAQ_grid)
        self.slice_f0 = np.array(f0_grid)

    def get_onsets(self):
       
        # Initialize record for the onset of each pulse - defined as start of "arterial" systole (opening of aortic valve)
        all_ton = list()

        dist = 0.2 # seconds - make sure infered HR<300

        for i, slice_bool in enumerate(self.slice_bools):
            # Get the slice we need

            # Need times to evaluate
            t_slice = self.t_data[slice_bool]

            # We'll distinguish individual pulses by peaks in the (normalized) 1st derivative of the pressure slice
            p1_slice = self.p1[slice_bool]
            p1_norm = utils.norm_trace(p1_slice)

            peaks, _ = find_peaks( 
                                  p1_norm,
                                  prominence=np.percentile(p1_norm, 98),
                                  distance=int(dist*self.fs)      
                                 )

            for j, peak in enumerate(peaks):
                flat_bool = np.abs(p1_norm)<0.05*p1_norm[peak]
                lookback_bool = (t_slice>t_slice[peak]-dist)&(t_slice<t_slice[peak])
                if np.sum(flat_bool&lookback_bool)==0:
                    continue
                else:
                    all_ton.append( t_slice[flat_bool&lookback_bool][-1] )


        self.pulse_onsets = np.array(all_ton)

    def get_pulses(self):
        pulse_list = list()
        for i, slice_bool in enumerate(self.slice_bools):
            offset = self.slice_offsets[i]

            q0_shift = np.roll( self.q0, offset )
            q1_shift = np.roll( self.q1, offset )
            q2_shift = np.roll( self.q2, offset ) 

            onsets_inrange = self.pulse_onsets[(self.pulse_onsets>=np.amin(self.t_data[slice_bool]))&(self.pulse_onsets<np.amax(self.t_data[slice_bool]))]

            for j in range( len(onsets_inrange)-1 ):
                pulse_range = (self.t_data>=onsets_inrange[j])&(self.t_data<onsets_inrange[j+1])

                this_pulse = Pulse(
                                self.t_data[pulse_range],
                                self.p0[pulse_range],
                                self.p1[pulse_range],
                                self.p2[pulse_range],
                                q0_shift[pulse_range], 
                                q1_shift[pulse_range], 
                                q2_shift[pulse_range] 
                        )
                pulse_list.append(this_pulse)

        self.pulses = pulse_list


