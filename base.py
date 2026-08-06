import numpy as np
import pandas as pd

import utils

from scipy.signal import welch, savgol_filter, find_peaks, csd, coherence, correlate, correlation_lags

class Pulse:
    def __init__(self, t, p0, p1, p2, q0, q1, q2, fs=1000):
        self.t=t
        self.p0=p0
        self.p1=p1
        self.p2=p2
        self.q0=q0
        self.q1=q1
        self.q2=q2
        self.WI = p1*q1
        self.fs=fs

        self.MAP = np.mean(p0)
        self.PP = np.amax(p0)-p0[0]
        self.MAQ = np.mean(q0)
        self.PQ = np.amax(q0)-q0[0]
        self.RI = self.PQ/np.amax(q0)
        self.PI = (np.amax(q0)-np.amin(q0))/np.mean(q0)
        self.RR = np.amax(t)-np.amin(t)
        self.t0 = np.amin(t)

    def segment_pulse(self):
        self.t_SBP = self.t[np.argmax(self.p0)]
        DN_range = (self.t>self.t_SBP)&(self.t<0.95*self.RR+self.t0)
        if np.sum(DN_range)==0:
            self.t_DN = np.nan
        else:
            self.t_DN = self.t[DN_range][np.argmax(self.p2[DN_range])]

        self.t_WImax = self.t[np.argmax(self.WI)]
        self.p0_WImax = self.p0[np.argmax(self.WI)]
        self.p1_WImax = self.p1[np.argmax(self.WI)]
        self.q0_WImax = self.q0[np.argmax(self.WI)]
        self.q1_WImax = self.q1[np.argmax(self.WI)]

        # Get minimum after WI peak
        post_WImax = self.t>self.t_WImax
        WI_grad = np.gradient( self.WI )
        WI_flat_bool = np.abs( WI_grad )<0.05*np.amax(WI_grad)
        if np.sum(post_WImax&WI_flat_bool)==0:
            self.t_postWIpeak = np.nan
        else:
            self.t_postWIpeak = self.t[ post_WImax&WI_flat_bool ][0]

        #self.teff_mid = 0.5*( self.t_WImax + self.t_DN )

        #self.WI_factor = self.p1_WImax/self.q1_WImax
        self.WI_fore = (1./4/self.WI_factor)*( self.p1 + self.WI_factor*self.q1 )**2
        self.WI_back = -(1./4/self.WI_factor)*( self.p1 - self.WI_factor*self.q1 )**2

        pre_range = self.t<self.t_postWIpeak
        mid_range = (self.t>=self.t_postWIpeak)&(self.t<self.t_DN)
        post_range = self.t>=self.t_DN

        if np.sum(pre_range)==0:
            self.WI_peak_pre = np.nan
            self.WI_mean_pre = np.nan
        else:
            self.WI_peak_pre = np.amax( self.WI[pre_range] )
            self.WI_mean_pre = np.mean( self.WI[pre_range] )
        
        if np.sum(mid_range)==0:
            self.WI_peak_mid = np.nan
            self.WI_mean_mid = np.nan
        else:
            self.WI_peak_mid = np.amax( self.WI[mid_range] )
            self.WI_mean_mid = np.mean( self.WI[mid_range] )
        
        if np.sum(post_range)==0:
            self.WI_peak_post = np.nan
            self.WI_mean_post = np.nan
        else:
            self.WI_peak_post = np.amax( self.WI[post_range] )
            self.WI_mean_post = np.mean( self.WI[post_range] )

        self.WI_mean_all = np.mean(self.WI)
        self.WI_mean_fore = np.mean(self.WI_fore)
        self.WI_mean_back = np.mean(self.WI_back)

        #self.dP_fore = 0.5*(self.p1+self.WI_factor*self.q1)
        #self.dP_back = 0.5*(self.p1-self.WI_factor*self.q1)

        #self.P_fore = self.p0[0] + np.array([ np.trapz( self.dP_fore[:i], self.t[:i] ) for i in range(len(self.t)) ])
        #self.P_back = self.p0[0] + np.array([ np.trapz( self.dP_back[:i], self.t[:i] ) for i in range(len(self.t)) ])

        #self.is_rise = self.t<=self.t_SBP
        #self.is_mid = (self.t>self.t_SBP)&(self.t<=self.t_DN)
        #self.is_fall = self.t>self.t_DN

class Experiment:

    def __init__(self, t_data, p_data, q_data, fs=1000, slice_len=20):

        # Store the data
        self.t_data = np.array(t_data)
        self.p_data = np.array(p_data)
        self.q_data = np.array(q_data)
        self.fs = fs
        self.slice_len = slice_len

        # also store filters+derivatives of pressure and flow
        self.p0 = savgol_filter(self.p_data, int(70e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.p1 = savgol_filter(self.p_data, int(70e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.p2 = savgol_filter(self.p_data, int(70e-3*fs)+1, 3, deriv=2, delta=1./fs)

        self.q0 = savgol_filter(self.q_data, int(70e-3*fs)+1, 3, deriv=0, delta=1./fs)
        self.q1 = savgol_filter(self.q_data, int(70e-3*fs)+1, 3, deriv=1, delta=1./fs)
        self.q2 = savgol_filter(self.q_data, int(70e-3*fs)+1, 3, deriv=2, delta=1./fs)


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
            
            # Dont save the slice if there's nothing there
            if np.sum(slice_bool)==0:
                continue
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
            coher = self.slice_coher[i]

            onsets_inrange = self.pulse_onsets[(self.pulse_onsets>=np.amin(self.t_data[slice_bool]))&(self.pulse_onsets<np.amax(self.t_data[slice_bool]))]

            for j in range( len(onsets_inrange)-1 ):
                # Defines pressure pulse
                pulse_range = (self.t_data>=onsets_inrange[j])&(self.t_data<onsets_inrange[j+1])
                
                # Extend the initial range to search for optimal offset between pressure and flow
                extend_delta = 0.1 # seconds
                extend_samps = int(extend_delta*self.fs)
                
                extend_range = (self.t_data>=onsets_inrange[j]-extend_delta)&(self.t_data<onsets_inrange[j+1]+extend_delta)
                
                # Get coarse offset by maximizing WI
                cor_grid = correlate( self.p1[pulse_range], self.q1[extend_range], mode='valid' )
                lag_grid = correlation_lags( np.sum(pulse_range), np.sum(extend_range), mode='valid' )
                
                coarse_off = lag_grid[np.argmax(cor_grid)]+extend_samps # Note added term. Lags are weird for unequal length arrays

                # Get fine offset by making systolic onset 'most linear' - equivalent to minimizing reflected wave
                fine_grid = np.arange(-10, 10) # +/- 10 ms

                t_max = self.t_data[pulse_range][ np.argmax( np.roll( self.q0, coarse_off )[pulse_range]) ]
                t_min = np.amin(self.t_data[pulse_range])
                delta = t_max-t_min
                lin_start = t_min + 0.1*delta
                lin_stop = t_min + 0.7*delta

                lin_range = (self.t_data>=lin_start)&(self.t_data<=lin_stop)

                if np.sum(lin_range)==0:
                    continue

                dA_grid = np.zeros(len(fine_grid))
                m_grid = np.zeros(len(fine_grid))
                for k, some_off in enumerate(fine_grid):
                    q_grid = np.roll( self.q0, coarse_off+some_off )[lin_range]
                    q0 = np.amin(q_grid)
                    q1 = np.amax(q_grid)
                    p0 = np.amin(self.p0[lin_range])
                    p1 = np.amax(self.p0[lin_range])

                    m = (p1-p0)/(q1-q0)

                    p_lin = m*(q_grid-q0)+p0
                    dA_grid[k] = np.trapz( np.abs( self.p0[lin_range]-p_lin ), q_grid )
                    m_grid[k] = m
                fine_off = fine_grid[np.argmin(dA_grid)]
                WI_factor = m_grid[np.argmin(dA_grid)]
                offset = coarse_off + fine_off
                #offset = utils.get_offset_dA( self.p0, self.q0, pulse_range )
                                
                q0_shift = np.roll( self.q0, offset )
                q1_shift = np.roll( self.q1, offset )
                q2_shift = np.roll( self.q2, offset ) 
                
                this_pulse = Pulse(
                                self.t_data[pulse_range],
                                self.p0[pulse_range],
                                self.p1[pulse_range],
                                self.p2[pulse_range],
                                q0_shift[pulse_range], 
                                q1_shift[pulse_range], 
                                q2_shift[pulse_range] 
                        )
                this_pulse.segment_pulse()

                # Maybe playing with fire here, but we ball
                this_pulse.WI_factor = WI_factor
                this_pulse.t_off = offset/self.fs
                this_pulse.coher = coher
                this_pulse.block_number = i

                pulse_list.append(this_pulse)

        self.pulses = pulse_list


