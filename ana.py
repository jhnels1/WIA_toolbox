import numpy as np
import pandas as pd

import base

def get_RQs( t_data, p_data, q_data ):
    
    # Inititalize the experiment
    my_experiment = base.Experiment( t_data, p_data, q_data )

    # A few pre-processing steps...
    # 1) Define slices
    my_experiment.get_slice_bools()

    # 2) Calculate offset between pressure and flow slices
    #my_experiment.offset_proc()

    # 3) Calculate coherence between pressure and flow
    my_experiment.coher_proc()

    # 4) Define the onset of each individual pulse
    my_experiment.get_onsets()

    # 5) Define the actual content of each pulse, along with RQs
    my_experiment.get_pulses()

    # Now we're ready to go through all the pulses and collect RQs in a DataFrame

    pulse_list = list()
    for pulse in my_experiment.pulses:
        RQ_entry = {
                    "T0":pulse.t0,
                    "MAP":pulse.MAP,
                    "MAQ":pulse.MAQ,
                    "PP":pulse.PP,
                    "PQ":pulse.PQ,
                    "RI":pulse.RI,
                    "PI":pulse.PI,
                    "RR":pulse.RR,
                    "t_SBP":pulse.t_SBP,
                    "t_DN":pulse.t_DN,
                    "t_WImax":pulse.t_WImax,
                    "t_postWImax":pulse.t_postWIpeak,
                    "WI_factor":pulse.WI_factor,
                    "WI_peak_pre":pulse.WI_peak_pre,
                    "WI_mean_pre":pulse.WI_mean_pre,
                    "WI_peak_mid":pulse.WI_peak_mid,
                    "WI_mean_mid":pulse.WI_mean_mid,
                    "WI_peak_post":pulse.WI_peak_post,
                    "WI_mean_post":pulse.WI_mean_post,
                    "WI_mean_all":pulse.WI_mean_all,
                    "WI_mean_fore":pulse.WI_mean_fore,
                    "WI_mean_back":pulse.WI_mean_back,
                    "coher":pulse.coher,
                    "t_off":pulse.t_off,
                    "block_number":pulse.block_number
                }
        pulse_list.append(RQ_entry)
    return pd.DataFrame(pulse_list), my_experiment.pulses
