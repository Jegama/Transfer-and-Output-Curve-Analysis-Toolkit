import numpy as np
import pandas as pd
import seaborn as sns
import os, re
from scipy import stats
from fet_toolkit import tc_analysis
import fet_plotter

def main():
    
    print('\n\nLet\'s begin the plotting\n\n')

    fet_plotter.plot_mobility_plus_reliability()
    fet_plotter.plot_max_hysteresis()
    fet_plotter.general_plotting('ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$')
    fet_plotter.general_plotting('Intercepts', r'Threshold voltage, $V_T$ (V)')
    fet_plotter.general_plotting('Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)')
    fet_plotter.general_plotting('Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)')
    fet_plotter.general_plotting('MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$')
    fet_plotter.general_plotting('Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)')

    print('\n\tDone\n')
    
if __name__ == '__main__':
    try:
        exit(main())
    except KeyboardInterrupt:
        print('\n\nExiting due to KeyboardInterrupt!\n')