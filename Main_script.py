import numpy as np
import pandas as pd
import seaborn as sns
import argparse, re, os
from scipy import stats
from fet_toolkit import tc_analysis
from oc_toolkit import oc_analysis
import fet_plotter

def check_type(value):
    if value not in ['single', 'multi']:
        raise argparse.ArgumentTypeError("\n%s is not implemented yet, please use one of the following\n\tsingle\n\tmulti" % value)
    else:
        return value

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("data_folder", help="Folder with the data to analyze", type=str)
    parser.add_argument("analysis_type", help="Choose between single or multifolder analysis", type=check_type)

    args = parser.parse_args()

    assert os.path.exists(args.data_folder), '%s does not exists' % args.data_folder

    if not os.path.exists('{}Results'.format(args.data_folder)):
        os.makedirs('{}Results'.format(args.data_folder))

    print('\n\tLet\'s begin\n\nDoing a {} analysis on {}'.format(args.analysis_type, args.data_folder))

    sub_folders = ['Char_curves', 'Hysteresis', 'Mobility_curves', 'SVG_plots', 'SVG_plots/Char_curves']

    for i in sub_folders:
        if not os.path.exists('{}Results/{}'.format(args.data_folder, i)):
            os.makedirs('{}Results/{}'.format(args.data_folder, i))

    # Get the list of files to process
    if args.analysis_type == 'single':
        master_file_list = os.listdir(args.data_folder)
        print('\nTotal number of files:', len(master_file_list))
    elif args.analysis_type == 'multi':
        master_file_list = []
        for i in os.listdir(args.data_folder):
            if os.path.isdir('{}{}'.format(args.data_folder, i)) and i != 'Results':
                master_file_list.extend(['{}/{}'.format(i, a) for a in os.listdir('{}/{}'.format(args.data_folder, i))])
        print('\nTotal number of files:', len(master_file_list))

    tc_analyzer = tc_analysis()
    oc_analyzer = oc_analysis()

    # Set color values for plots
    sns.set_style('whitegrid')
    sns.set_context('talk')

    print('\nValidating files')

    TC_master = tc_analyzer.generate_empty_dataframes(master_file_list)
    OC_master = oc_analyzer.generate_empty_dataframes(master_file_list)

    for filename in master_file_list:
        if filename.endswith('OC.csv'):
            print('\nWorking on:', filename[:-4])
            try:
                df = pd.read_csv(os.path.join(args.data_folder, filename))

                unique_VG = df.VG.unique()  # list unique values in VG

                df0 = df.loc[df['VG'] == 0]  # pull out VD from first data set

                # make two data frames with x = VD. Otherwise, they are empty
                df_ID = pd.DataFrame(index=df0['VD'])
                df_IG = pd.DataFrame(index=df0['VD'])

                # Preparing Data

                for n in unique_VG:
                    ID = df.loc[df['VG'] == n]['ID']
                    IG = df.loc[df['VG'] == n]['IG']
                    # waf1 4A3 2mm L20W2B_weird_last_Vg F4TCNQ_0.1 OC.csv breaks the script.
                    df_ID[n] = list(ID)
                    df_IG[n] = list(IG)

                # Plot ID

                ax = df_ID.plot(figsize=(14, 10), colormap='winter_r')

                ax.legend(loc=(1, -0.01))

                ax.set_ylabel('ID (A)')
                ax.set_xlabel('VD (V)')

                ax.invert_yaxis()
                ax.invert_xaxis()

                fig = ax.get_figure()
                fig.savefig(os.path.join(
                    args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_ID.png'))
                fig.savefig(os.path.join(
                    args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_ID.svg'))

                # Plot IG

                pltIG = df_IG.plot(figsize=(14, 10), colormap='winter_r')
                pltIG.set_ylabel('IG (A)')
                pltIG.set_xlabel('VD (V)')

                pltIG.invert_yaxis()
                pltIG.invert_xaxis()

                fig2 = pltIG.get_figure()

                fig2.savefig(os.path.join(
                    args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_IG.png'))
                fig2.savefig(os.path.join(
                    args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_IG.svg'))

                # New analysis

                device_name, file_voltage, file_condition = oc_analyzer.current_row(
                    master_file_list, filename)

                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    df_ID.reset_index().loc[:4, 0].tolist(), df_ID.index[:5].tolist())

                resistance = 1/slope

                # OC_master = oc_analyzer.populate_dataframes(OC_master, filename, resistance)

                device_length = float(
                    re.sub('\D', '', re.findall('L\d+W', filename)[0]))

                OC_master.loc[device_name, file_voltage,
                              file_condition]['Resistance'] = resistance

                OC_master.loc[device_name, file_voltage,
                              file_condition]['Length'] = device_length

            except ValueError as e:
                print(e)
                print('\n\tError on:', filename[:-4])

        elif filename.endswith('TC.csv'):
            print('\nWorking on:', filename[:-4])

            tc_analyzer.get_l_and_w(filename)

            main0, VD, df_TC, df_TC_sqrt = tc_analyzer.separate_data(args.data_folder, filename)

            tc_analyzer.current_row(master_file_list, filename)

            # VGs for slope
            main_VG = list(main0['VG'])
            middle_point = int(len(main_VG) / 2)
            fwd_VG = main_VG[middle_point-5:middle_point]
            rev_VG = main_VG[middle_point:middle_point+5]

            # Vd identifiers for slopes
            Vd1 = sorted(list(set(VD)))[-2]
            Vd2 = sorted(list(set(VD)))[-3]
            VdMinus2 = sorted(list(set(VD)))[1]
            VdMinus1 = sorted(list(set(VD)))[0]

            # Get VDs for slopes
            fwd_id1 = list(df_TC[Vd1][middle_point-5:middle_point])
            rev_id1 = list(df_TC[Vd1][middle_point:middle_point+5])

            fwd_id2 = list(df_TC[Vd2][middle_point-5:middle_point])
            rev_id2 = list(df_TC[Vd2][middle_point:middle_point+5])

            fwd_idminus2 = list(
                df_TC_sqrt[VdMinus2][middle_point-5:middle_point])
            rev_idminus2 = list(
                df_TC_sqrt[VdMinus2][middle_point:middle_point+5])

            fwd_idminus1 = list(
                df_TC_sqrt[VdMinus1][middle_point-5:middle_point])
            rev_idminus1 = list(
                df_TC_sqrt[VdMinus1][middle_point:middle_point+5])

            # Save slopes and X intercept to master DataFrame

            # ['Linear (ID)', 'Vd1', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_id1)
            intercept_line_column = '{}_FWD'.format(Vd1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, True, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC, middle_point-1, 0, -2, ['Linear (ID)', 'Vd1', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC[Vd1][:middle_point]), 'FWD', vd_=Vd1)
            tc_analyzer.populate_dataframes(
                ['Linear (ID)', 'Vd1', 'FWD'], slope, intercept, vd_=Vd1)

            # ['Linear (ID)', 'Vd1', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_id1)
            intercept_line_column = '{}_REV'.format(Vd1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, True, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC, middle_point, -1, -2, ['Linear (ID)', 'Vd1', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC[Vd1][:middle_point]), 'REV', vd_=Vd1)
            tc_analyzer.populate_dataframes(
                ['Linear (ID)', 'Vd1', 'REV'], slope, intercept, vd_=Vd1)

            # ['Linear (ID)', 'Vd2', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_id2)
            intercept_line_column = '{}_FWD'.format(Vd2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, True, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC, middle_point-1, 0, -3, ['Linear (ID)', 'Vd2', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC[Vd2][:middle_point]), 'FWD', vd_=Vd2)
            tc_analyzer.populate_dataframes(
                ['Linear (ID)', 'Vd2', 'FWD'], slope, intercept, vd_=Vd2)

            # ['Linear (ID)', 'Vd2', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_id2)
            intercept_line_column = '{}_REV'.format(Vd2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, True, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC, middle_point, -1, -3, ['Linear (ID)', 'Vd2', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC[Vd2][middle_point:]), 'REV', vd_=Vd2)
            tc_analyzer.populate_dataframes(
                ['Linear (ID)', 'Vd2', 'REV'], slope, intercept, vd_=Vd2)

            # ['Saturation (ID)', 'Vd(-2)', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_idminus2)
            intercept_line_column = '{}_FWD'.format(VdMinus2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, False, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point-1, 0, 1, ['Saturation (ID)', 'Vd(-2)', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC_sqrt[VdMinus2][:middle_point]), 'FWD', vd_=VdMinus2, linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-2)', 'FWD'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-2)', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_idminus2)
            intercept_line_column = '{}_REV'.format(VdMinus2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, False, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point, -1, 1, ['Saturation (ID)', 'Vd(-2)', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC_sqrt[VdMinus2][middle_point:]), 'REV', vd_=VdMinus2, linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-2)', 'REV'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-1)', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_idminus1)
            intercept_line_column = '{}_FWD'.format(VdMinus1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, False, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point-1, 0, 0, ['Saturation (ID)', 'Vd(-1)', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC_sqrt[VdMinus1][:middle_point]), 'FWD', vd_=VdMinus1, linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-1)', 'FWD'], slope, intercept, linear=False)

           # ['Saturation (ID)', 'Vd(-1)', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_idminus1)
            intercept_line_column = '{}_REV'.format(VdMinus1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, False, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point, -1, 0, ['Saturation (ID)', 'Vd(-1)', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC_sqrt[VdMinus1][middle_point:]), 'REV', vd_=VdMinus1, linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-1)', 'REV'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-2)', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_idminus2)
            intercept_line_column = '{}_FWD'.format(VdMinus2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, False, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point-1, 0, 1, ['Saturation (ID)', 'Vd(-2)', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC_sqrt[VdMinus2][:middle_point]), 'FWD', linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-2)', 'FWD'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-2)', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_idminus2)
            intercept_line_column = '{}_REV'.format(VdMinus2)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, False, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point, -1, 1, ['Saturation (ID)', 'Vd(-2)', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC_sqrt[VdMinus2][middle_point:]), 'REV', linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-2)', 'REV'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-1)', 'FWD']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                fwd_VG, fwd_idminus1)
            intercept_line_column = '{}_FWD'.format(VdMinus1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[:middle_point], slope, intercept, False, 'FWD')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point-1, 0, 0, ['Saturation (ID)', 'Vd(-1)', 'FWD'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[:middle_point], list(
                df_TC_sqrt[VdMinus1][:middle_point]), 'FWD', linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-1)', 'FWD'], slope, intercept, linear=False)

            # ['Saturation (ID)', 'Vd(-1)', 'REV']
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                rev_VG, rev_idminus1)
            intercept_line_column = '{}_REV'.format(VdMinus1)
            tc_analyzer.update_lines(intercept_line_column, middle_point,
                                     main_VG[middle_point:], slope, intercept, False, 'REV')
            tc_analyzer.populate_min_max_ratio_files(
                df_TC_sqrt, middle_point, -1, 0, ['Saturation (ID)', 'Vd(-1)', 'REV'], middle_point)
            tc_analyzer.generate_mobility_curves(args.data_folder, filename, main_VG[middle_point:], list(
                df_TC_sqrt[VdMinus1][middle_point:]), 'REV', linear=False)
            tc_analyzer.populate_dataframes(
                ['Saturation (ID)', 'Vd(-1)', 'REV'], slope, intercept, linear=False)

            tc_analyzer.generate_hysteresis(args.data_folder, filename, middle_point)

            df_TC_lines, df_TC_sqrt_lines = tc_analyzer.remove_zeros_from_lines()

            # Generate plot for linear ID
            pltTC = df_TC.plot(figsize=(14, 10), colormap='inferno')
            pltTC.set_ylabel('ID (A)')
            pltTC.set_xlabel('VG (V)')

            # Generate plot for saturation/sqrt ID
            pltTC_sqrt = df_TC_sqrt.plot(figsize=(14, 10), colormap='inferno')
            pltTC_sqrt.set_ylabel('√ID (√A)')
            pltTC_sqrt.set_xlabel('VG (V)')

            # Generate plot for linear ID with line intercept
            pltTC_intercept = df_TC_lines.plot(
                figsize=(14, 10), colormap='inferno')
            pltTC_intercept.set_ylabel('ID (A)')
            pltTC_intercept.set_xlabel('VG (V)')

            # Generate plot for saturation/sqrt ID with line intercept
            pltTC_sqrt_intercept = df_TC_sqrt_lines.plot(
                figsize=(14, 10), colormap='inferno')
            pltTC_sqrt_intercept.set_ylabel('√ID (√A)')
            pltTC_sqrt_intercept.set_xlabel('VG (V)')

            # Invert plot to first quadrant 1
            pltTC.invert_yaxis()
            pltTC.invert_xaxis()
            pltTC_sqrt.invert_yaxis()
            pltTC_sqrt.invert_xaxis()
            pltTC_intercept.invert_yaxis()
            pltTC_intercept.invert_xaxis()
            pltTC_sqrt_intercept.invert_yaxis()
            pltTC_sqrt_intercept.invert_xaxis()

            # Save plots
            linear_tc = pltTC.get_figure()
            linear_tc.savefig(os.path.join(
                args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_lin_ID.png'))
            linear_tc.savefig(os.path.join(
                args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_lin_ID.svg'))

            sat_tc = pltTC_sqrt.get_figure()
            sat_tc.savefig(os.path.join(
                args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_sat_ID.png'))
            sat_tc.savefig(os.path.join(
                args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_sat_ID.svg'))

            linear_tc_intercept = pltTC_intercept.get_figure()
            linear_tc_intercept.savefig(os.path.join(
                args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_lin_ID_intercept.png'))
            linear_tc_intercept.savefig(os.path.join(
                args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_lin_ID_intercept.svg'))

            sat_tc_intercept = pltTC_sqrt_intercept.get_figure()
            sat_tc_intercept.savefig(os.path.join(
                args.data_folder, 'Results/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_sat_ID_intercept.png'))
            sat_tc_intercept.savefig(os.path.join(
                args.data_folder, 'Results/SVG_plots/Char_curves', re.sub(r'/', '_', filename[:-4]) + '_sat_ID_intercept.svg'))

    # Save TC_slopes
    tc_analyzer.export_dataframes(args.data_folder)

    print('\n\nLet\'s begin the plotting\n\n')

    if (len(TC_master.index.get_level_values(2).unique()) == 1) and ('SOLO' in TC_master.index.get_level_values(2).unique().tolist()):
        print('\n\tDone\n')
    else:
        if (any([len(re.findall("\d", x)) > 0 for x in TC_master.index.get_level_values(1).unique().tolist()])) and ('AFTER' in TC_master.index.get_level_values(2).unique().tolist()):
            print('\n\nPlotting voltage')
            fet_plotter.plot_mobility_plus_reliability(args.data_folder, 'voltage')
            fet_plotter.plot_max_hysteresis(args.data_folder, 'voltage')
            fet_plotter.general_plotting(args.data_folder, 
                'ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$', 'voltage', r'$\Delta\ I_D\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Intercepts', r'Threshold voltage, $V_T$ (V)', 'voltage', r'$\Delta\ Threshold\ voltage,\ V_T$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)', 'voltage', r'$\Delta\ Max\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)', 'voltage', r'$\Delta\ Min\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$', 'voltage', r'$\Delta\ Effective\ \mu\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)', 'voltage', r'$\Delta\ I_{ON}/I_{OFF}$ ratio (%)')
        if 'BEFORE_ONLY' in TC_master.index.get_level_values(2).unique().tolist() or 'AFTER_ONLY' in TC_master.index.get_level_values(2).unique().tolist():
            print('\n\nPlotting ONLY')
            fet_plotter.plot_mobility_plus_reliability(args.data_folder, 'only')
            fet_plotter.plot_max_hysteresis(args.data_folder, 'only')
            fet_plotter.general_plotting(args.data_folder, 
                'ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$', 'only', r'$\Delta\ I_D\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Intercepts', r'Threshold voltage, $V_T$ (V)', 'only', r'$\Delta\ Threshold\ voltage,\ V_T$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)', 'only', r'$\Delta\ Max\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)', 'only', r'$\Delta\ Min\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$', 'only', r'$\Delta\ Effective\ \mu\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)', 'only', r'$\Delta\ I_{ON}/I_{OFF}$ ratio (%)')
        if 'CYCLE' in TC_master.index.get_level_values(2).unique().tolist():
            print('\n\nPlotting CYCLE')
            fet_plotter.plot_mobility_plus_reliability(args.data_folder, 'cycle')
            fet_plotter.plot_max_hysteresis(args.data_folder, 'cycle')
            fet_plotter.general_plotting(args.data_folder, 
                'ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$', 'cycle', r'$\Delta\ I_D\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Intercepts', r'Threshold voltage, $V_T$ (V)', 'cycle', r'$\Delta\ Threshold\ voltage,\ V_T$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)', 'cycle', r'$\Delta\ Max\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)', 'cycle', r'$\Delta\ Min\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$', 'cycle', r'$\Delta\ Effective\ \mu\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)', 'cycle', r'$\Delta\ I_{ON}/I_{OFF}$ ratio (%)')
        if 'CONC' in TC_master.index.get_level_values(2).unique().tolist():
            print('\n\nPlotting CONC')
            fet_plotter.plot_mobility_plus_reliability(args.data_folder, 'conc')
            fet_plotter.plot_max_hysteresis(args.data_folder, 'conc')
            fet_plotter.general_plotting(args.data_folder, 
                'ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$', 'conc', r'$\Delta\ I_D\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Intercepts', r'Threshold voltage, $V_T$ (V)', 'conc', r'$\Delta\ Threshold\ voltage,\ V_T$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)', 'conc', r'$\Delta\ Max\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)', 'conc', r'$\Delta\ Min\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$', 'conc', r'$\Delta\ Effective\ \mu\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)', 'conc', r'$\Delta\ I_{ON}/I_{OFF}$ ratio (%)')
        if 'post-ACN' in TC_master.index.get_level_values(1).unique().tolist():
            print('\n\nPlotting post-ACN')
            fet_plotter.plot_mobility_plus_reliability(args.data_folder, 'post-ACN')
            fet_plotter.plot_max_hysteresis(args.data_folder, 'post-ACN')
            fet_plotter.general_plotting(args.data_folder, 
                'ID_at_VG_0', r'$I_D\ x\ 10^{-7} (A)$', 'post-ACN', r'$\Delta\ I_D\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Intercepts', r'Threshold voltage, $V_T$ (V)', 'post-ACN', r'$\Delta\ Threshold\ voltage,\ V_T$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Max_ID', r'Max $I_D\ x\ 10^{-7}$ (A)', 'post-ACN', r'$\Delta\ Max\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'Min_ID', r'Min $I_D\ x\ 10^{-7}$ (A)', 'post-ACN', r'$\Delta\ Min\ I_D$ (%)')
            fet_plotter.general_plotting(args.data_folder, 
                'MU_eff', r'Effective $\mu\ x\ 10^{-4} (cm^2/Vs)$', 'post-ACN', r'$\Delta\ Effective\ \mu\ (\%)$')
            fet_plotter.general_plotting(args.data_folder, 
                'Ratio_ID', r'$I_{ON}/I_{OFF}$ ratio (A)', 'post-ACN', r'$\Delta\ I_{ON}/I_{OFF}$ ratio (%)')

    print('\nGenerating AVG DF')

    tc_analyzer.generate_avg_df(args.data_folder, 'Mobility')
    tc_analyzer.generate_avg_df(args.data_folder, 'Max_Hysteresis')
    tc_analyzer.generate_avg_df(args.data_folder, 'ID_at_VG_0')
    tc_analyzer.generate_avg_df(args.data_folder, 'Intercepts')
    tc_analyzer.generate_avg_df(args.data_folder, 'Max_ID')
    tc_analyzer.generate_avg_df(args.data_folder, 'Min_ID')
    tc_analyzer.generate_avg_df(args.data_folder, 'MU_eff')
    tc_analyzer.generate_avg_df(args.data_folder, 'Ratio_ID')
    
    OC_master.to_csv('%s/Results/OC_master.csv' % args.data_folder)

    print('\n\nStarting OC Analysis\n\n')

    # OC_master = pd.read_csv('%s/Results/OC_master.csv' % args.data_folder, index_col=[0,1,2])

    temp = OC_master.index.get_level_values(0)
    new = []
    for i in temp:
        new.append(re.findall('\S+\s\S+\s\S+', i)[0])

    new_index = [new, OC_master.index.get_level_values(
        1), OC_master.index.get_level_values(2)]
    tuples = list(zip(*new_index))
    new_index_temp = pd.MultiIndex.from_tuples(tuples) # names=['Device', 'Voltage', 'Condition']

    oc_master_new = pd.DataFrame(OC_master.values, index=new_index_temp, columns=[
                                 'Length', 'Resistance']).sort_index(level=0)

    oc_master_RcontactFit = {'device_name': [], 'device_voltage': [], 'device_condition': [], 'intercept': [], 'r_value': [], 'std_err': []}

    print('\n\nPlotting voltage')
    oc_master_RcontactFit = oc_analyzer.generate_r2_lines(
        args.data_folder, oc_master_new, 'voltage', oc_master_RcontactFit)

    print('\n\nPlotting ONLY')
    oc_master_RcontactFit = oc_analyzer.generate_r2_lines(
        args.data_folder, oc_master_new, 'only', oc_master_RcontactFit)

    print('\n\nPlotting CYCLE')
    oc_master_RcontactFit = oc_analyzer.generate_r2_lines(
        args.data_folder, oc_master_new, 'cycle', oc_master_RcontactFit)

    print('\n\nPlotting CONC')
    oc_master_RcontactFit = oc_analyzer.generate_r2_lines(
        args.data_folder, oc_master_new, 'conc', oc_master_RcontactFit)

    print('\n\nPlotting post-ACN')
    oc_master_RcontactFit = oc_analyzer.generate_r2_lines(
        args.data_folder, oc_master_new, 'post-ACN', oc_master_RcontactFit)

    # Generate oc_master_RcontactFit CSV

    column_names = ['R_Contact', 'R2', 'R_Contact_ERROR']

    row_index = [oc_master_RcontactFit['device_name'],
                 oc_master_RcontactFit['device_voltage'], oc_master_RcontactFit['device_condition']]

    tuples = list(zip(*row_index))

    index = pd.MultiIndex.from_tuples(tuples) # names=['Device', 'Voltage', 'Condition']
    OC_master_RcontactFit_DF = pd.DataFrame(
        np.zeros((len(tuples), 3)), index=index, columns=column_names)

    OC_master_RcontactFit_DF['R_Contact'] = oc_master_RcontactFit['intercept']
    OC_master_RcontactFit_DF['R2'] = oc_master_RcontactFit['r_value']
    OC_master_RcontactFit_DF['R_Contact_ERROR'] = oc_master_RcontactFit['std_err']

    OC_master_RcontactFit_DF.sort_index(level=0).to_csv(
        '%s/Results/OC_master_RcontactFit.csv' % args.data_folder)

    print('\nPlotting OC master\n')

    oc_analysis.plot_oc_master(OC_master_RcontactFit_DF, args.data_folder, 'voltage')
    if 'BEFORE_ONLY' in OC_master_RcontactFit_DF.index.get_level_values(2).unique().tolist() or 'AFTER_ONLY' in TC_master.index.get_level_values(2).unique().tolist():
        oc_analysis.plot_oc_master(OC_master_RcontactFit_DF, args.data_folder, 'only')
    if 'post-ACN' in OC_master_RcontactFit_DF.index.get_level_values(1).unique().tolist():
        oc_analysis.plot_oc_master(OC_master_RcontactFit_DF, args.data_folder, 'post-ACN')
    if 'CONC' in OC_master_RcontactFit_DF.index.get_level_values(2).unique().tolist():
       oc_analysis.plot_oc_master(OC_master_RcontactFit_DF, args.data_folder, 'conc')
    if 'CYCLE' in OC_master_RcontactFit_DF.index.get_level_values(2).unique().tolist():
        oc_analysis.plot_oc_master(OC_master_RcontactFit_DF, args.data_folder, 'cycle')
    
    print('\n\tDone\n')


if __name__ == '__main__':
    try:
        exit(main())
    except KeyboardInterrupt:
        print('\n\nExiting due to KeyboardInterrupt!\n')
