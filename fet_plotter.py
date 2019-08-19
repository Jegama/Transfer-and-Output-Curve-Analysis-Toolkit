import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import pandas as pd
import numpy as np
import math

import seaborn as sns

def remove_outliers(data_lists, deltas = None, m=3): 
    # Remember to put the x axis on the first list and the list that you want to filter as the second
    data = pd.DataFrame(list(zip(*data_lists)))
    main  = data.loc[list(abs((data[1] - np.mean(data[1]) < (m * np.std(data[1])))))]
    main_output = {}
    if deltas:
        extra = main.groupby(0).describe()
        main_output['avg_output'] = extra[1][['mean', 'std']]
    main_output['default'] = main.transpose().values.tolist()
    return main_output

sns.set_style('whitegrid')
sns.set_context('talk')

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

def plot_mobility_plus_reliability(data_folder, plot_id):
    print('\tPlotting the mobility plus reliability')

    mobility_data_temp = pd.read_csv('%s/Results/Mobility.csv' % data_folder, index_col=[0,1,2], header=[0,1,2])
    reliability_factor_temp = pd.read_csv('%s/Results/Reliability_factor.csv' % data_folder, index_col=[0,1,2], header=[0,1,2])

    if plot_id == 'voltage':

        mobility_data = mobility_data_temp.iloc[(mobility_data_temp.index.get_level_values(2) == 'BEFORE') | (mobility_data_temp.index.get_level_values(2) == 'AFTER')]
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'post-ACN']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'DOUBLECHECK']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'REDO']

        # For Delta_mu
        before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        reliability_factor = reliability_factor_temp.iloc[(reliability_factor_temp.index.get_level_values(2) == 'BEFORE') | (reliability_factor_temp.index.get_level_values(2) == 'AFTER')]
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'post-ACN']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'DOUBLECHECK']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'REDO']

        # Get subsets for plotting
        fwd_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'REV']
        rev_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        reliability_fwd_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'FWD']
        reliability_fwd_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'FWD']

        reliability_rev_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'REV']
        reliability_rev_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'REV']

        # Getting X axis - Before
        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))

        # FWD plot

        norm = plt.Normalize(0,np.max(reliability_factor.values))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()

        color_lin_fwd_before = []
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()

        color_lin_fwd_after = []
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()

        # Lin - Delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        color_sat_fwd_before = []
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()

        color_sat_fwd_after = []
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_fwd_before, color_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before, color_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before, color_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before, color_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after, color_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after, color_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after, color_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after, color_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.xlim(0, -1.9)
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV plot

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()

        color_lin_rev_before = []
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()

        color_lin_rev_after = []
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()

        color_sat_rev_before = []
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()

        color_sat_rev_after = []
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_rev_before, color_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before, color_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before, color_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before, color_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after, color_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after, color_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after, color_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after, color_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.xlim((0, -1.9))
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")
        
        plt.gcf().clear()

    elif plot_id == 'only':
        mobility_data = mobility_data_temp.iloc[(mobility_data_temp.index.get_level_values(2) == 'BEFORE_ONLY') | (mobility_data_temp.index.get_level_values(2) == 'AFTER_ONLY')]
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'post-ACN']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'DOUBLECHECK']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'REDO']

        reliability_factor = reliability_factor_temp.iloc[(reliability_factor_temp.index.get_level_values(2) == 'BEFORE_ONLY') | (reliability_factor_temp.index.get_level_values(2) == 'AFTER_ONLY')]
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'post-ACN']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'DOUBLECHECK']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'REDO']

        # For Delta_mu
        before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE_ONLY', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER_ONLY', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        # Get subsets for plotting
        fwd_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE_ONLY', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER_ONLY', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE_ONLY', mobility_data.columns.get_level_values(2) == 'REV']
        rev_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER_ONLY', mobility_data.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        reliability_fwd_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE_ONLY', reliability_factor.columns.get_level_values(2) == 'FWD']
        reliability_fwd_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER_ONLY', reliability_factor.columns.get_level_values(2) == 'FWD']

        reliability_rev_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE_ONLY', reliability_factor.columns.get_level_values(2) == 'REV']
        reliability_rev_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER_ONLY', reliability_factor.columns.get_level_values(2) == 'REV']

        # Getting X axis - Before
        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))

        # FWD plot

        norm = plt.Normalize(0,np.max(reliability_factor.values))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()

        color_lin_fwd_before = []
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()

        color_lin_fwd_after = []
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

         # Lin - Delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()

        # Lin - Delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        color_sat_fwd_before = []
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()

        color_sat_fwd_after = []
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_fwd_before, color_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before, color_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before, color_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before, color_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after, color_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after, color_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after, color_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after, color_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.xlim([-1.9, 1.9])
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        plt.xlim([-1.9, 1.9])
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV plot

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()

        color_lin_rev_before = []
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()

        color_lin_rev_after = []
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()

        color_sat_rev_before = []
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()

        color_sat_rev_after = []
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_rev_before, color_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before, color_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before, color_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before, color_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after, color_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after, color_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after, color_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after, color_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.xlim([-1.9, 1.9])
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")
        plt.gcf().clear()

    elif plot_id == 'post-ACN':
        mobility_data = mobility_data_temp.iloc[mobility_data_temp.index.get_level_values(1) == 'post-ACN']

        reliability_factor = reliability_factor_temp.iloc[reliability_factor_temp.index.get_level_values(1) == 'post-ACN']

        # For Delta_mu
        before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        # Get subsets for plotting
        fwd_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'REV']
        rev_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        reliability_fwd_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'FWD']
        reliability_fwd_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'FWD']

        reliability_rev_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'REV']
        reliability_rev_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'REV']

        # Getting X axis - Before
        x_master_before = []
        # VD1
        x_master_before.extend(fwd_before.index.get_level_values(0).values)
        # VD2
        x_master_before.extend(fwd_before.index.get_level_values(0).values)

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(fwd_after.index.get_level_values(0).values)
        x_master_after.extend(fwd_after.index.get_level_values(0).values)

        # Getting X axis - Delta
        x_master_delta = []
        x_master_delta.extend(fwd_delta.index.get_level_values(0).values)
        x_master_delta.extend(fwd_delta.index.get_level_values(0).values)

        # FWD plot

        norm = plt.Normalize(0,np.max(reliability_factor.values))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()

        color_lin_fwd_before = []
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()

        color_lin_fwd_after = []
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()

        # Lin - delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        color_sat_fwd_before = []
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()

        color_sat_fwd_after = []
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_fwd_before, color_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before, color_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before, color_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before, color_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after, color_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after, color_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after, color_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after, color_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolor='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolor='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV plot

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()

        color_lin_rev_before = []
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()

        color_lin_rev_after = []
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()

        color_sat_rev_before = []
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 10000).tolist()

        color_sat_rev_after = []
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        x_master_before_1, y_lin_rev_before, color_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before, color_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before, color_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before, color_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after, color_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after, color_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after, color_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after, color_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")
            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')
            plt.clim(0,1)
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('REV')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolor='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolor='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', label = 'AFTER (Saturation)')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ \mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()
        
        # REV delta_percentage
        
        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ \mu\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_MU_percentage_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_MU_percentage_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()
        
    elif plot_id == 'conc':
        mobility_data = mobility_data_temp.iloc[mobility_data_temp.index.get_level_values(2) == 'CONC']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'DOUBLECHECK']
        mobility_data = mobility_data.iloc[mobility_data.index.get_level_values(1) != 'REDO']

        reliability_factor = reliability_factor_temp.iloc[reliability_factor_temp.index.get_level_values(2) == 'CONC']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'DOUBLECHECK']
        reliability_factor = reliability_factor.iloc[reliability_factor.index.get_level_values(1) != 'REDO']

        # Get subsets for plotting
        fwd = mobility_data.iloc[:, mobility_data.columns.get_level_values(2) == 'FWD']

        rev = mobility_data.iloc[:, mobility_data.columns.get_level_values(2) == 'REV']

        reliability_fwd = reliability_factor.iloc[:, reliability_factor.columns.get_level_values(2) == 'FWD']

        reliability_rev = reliability_factor.iloc[:, reliability_factor.columns.get_level_values(2) == 'REV']

        # Getting X axis
        x_master = []
        # VD1
        x_master.extend(np.array(fwd.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master.extend(np.array(fwd.index.get_level_values(1).values).astype(np.float))

        # FWD plot

        norm = plt.Normalize(0,np.max(reliability_factor.values))

        # Lin

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()

        color_lin_fwd_before = []
        color_lin_fwd_before.extend(list(zip(*reliability_fwd.iloc[:, reliability_fwd.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_before.extend(list(zip(*reliability_fwd.iloc[:, reliability_fwd.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        color_sat_fwd_before = []
        color_sat_fwd_before.extend(list(zip(*reliability_fwd.iloc[:, reliability_fwd.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_before.extend(list(zip(*reliability_fwd.iloc[:, reliability_fwd.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        x_master_before_1, y_lin_fwd_before, color_lin_fwd_before = remove_outliers([x_master, y_lin_fwd_before, color_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before, color_sat_fwd_before = remove_outliers([x_master, y_sat_fwd_before, color_sat_fwd_before])['default']
        
        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'Linear')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'Saturation')
            sc2.set_facecolor("none")
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'Linear')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'Saturation')
            sc2.set_facecolor("none")
            plt.clim(0,1)
        plt.xscale('log')
        # plt.xticks(rotation=45)
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        plt.xlim([0.001, 1])
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('FWD')
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        plt.savefig('{}/Results/Mobility_plus_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolor='b', lw=2, marker='o', label = 'Linear')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolor='b', lw=2, marker='s', label = 'Saturation')
        sc2.set_facecolor("none")
        plt.xscale('log')
        plt.xlim([0.001, 1])
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV plot

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()

        color_lin_rev_before = []
        color_lin_rev_before.extend(list(zip(*reliability_rev.iloc[:, reliability_rev.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_before.extend(list(zip(*reliability_rev.iloc[:, reliability_rev.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])


        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()

        color_sat_rev_before = []
        color_sat_rev_before.extend(list(zip(*reliability_rev.iloc[:, reliability_rev.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_before.extend(list(zip(*reliability_rev.iloc[:, reliability_rev.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        
        x_master_before_1, y_lin_rev_before, color_lin_rev_before = remove_outliers([x_master, y_lin_rev_before, color_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before, color_sat_rev_before = remove_outliers([x_master, y_sat_rev_before, color_sat_rev_before])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'Linear')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'Saturation')
            sc2.set_facecolor("none")
            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'Linear')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'Saturation')
            sc2.set_facecolor("none")
            plt.clim(0,1)
        plt.xscale('log')
        # plt.xticks(rotation=45)
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        plt.xlim([0.001, 1])
        # plt.gca().invert_xaxis()
        plt.colorbar(label = 'Reliability Factor')
        # plt.title('REV')
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        plt.savefig('{}/Results/Mobility_plus_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolor='b', lw=2, marker='o', label = 'Linear')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolor='b', lw=2, marker='s', label = 'Saturation')
        sc2.set_facecolor("none")
        plt.xscale('log')
        plt.xlim([0.001, 1])
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'cycle':
        temp = mobility_data_temp.iloc[mobility_data_temp.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
        test = pd.DataFrame()
        for i in temp:
            test = test.append(mobility_data_temp.iloc[mobility_data_temp.index.get_level_values(0) == i])

        temp_2 = reliability_factor_temp.iloc[reliability_factor_temp.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
        test_2 = pd.DataFrame()
        for i in temp_2:
            test_2 = test_2.append(reliability_factor_temp.iloc[reliability_factor_temp.index.get_level_values(0) == i])
            
        mobility_data = test
        reliability_factor = test_2

        # Get subsets for plotting
        fwd_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'FWD']
        fwd_2x = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'CYCLE', mobility_data.columns.get_level_values(2) == 'FWD']

        rev_before = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'BEFORE', mobility_data.columns.get_level_values(2) == 'REV']
        rev_after = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'AFTER', mobility_data.columns.get_level_values(2) == 'REV']
        rev_2x = mobility_data.iloc[mobility_data.index.get_level_values(2) == 'CYCLE', mobility_data.columns.get_level_values(2) == 'REV']

        reliability_fwd_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'FWD']
        reliability_fwd_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'FWD']
        reliability_fwd_2x = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'CYCLE', reliability_factor.columns.get_level_values(2) == 'FWD']

        reliability_rev_before = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'BEFORE', reliability_factor.columns.get_level_values(2) == 'REV']
        reliability_rev_after = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'AFTER', reliability_factor.columns.get_level_values(2) == 'REV']
        reliability_rev_2x = reliability_factor.iloc[reliability_factor.index.get_level_values(2) == 'CYCLE', reliability_factor.columns.get_level_values(2) == 'REV']

        # Getting X axis - Before
        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - 2x
        x_master_2x = []
        x_master_2x.extend(np.array(fwd_2x.index.get_level_values(1).values).astype(np.float))
        x_master_2x.extend(np.array(fwd_2x.index.get_level_values(1).values).astype(np.float))

        # FWD plot

        norm = plt.Normalize(0,np.max(reliability_factor.values))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()

        color_lin_fwd_before = []
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()

        color_lin_fwd_after = []
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - 2x

        y_lin_fwd_2x = []
        y_lin_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_2x = np.multiply(y_lin_fwd_2x, 10000).tolist()

        color_lin_fwd_2x = []
        color_lin_fwd_2x.extend(list(zip(*reliability_fwd_2x.iloc[:, reliability_fwd_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_fwd_2x.extend(list(zip(*reliability_fwd_2x.iloc[:, reliability_fwd_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        color_sat_fwd_before = []
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_before.extend(list(zip(*reliability_fwd_before.iloc[:, reliability_fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()

        color_sat_fwd_after = []
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_after.extend(list(zip(*reliability_fwd_after.iloc[:, reliability_fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - 2x

        y_sat_fwd_2x = []
        y_sat_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_2x = np.multiply(y_sat_fwd_2x, 10000).tolist()

        color_sat_fwd_2x = []
        color_sat_fwd_2x.extend(list(zip(*reliability_fwd_2x.iloc[:, reliability_fwd_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_fwd_2x.extend(list(zip(*reliability_fwd_2x.iloc[:, reliability_fwd_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        x_master_before_1, y_lin_fwd_before, color_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before, color_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before, color_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before, color_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after, color_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after, color_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after, color_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after, color_sat_fwd_after])['default']
        x_master_2x_1, y_lin_fwd_2x, color_lin_fwd_2x = remove_outliers([x_master_2x, y_lin_fwd_2x, color_lin_fwd_2x])['default']
        x_master_2x_2, y_sat_fwd_2x, color_sat_fwd_2x = remove_outliers([x_master_2x, y_sat_fwd_2x, color_sat_fwd_2x])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")

            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, alpha=0.5, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, alpha=0.5, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')
            
            # 2x
            plt.scatter(x_master_2x_1, y_lin_fwd_2x, c=color_lin_fwd_2x, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = '2X DOPED (Linear)')
            plt.scatter(x_master_2x_2, y_sat_fwd_2x, c=color_sat_fwd_2x, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = '2X DOPED (Saturation)')

            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_fwd_before, c=color_lin_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, c=color_sat_fwd_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")

            # After
            plt.scatter(x_master_after_1, y_lin_fwd_after, c=color_lin_fwd_after, alpha=0.5, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_fwd_after, c=color_sat_fwd_after, alpha=0.5, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')

            # 2x
            plt.scatter(x_master_2x_1, y_lin_fwd_2x, c=color_lin_fwd_2x, marker='o', cmap='rainbow_r', norm=norm, label = '2X DOPED (Linear)')
            plt.scatter(x_master_2x_2, y_sat_fwd_2x, c=color_sat_fwd_2x, marker='s', cmap='rainbow_r', norm=norm, label = '2X DOPED (Saturation)')

            plt.clim(0,1)
        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        color_bar = plt.colorbar(label = 'Reliability Factor')
        color_bar.set_alpha(1)
        color_bar.draw_all()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        lgd.legendHandles[4].set_color('gray')
        lgd.legendHandles[5].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolor='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolor='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        sc3 = plt.scatter(x_master_after_1, y_lin_fwd_after, edgecolor='r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc3.set_facecolor("none")
        sc4 = plt.scatter(x_master_after_2, y_sat_fwd_after, edgecolor='r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc4.set_facecolor("none")
        # 2x
        plt.scatter(x_master_2x_1, y_lin_fwd_2x, c='r', marker='o', label = '2X DOPED (Linear)')
        plt.scatter(x_master_2x_2, y_sat_fwd_2x, c='r', marker='s', label = '2X DOPED (Saturation)')
        plt.xlim(0, -1.9)
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV plot

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()

        color_lin_rev_before = []
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()

        color_lin_rev_after = []
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - 2x

        y_lin_rev_2x = []
        y_lin_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_2x = np.multiply(y_lin_rev_2x, 10000).tolist()

        color_lin_rev_2x = []
        color_lin_rev_2x.extend(list(zip(*reliability_rev_2x.iloc[:, reliability_rev_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        color_lin_rev_2x.extend(list(zip(*reliability_rev_2x.iloc[:, reliability_rev_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()

        color_sat_rev_before = []
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_before.extend(list(zip(*reliability_rev_before.iloc[:, reliability_rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()

        color_sat_rev_after = []
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_after.extend(list(zip(*reliability_rev_after.iloc[:, reliability_rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - 2x

        y_sat_rev_2x = []
        y_sat_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_2x = np.multiply(y_sat_rev_2x, 10000).tolist()

        color_sat_rev_2x = []
        color_sat_rev_2x.extend(list(zip(*reliability_rev_2x.iloc[:, reliability_rev_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        color_sat_rev_2x.extend(list(zip(*reliability_rev_2x.iloc[:, reliability_rev_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        x_master_before_1, y_lin_rev_before, color_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before, color_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before, color_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before, color_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after, color_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after, color_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after, color_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after, color_sat_rev_after])['default']
        x_master_2x_1, y_lin_rev_2x, color_lin_rev_2x = remove_outliers([x_master_2x, y_lin_rev_2x, color_lin_rev_2x])['default']
        x_master_2x_2, y_sat_rev_2x, color_sat_rev_2x = remove_outliers([x_master_2x, y_sat_rev_2x, color_sat_rev_2x])['default']

        plt.figure(figsize=(14,10))
        if np.max(reliability_factor.values) > 1:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=MidpointNormalize(midpoint=1.), lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")

            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, alpha=0.5, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, alpha=0.5, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = 'AFTER (Saturation)')

            # 2x
            plt.scatter(x_master_2x_1, y_lin_rev_2x, c=color_lin_rev_2x, marker='o', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = '2X DOPED (Linear)')
            plt.scatter(x_master_2x_2, y_sat_rev_2x, c=color_sat_rev_2x, marker='s', norm=MidpointNormalize(midpoint=1.), cmap='rainbow_r', label = '2X DOPED (Saturation)')

            plt.clim(0, np.max(reliability_factor.values))
        else:
            # Before
            sc = plt.scatter(x_master_before_1, y_lin_rev_before, c=color_lin_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='o', label = 'BEFORE (Linear)')
            sc.set_facecolor("none")
            sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, c=color_sat_rev_before, cmap='rainbow_r', norm=norm, lw=2, marker='s', label = 'BEFORE (Saturation)')
            sc2.set_facecolor("none")

            # After
            plt.scatter(x_master_after_1, y_lin_rev_after, c=color_lin_rev_after, alpha=0.5, marker='o', cmap='rainbow_r', norm=norm, label = 'AFTER (Linear)')
            plt.scatter(x_master_after_2, y_sat_rev_after, c=color_sat_rev_after, alpha=0.5, marker='s', cmap='rainbow_r', norm=norm, label = 'AFTER (Saturation)')

            # 2x
            plt.scatter(x_master_2x_1, y_lin_rev_2x, c=color_lin_rev_2x, marker='o', cmap='rainbow_r', norm=norm, label = '2X DOPED (Linear)')
            plt.scatter(x_master_2x_2, y_sat_rev_2x, c=color_sat_rev_2x, marker='s', cmap='rainbow_r', norm=norm, label = '2X DOPED (Saturation)')

            plt.clim(0,1)
        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        color_bar = plt.colorbar(label = 'Reliability Factor')
        color_bar.set_alpha(1)
        color_bar.draw_all()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4} (cm^2/Vs)$')
        lgd = plt.legend()
        lgd.legendHandles[0].set_edgecolor('gray')
        lgd.legendHandles[1].set_edgecolor('gray')
        lgd.legendHandles[2].set_color('gray')
        lgd.legendHandles[3].set_color('gray')
        lgd.legendHandles[4].set_color('gray')
        lgd.legendHandles[5].set_color('gray')
        plt.savefig('{}/Results/Mobility_plus_reliability_REV_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_plus_reliability_REV_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Mobility without reliability 

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolor='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolor='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        sc3 = plt.scatter(x_master_after_1, y_lin_rev_after, edgecolor='r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc3.set_facecolor("none")
        sc4 = plt.scatter(x_master_after_2, y_sat_rev_after, edgecolor='r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc4.set_facecolor("none")
        # 2x
        plt.scatter(x_master_2x_1, y_lin_rev_2x, c='r', marker='o', label = '2X DOPED (Linear)')
        plt.scatter(x_master_2x_2, y_sat_rev_2x, c='r', marker='s', label = '2X DOPED (Saturation)')
        plt.xlim(0, -1.9)
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\mu\ x\ 10^{-4}\ (cm^2/Vs)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Mobility_without_reliability_FWD_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Mobility_without_reliability_FWD_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

def plot_max_hysteresis(data_folder, plot_id):

    print('\tPlotting Max Hysteresis')

    df = pd.read_csv('%s/Results/Max_Hysteresis.csv' % data_folder, index_col=[0,1,2], header=[0,1])

    if plot_id == 'voltage':
        df = df.iloc[(df.index.get_level_values(2) == 'BEFORE') | (df.index.get_level_values(2) == 'AFTER')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]

        # For Delta_mu
        before_delta = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before_delta.reset_index(level=2, drop=True, inplace=True)
        after_delta = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after_delta.reset_index(level=2, drop=True, inplace=True)

        delta_mu = after_delta - before_delta
        delta_mu_percentage = (after_delta - before_delta) / np.absolute(before_delta)

        x_master_before = []
        # VD1
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))

        x_master_after = []
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(delta_mu.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(delta_mu.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_before = []
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_before = np.multiply(y_lin_before, 10000000).tolist()

        # Lin - After

        y_lin_after = []
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_after = np.multiply(y_lin_after, 10000000).tolist()

        # Lin - Delta

        y_lin_delta = []
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta = np.multiply(y_lin_delta, 10000000).tolist()

        # Lin - Delta_percentage

        y_lin_delta_percentage = []
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta_percentage = np.multiply(y_lin_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_before = []
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_before = np.multiply(y_sat_before, 10000000).tolist()

        # Sat - After

        y_sat_after = []
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_after = np.multiply(y_sat_after, 10000000).tolist()

        # Sat - Delta

        y_sat_delta = []
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta = np.multiply(y_sat_delta, 10000000).tolist()

        # Sat - Delta_percentage

        y_sat_delta_percentage = []
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta_percentage = np.multiply(y_sat_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_before = remove_outliers([x_master_before, y_lin_before])['default']
        x_master_before_2, y_sat_before = remove_outliers([x_master_before, y_sat_before])['default']
        x_master_after_1, y_lin_after = remove_outliers([x_master_after, y_lin_after])['default']
        x_master_after_2, y_sat_after = remove_outliers([x_master_after, y_sat_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'Hysteresis in $I_D\ x\ 10^{-7}$ (A)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta

        temp = remove_outliers([x_master_delta, y_lin_delta], deltas = True)
        x_master_delta_1, y_lin_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta], deltas = True)
        x_master_delta_2, y_sat_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_delta) < 0) & np.all(np.array(y_sat_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta) > 0) & np.all(np.array(y_sat_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ x\ 10^{-7} (A)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_delta_percentage) < 0) & np.all(np.array(y_sat_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta_percentage) > 0) & np.all(np.array(y_sat_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_percentage_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_percentage_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'only':
        df = df.iloc[(df.index.get_level_values(2) == 'BEFORE_ONLY') | (df.index.get_level_values(2) == 'AFTER_ONLY')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY']
        after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY']

        # For Delta_mu
        before_delta = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', :]
        before_delta.reset_index(level=2, drop=True, inplace=True)
        after_delta = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', :]
        after_delta.reset_index(level=2, drop=True, inplace=True)

        delta_mu = after_delta - before_delta
        delta_mu_percentage = (after_delta - before_delta) / np.absolute(before_delta)

        x_master_before = []
        # VD1
        x_master_before = []
        # VD1
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))


        x_master_after = []
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(delta_mu.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(delta_mu.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_before = []
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_before = np.multiply(y_lin_before, 10000000).tolist()

        # Lin - After

        y_lin_after = []
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_after = np.multiply(y_lin_after, 10000000).tolist()

        # Lin - Delta

        y_lin_delta = []
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta = np.multiply(y_lin_delta, 10000000).tolist()

        # Lin - Delta_percentage

        y_lin_delta_percentage = []
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta_percentage = np.multiply(y_lin_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_before = []
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_before = np.multiply(y_sat_before, 10000000).tolist()

        # Sat - After

        y_sat_after = []
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_after = np.multiply(y_sat_after, 10000000).tolist()

        # Sat - Delta

        y_sat_delta = []
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta = np.multiply(y_sat_delta, 10000000).tolist()

        # Sat - Delta_percentage

        y_sat_delta_percentage = []
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta_percentage = np.multiply(y_sat_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_before = remove_outliers([x_master_before, y_lin_before])['default']
        x_master_before_2, y_sat_before = remove_outliers([x_master_before, y_sat_before])['default']
        x_master_after_1, y_lin_after = remove_outliers([x_master_after, y_lin_after])['default']
        x_master_after_2, y_sat_after = remove_outliers([x_master_after, y_sat_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([-1.9, 1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'Hysteresis in $I_D\ x\ 10^{-7}$ (A)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta

        temp = remove_outliers([x_master_delta, y_lin_delta], deltas = True)
        x_master_delta_1, y_lin_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta], deltas = True)
        x_master_delta_2, y_sat_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_delta) < 0) & np.all(np.array(y_sat_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta) > 0) & np.all(np.array(y_sat_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ x\ 10^{-7} (A)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_delta_percentage) < 0) & np.all(np.array(y_sat_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta_percentage) > 0) & np.all(np.array(y_sat_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_percentage_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_percentage_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'post-ACN':
        df = df.iloc[df.index.get_level_values(1) == 'post-ACN']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE']
        after = df.iloc[df.index.get_level_values(2) == 'AFTER']

        # For Delta_mu
        before_delta = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before_delta.reset_index(level=2, drop=True, inplace=True)
        after_delta = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after_delta.reset_index(level=2, drop=True, inplace=True)

        delta_mu = after_delta - before_delta
        delta_mu_percentage = (after_delta - before_delta) / np.absolute(before_delta)

        x_master_before = []
        # VD1
        x_master_before.extend(before.index.get_level_values(0).values)
        # VD2
        x_master_before.extend(before.index.get_level_values(0).values)


        x_master_after = []
        x_master_after.extend(after.index.get_level_values(0).values)
        x_master_after.extend(after.index.get_level_values(0).values)

        x_master_delta = []
        x_master_delta.extend(delta_mu.index.get_level_values(0).values)
        x_master_delta.extend(delta_mu.index.get_level_values(0).values)

        # Lin - Before

        y_lin_before = []
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_before = np.multiply(y_lin_before, 10000000).tolist()

        # Lin - After

        y_lin_after = []
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_after = np.multiply(y_lin_after, 10000000).tolist()

        # Lin - Delta

        y_lin_delta = []
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta = np.multiply(y_lin_delta, 10000000).tolist()

        # Lin - Delta_percentage

        y_lin_delta_percentage = []
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_delta_percentage = np.multiply(y_lin_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_before = []
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_before = np.multiply(y_sat_before, 10000000).tolist()

        # Sat - After

        y_sat_after = []
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_after = np.multiply(y_sat_after, 10000000).tolist()

        # Sat - Delta

        y_sat_delta = []
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta.extend(list(zip(*delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta = np.multiply(y_sat_delta, 10000000).tolist()

        # Sat - Delta_percentage

        y_sat_delta_percentage = []
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_delta_percentage.extend(list(zip(*delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_delta_percentage = np.multiply(y_sat_delta_percentage, 100).tolist()

        x_master_before_1, y_lin_before = remove_outliers([x_master_before, y_lin_before])['default']
        x_master_before_2, y_sat_before = remove_outliers([x_master_before, y_sat_before])['default']
        x_master_after_1, y_lin_after = remove_outliers([x_master_after, y_lin_after])['default']
        x_master_after_2, y_sat_after = remove_outliers([x_master_after, y_sat_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'Hysteresis in $I_D\ x\ 10^{-7}$ (A)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta

        temp = remove_outliers([x_master_delta, y_lin_delta], deltas = True)
        x_master_delta_1, y_lin_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta], deltas = True)
        x_master_delta_2, y_sat_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_delta) < 0) & np.all(np.array(y_sat_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta) > 0) & np.all(np.array(y_sat_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ x\ 10^{-7} (A)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_delta_percentage) < 0) & np.all(np.array(y_sat_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_delta_percentage) > 0) & np.all(np.array(y_sat_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xticks(rotation=45)
        plt.xlabel('Device name')
        plt.ylabel(r'$\Delta\ Hysteresis\ in\ I_D\ (\%)$')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_Max_Hysteresis_percentage_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_Max_Hysteresis_percentage_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'conc':
        df = df.iloc[df.index.get_level_values(2) == 'CONC']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        x_master_before = []
        x_master_before.extend(np.array(df.index.get_level_values(1).values).astype(np.float))
        x_master_before.extend(np.array(df.index.get_level_values(1).values).astype(np.float))

        # Lin

        y_lin_before = []
        y_lin_before.extend(list(zip(*df.iloc[:, df.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_before.extend(list(zip(*df.iloc[:, df.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_before = np.multiply(y_lin_before, 10000000).tolist()

        # Sat

        y_sat_before = []
        y_sat_before.extend(list(zip(*df.iloc[:, df.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_before.extend(list(zip(*df.iloc[:, df.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_before = np.multiply(y_sat_before, 10000000).tolist()

        x_master_before_1, y_lin_before = remove_outliers([x_master_before, y_lin_before])['default']
        x_master_before_2, y_sat_before = remove_outliers([x_master_before, y_sat_before])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'Linear')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'Saturation')
        sc2.set_facecolor("none")
        plt.xscale('log')
        plt.xlim([0.001, 1])
        # plt.xticks(rotation=45)
        # plt.title('FWD')
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(r'Hysteresis in $I_D\ x\ 10^{-7}$ (A)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Max_Hysteresis_plot%s.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'cycle':

        temp = df.iloc[df.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
        test = pd.DataFrame()
        for i in temp:
            test = test.append(df.iloc[df.index.get_level_values(0) == i])
        
        df = test

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE']
        after = df.iloc[df.index.get_level_values(2) == 'AFTER']
        _2x = df.iloc[df.index.get_level_values(2) == 'CYCLE']

        x_master_before = []
        # VD1
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(before.index.get_level_values(1).values).astype(np.float))


        x_master_after = []
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(after.index.get_level_values(1).values).astype(np.float))

        x_master_2x = []
        x_master_2x.extend(np.array(_2x.index.get_level_values(1).values).astype(np.float))
        x_master_2x.extend(np.array(_2x.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_before = []
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_before = np.multiply(y_lin_before, 10000000).tolist()

        # Lin - After

        y_lin_after = []
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_after = np.multiply(y_lin_after, 10000000).tolist()

        # Lin - 2x

        y_lin_2x = []
        y_lin_2x.extend(list(zip(*_2x.iloc[:, _2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_2x.extend(list(zip(*_2x.iloc[:, _2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_2x = np.multiply(y_lin_2x, 10000000).tolist()

        # Sat - Before

        y_sat_before = []
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_before.extend(list(zip(*before.iloc[:, before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_before = np.multiply(y_sat_before, 10000000).tolist()

        # Sat - After

        y_sat_after = []
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_after.extend(list(zip(*after.iloc[:, after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_after = np.multiply(y_sat_after, 10000000).tolist()

        # Sat - 2x

        y_sat_2x = []
        y_sat_2x.extend(list(zip(*_2x.iloc[:, _2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_2x.extend(list(zip(*_2x.iloc[:, _2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_2x = np.multiply(y_sat_2x, 10000000).tolist()

        x_master_before_1, y_lin_before = remove_outliers([x_master_before, y_lin_before])['default']
        x_master_before_2, y_sat_before = remove_outliers([x_master_before, y_sat_before])['default']
        x_master_after_1, y_lin_after = remove_outliers([x_master_after, y_lin_after])['default']
        x_master_after_2, y_sat_after = remove_outliers([x_master_after, y_sat_after])['default']
        x_master_2x_1, y_lin_2x = remove_outliers([x_master_2x, y_lin_2x])['default']
        x_master_2x_2, y_sat_2x = remove_outliers([x_master_2x, y_sat_2x])['default']
        
        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")

        # After
        sc = plt.scatter(x_master_after_1, y_lin_after, edgecolors='r', lw=2, marker='o', label = 'AFTER (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_after_2, y_sat_after, edgecolors='r', lw=2, marker='s', label = 'AFTER (Saturation)')
        sc2.set_facecolor("none")

        # 2x
        plt.scatter(x_master_2x_1, y_lin_2x, c='r', marker='o', label = '2X DOPED (Linear)')
        plt.scatter(x_master_2x_2, y_sat_2x, c='r', marker='s', label = '2X DOPED (Saturation)')

        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'Hysteresis in $I_D\ x\ 10^{-7}$ (A)')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Max_Hysteresis_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Max_Hysteresis_plot%s.svg'.format(data_folder, plot_id), bbox_inches="tight")

        plt.gcf().clear()

def general_plotting(data_folder, plot_name, y_label, plot_id, lable_2 = None):

    print('\tPlotting %s' % plot_name)

    df = pd.read_csv('{}/Results/{}.csv'.format(data_folder, plot_name), index_col=[0,1,2], header=[0,1,2]).sort_index(level=0)

    if plot_id == 'voltage':
        df = df.iloc[(df.index.get_level_values(2) == 'BEFORE') | (df.index.get_level_values(2) == 'AFTER')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        # For Delta_mu
        before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        fwd_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'FWD']
        fwd_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'REV']
        rev_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        x_master_before_1, y_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        x_master_before_1, y_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_rev_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_rev_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'only':
        df = df.iloc[(df.index.get_level_values(2) == 'BEFORE_ONLY') | (df.index.get_level_values(2) == 'AFTER_ONLY')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        # For Delta_mu
        before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        fwd_before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', df.columns.get_level_values(2) == 'FWD']
        fwd_after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', df.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', df.columns.get_level_values(2) == 'REV']
        rev_after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', df.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - Delta mu
        x_master_delta = []
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))
        x_master_delta.extend(np.array(fwd_delta.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        x_master_before_1, y_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([-1.9, 1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([0, -1.9])
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlim([-1.9, 1.9])
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        x_master_before_1, y_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_rev_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_rev_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        plt.xlim([-1.9, 1.9])
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        plt.xlim([-1.9, 1.9])
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'post-ACN':
        df = df.iloc[df.index.get_level_values(1) == 'post-ACN']

        # For Delta_mu
        before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu_percentage = (after - before) / np.absolute(before)

        fwd_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'FWD']
        fwd_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'FWD']
        fwd_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'FWD']
        fwd_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'FWD']

        rev_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'REV']
        rev_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'REV']
        rev_delta = delta_mu.iloc[:, delta_mu.columns.get_level_values(2) == 'REV']
        rev_delta_percentage = delta_mu_percentage.iloc[:, delta_mu_percentage.columns.get_level_values(2) == 'REV']

        x_master_before = []
        # VD1
        x_master_before.extend(fwd_before.index.get_level_values(0).values)
        # VD2
        x_master_before.extend(fwd_before.index.get_level_values(0).values)

        x_master_after = []
        x_master_after.extend(fwd_after.index.get_level_values(0).values)
        x_master_after.extend(fwd_after.index.get_level_values(0).values)

        # Getting X axis - Delta
        x_master_delta = []
        x_master_delta.extend(fwd_delta.index.get_level_values(0).values)
        x_master_delta.extend(fwd_delta.index.get_level_values(0).values)

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_fwd_delta = []
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_fwd_delta_percentage = []
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_fwd_delta_percentage = np.multiply(y_lin_fwd_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_fwd_delta = []
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta.extend(list(zip(*fwd_delta.iloc[:, fwd_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_fwd_delta_percentage = []
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage.extend(list(zip(*fwd_delta_percentage.iloc[:, fwd_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_fwd_delta_percentage = np.multiply(y_sat_fwd_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()
            y_lin_fwd_delta = np.multiply(y_lin_fwd_delta, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()
            y_sat_fwd_delta = np.multiply(y_sat_fwd_delta, 10000).tolist()

        x_master_before_1, y_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_fwd_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_fwd_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta], deltas = True)
        x_master_delta_1, y_lin_fwd_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta], deltas = True)
        x_master_delta_2, y_sat_fwd_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_fwd_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_fwd_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_fwd_delta) < 0) & np.all(np.array(y_sat_fwd_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta) > 0) & np.all(np.array(y_sat_fwd_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # FWD delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_fwd_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_fwd_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_fwd_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_fwd_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_fwd_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_fwd_delta_percentage) < 0) & np.all(np.array(y_sat_fwd_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_fwd_delta_percentage) > 0) & np.all(np.array(y_sat_fwd_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta

        y_lin_rev_delta = []
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - Delta_percentage

        y_lin_rev_delta_percentage = []
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])
        y_lin_rev_delta_percentage = np.multiply(y_lin_rev_delta_percentage, 100).tolist()

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta

        y_sat_rev_delta = []
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta.extend(list(zip(*rev_delta.iloc[:, rev_delta.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - Delta_percentage

        y_sat_rev_delta_percentage = []
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_delta_percentage.extend(list(zip(*rev_delta_percentage.iloc[:, rev_delta_percentage.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])
        y_sat_rev_delta_percentage = np.multiply(y_sat_rev_delta_percentage, 100).tolist()

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000).tolist()
            y_lin_rev_delta = np.multiply(y_lin_rev_delta, 10000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000).tolist()
            y_sat_rev_delta = np.multiply(y_sat_rev_delta, 10000).tolist()

        x_master_before_1, y_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', cmap='rainbow_r', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")
        # After
        plt.scatter(x_master_after_1, y_lin_rev_after, c='r', marker='o', cmap='rainbow_r', label = 'AFTER (Linear)')
        plt.scatter(x_master_after_2, y_sat_rev_after, c='r', marker='s', cmap='rainbow_r', label = 'AFTER (Saturation)')
        # plt.xlim([0, -1.9])
        plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta

        temp = remove_outliers([x_master_delta, y_lin_rev_delta], deltas = True)
        x_master_delta_1, y_lin_rev_delta = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta], deltas = True)
        x_master_delta_2, y_sat_rev_delta = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_1, y_lin_rev_delta, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_2, y_sat_rev_delta, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_rev_delta) < 0) & np.all(np.array(y_sat_rev_delta) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta) > 0) & np.all(np.array(y_sat_rev_delta) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(r'$\Delta\ $' + y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # REV delta_percentage

        temp = remove_outliers([x_master_delta, y_lin_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_1, y_lin_rev_delta_percentage = temp['default']
        avg_lin = temp['avg_output']
        temp = remove_outliers([x_master_delta, y_sat_rev_delta_percentage], deltas = True)
        x_master_delta_percentage_2, y_sat_rev_delta_percentage = temp['default']
        avg_sat = temp['avg_output']

        plt.figure(figsize=(14,10))
        plt.plot(avg_lin.index.values, avg_lin['mean'].values, '--', color='r', alpha = 0.75)
        plt.scatter(avg_lin.index.values, avg_lin['mean'].values, c='r', marker='o')
        plt.fill_between(avg_lin.index.values, avg_lin['mean'].values - avg_lin['std'].values, avg_lin['mean'].values + avg_lin['std'].values, color='r', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_1, y_lin_rev_delta_percentage, c='r', edgecolors = 'none', alpha=0.25, marker='o', label = 'Linear')
        plt.plot(avg_sat.index.values, avg_sat['mean'].values, '--', color='b', alpha = 0.75)
        plt.scatter(avg_sat.index.values, avg_sat['mean'].values, c='b', marker='s')
        plt.fill_between(avg_sat.index.values, avg_sat['mean'].values - avg_sat['std'].values, avg_sat['mean'].values + avg_sat['std'].values, color='b', alpha=0.2, lw = 0)
        plt.scatter(x_master_delta_percentage_2, y_sat_rev_delta_percentage, c='blue', edgecolors = 'none', alpha=0.25, marker='s', label = 'Saturation')
        if np.all(np.array(y_lin_rev_delta_percentage) < 0) & np.all(np.array(y_sat_rev_delta_percentage) < 0):
            plt.ylim(top = 0)
        elif np.all(np.array(y_lin_rev_delta_percentage) > 0) & np.all(np.array(y_sat_rev_delta_percentage) > 0):
            plt.ylim(bottom = 0)
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Device name')
        plt.xticks(rotation=45)
        plt.ylabel(lable_2)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/Delta_percentage_{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/Delta_percentage_{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'conc':
        df = df.iloc[df.index.get_level_values(2) == 'CONC']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        # Get subsets for plotting
        fwd = df.iloc[:, df.columns.get_level_values(2) == 'FWD']

        rev = df.iloc[:, df.columns.get_level_values(2) == 'REV']

        x_master_before = []

        # test_label = [device.replace('_', '\n') for device in fwd.index.get_level_values(0).values]

        # VD1
        x_master_before.extend(np.array(fwd.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd.iloc[:, fwd.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        x_master_before_1, y_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before])['default']

        plt.figure(figsize=(14,10))

        plt.scatter(x_master_before_1, y_lin_fwd_before, c='r', marker='o', cmap='rainbow_r', label = 'Linear')
        plt.scatter(x_master_before_2, y_sat_fwd_before, c='r', marker='s', cmap='rainbow_r', label = 'Saturation')

        plt.xscale('log')
        # plt.xticks(rotation=45)
        plt.xlim([0.001, 1])
        # plt.title('FWD')
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev.iloc[:, rev.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()

        x_master_before_1, y_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before])['default']

        plt.figure(figsize=(14,10))

        plt.scatter(x_master_before_1, y_lin_rev_before, c='r', marker='o', cmap='rainbow_r', label = 'Linear')
        plt.scatter(x_master_before_2, y_sat_rev_before, c='r', marker='s', cmap='rainbow_r', label = 'Saturation')
        plt.xscale('log')
        # plt.xticks(rotation=45)
        plt.xlim([0.001, 1])
        # plt.title('REV')
        plt.xlabel('F4TCNQ Concentration (mg/mL)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

    elif plot_id == 'cycle':
        temp = df.iloc[df.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
        test = pd.DataFrame()
        for i in temp:
            test = test.append(df.iloc[df.index.get_level_values(0) == i])
        
        df = test

        fwd_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'FWD']
        fwd_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'FWD']
        fwd_2x = df.iloc[df.index.get_level_values(2) == 'CYCLE', df.columns.get_level_values(2) == 'FWD']

        rev_before = df.iloc[df.index.get_level_values(2) == 'BEFORE', df.columns.get_level_values(2) == 'REV']
        rev_after = df.iloc[df.index.get_level_values(2) == 'AFTER', df.columns.get_level_values(2) == 'REV']
        rev_2x = df.iloc[df.index.get_level_values(2) == 'CYCLE', df.columns.get_level_values(2) == 'REV']

        x_master_before = []
        # VD1
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))
        # VD2
        x_master_before.extend(np.array(fwd_before.index.get_level_values(1).values).astype(np.float))

        # Getting X axis - After
        x_master_after = []
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))
        x_master_after.extend(np.array(fwd_after.index.get_level_values(1).values).astype(np.float))

        x_master_2x = []
        x_master_2x.extend(np.array(fwd_2x.index.get_level_values(1).values).astype(np.float))
        x_master_2x.extend(np.array(fwd_2x.index.get_level_values(1).values).astype(np.float))

        # Lin - Before

        y_lin_fwd_before = []
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_fwd_after = []
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - 2x

        y_lin_fwd_2x = []
        y_lin_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_fwd_before = []
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_before.extend(list(zip(*fwd_before.iloc[:, fwd_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_fwd_after = []
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_after.extend(list(zip(*fwd_after.iloc[:, fwd_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - 2x

        y_sat_fwd_2x = []
        y_sat_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_fwd_2x.extend(list(zip(*fwd_2x.iloc[:, fwd_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000000).tolist()
            y_lin_fwd_2x = np.multiply(y_lin_fwd_2x, 10000000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000000).tolist()
            y_sat_fwd_2x = np.multiply(y_sat_fwd_2x, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()
            y_lin_fwd_2x = np.multiply(y_lin_fwd_2x, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()
            y_sat_fwd_2x = np.multiply(y_sat_fwd_2x, 10000).tolist()

        x_master_before_1, y_lin_fwd_before = remove_outliers([x_master_before, y_lin_fwd_before])['default']
        x_master_before_2, y_sat_fwd_before = remove_outliers([x_master_before, y_sat_fwd_before])['default']
        x_master_after_1, y_lin_fwd_after = remove_outliers([x_master_after, y_lin_fwd_after])['default']
        x_master_after_2, y_sat_fwd_after = remove_outliers([x_master_after, y_sat_fwd_after])['default']
        x_master_2x_1, y_lin_fwd_2x = remove_outliers([x_master_2x, y_lin_fwd_2x])['default']
        x_master_2x_2, y_sat_fwd_2x = remove_outliers([x_master_2x, y_sat_fwd_2x])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_fwd_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_fwd_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")

        # After
        sc3 = plt.scatter(x_master_after_1, y_lin_fwd_after, edgecolors='r', lw=2, marker='o', label = 'AFTER (Linear)')
        sc3.set_facecolor("none")
        sc4 = plt.scatter(x_master_after_2, y_sat_fwd_after, edgecolors='r', lw=2, marker='s', label = 'AFTER (Saturation)')
        sc4.set_facecolor("none")

        # 2X
        plt.scatter(x_master_2x_1, y_lin_fwd_2x, c='r', marker='o', label = '2X DOPED (Linear)')
        plt.scatter(x_master_2x_2, y_sat_fwd_2x, c='r', marker='s', label = '2X DOPED (Saturation)')

        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('FWD')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_FWD_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_FWD_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()

        # Lin - Before

        y_lin_rev_before = []
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - After

        y_lin_rev_after = []
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Lin - 2X

        y_lin_rev_2x = []
        y_lin_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd1'].values.tolist()))[0])
        y_lin_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd2'].values.tolist()))[0])

        # Sat - Before

        y_sat_rev_before = []
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_before.extend(list(zip(*rev_before.iloc[:, rev_before.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - After

        y_sat_rev_after = []
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_after.extend(list(zip(*rev_after.iloc[:, rev_after.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        # Sat - 2X

        y_sat_rev_2x = []
        y_sat_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd(-2)'].values.tolist()))[0])
        y_sat_rev_2x.extend(list(zip(*rev_2x.iloc[:, rev_2x.columns.get_level_values(1) == 'Vd(-1)'].values.tolist()))[0])

        if plot_name in ['ID_at_VG_0', 'Max_ID', 'Min_ID']:
            y_lin_rev_before = np.multiply(y_lin_rev_before, 10000000).tolist()
            y_lin_rev_after = np.multiply(y_lin_rev_after, 10000000).tolist()
            y_lin_rev_2x = np.multiply(y_lin_rev_2x, 10000000).tolist()
            y_sat_rev_before = np.multiply(y_sat_rev_before, 10000000).tolist()
            y_sat_rev_after = np.multiply(y_sat_rev_after, 10000000).tolist()
            y_sat_rev_2x = np.multiply(y_sat_rev_2x, 10000000).tolist()
        elif plot_name == 'MU_eff':
            y_lin_fwd_before = np.multiply(y_lin_fwd_before, 10000).tolist()
            y_lin_fwd_after = np.multiply(y_lin_fwd_after, 10000).tolist()
            y_lin_fwd_2x = np.multiply(y_lin_fwd_2x, 10000).tolist()
            y_sat_fwd_before = np.multiply(y_sat_fwd_before, 10000).tolist()
            y_sat_fwd_after = np.multiply(y_sat_fwd_after, 10000).tolist()
            y_sat_fwd_2x = np.multiply(y_sat_fwd_2x, 10000).tolist()

        x_master_before_1, y_lin_rev_before = remove_outliers([x_master_before, y_lin_rev_before])['default']
        x_master_before_2, y_sat_rev_before = remove_outliers([x_master_before, y_sat_rev_before])['default']
        x_master_after_1, y_lin_rev_after = remove_outliers([x_master_after, y_lin_rev_after])['default']
        x_master_after_2, y_sat_rev_after = remove_outliers([x_master_after, y_sat_rev_after])['default']
        x_master_2x_1, y_lin_rev_2x = remove_outliers([x_master_2x, y_lin_rev_2x])['default']
        x_master_2x_2, y_sat_rev_2x = remove_outliers([x_master_2x, y_sat_rev_2x])['default']

        plt.figure(figsize=(14,10))
        # Before
        sc = plt.scatter(x_master_before_1, y_lin_rev_before, edgecolors='b', lw=2, marker='o', label = 'BEFORE (Linear)')
        sc.set_facecolor("none")
        sc2 = plt.scatter(x_master_before_2, y_sat_rev_before, edgecolors='b', lw=2, marker='s', label = 'BEFORE (Saturation)')
        sc2.set_facecolor("none")

        # After
        sc3 = plt.scatter(x_master_after_1, y_lin_rev_after, edgecolors='r', lw=2, marker='o', label = 'AFTER (Linear)')
        sc3.set_facecolor("none")
        sc4 = plt.scatter(x_master_after_2, y_sat_rev_after, edgecolors='r', lw=2, marker='s', label = 'AFTER (Saturation)')
        sc4.set_facecolor("none")

        # 2X
        plt.scatter(x_master_2x_1, y_lin_rev_2x, c='r', marker='o', label = '2X DOPED (Linear)')
        plt.scatter(x_master_2x_2, y_sat_rev_2x, c='r', marker='s', label = '2X DOPED (Saturation)')

        plt.xlim([0, -1.9])
        # plt.gca().invert_xaxis()
        # plt.title('REV')
        plt.xlabel('Voltage applied (V)')
        plt.ylabel(y_label)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig('{}/Results/{}_REV_plot_{}.png'.format(data_folder, plot_name, plot_id), bbox_inches="tight")
        plt.savefig('{}/Results/SVG_plots/{}_REV_plot_{}.svg'.format(data_folder, plot_name, plot_id), bbox_inches="tight")

        plt.gcf().clear()




