import numpy as np
import pandas as pd
import seaborn as sns
import os, re, math
from scipy import stats
import matplotlib.pyplot as plt

sns.set_style('whitegrid')
sns.set_context('talk')

def create_intercept_line(x, slope, intercept):
    return (np.array(x) * slope + intercept).tolist()
    # return [slope * i + intercept for i in x]


class oc_analysis:
    def __init__(self):
        self.Ci = 1.72575*(10**(-8))

    def generate_empty_dataframes(self, file_list):
        # Go through all the files and get the main DataFrame to analyze TC
        first_level = []
        second_level = []
        third_level = []
        temp_filenames = []

        for filename in file_list:
            filename = re.sub(r'/', '_',filename)
            if filename.endswith('OC.csv'):
                foreveralone = True
                # This section deals with the afters
                first_level.append(re.findall('\S+\s\S+\s\S+\s\S+', filename)[0])
                temp_filenames.append(filename)
                print('\n%s' % filename)
                if re.findall("\+-1V", filename):
                    if re.findall('redo', filename.lower()):
                        second_level.append('REDO')
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('doublecheck', filename):
                        second_level.append('DOUBLECHECK')
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('ONLY', filename):
                        second_level.append(re.findall("\S\d+V", filename)[0][:-1])
                        third_level.append('ONLY')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('x2extra', filename):
                        second_level.append(re.findall("\S\d+V", filename)[0][:-1])
                        third_level.append('CYCLE')
                        print(second_level[-1], third_level[-1])
                    else:
                        second_level.append(re.findall("\S\d+V", filename)[0][:-1])
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                elif re.findall("\S\d+\.\d+V", filename):
                    if re.findall('redo', filename.lower()):
                        second_level.append('REDO')
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('doublecheck', filename):
                        second_level.append('DOUBLECHECK')
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('ONLY', filename):
                        second_level.append(re.findall("\S\d+\.\d+", filename)[0])
                        third_level.append('AFTER_ONLY')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('x2extra', filename):
                        second_level.append(re.findall("\S\d+\.\d+", filename)[0])
                        third_level.append('CYCLE')
                        print(second_level[-1], third_level[-1])
                    else:
                        second_level.append(re.findall("\S\d+\.\d+", filename)[0])
                        third_level.append('AFTER')
                        print(second_level[-1], third_level[-1])
                elif re.findall('post-ACN', filename):
                    second_level.append('post-ACN')
                    third_level.append('AFTER')
                    print(second_level[-1], third_level[-1])
                elif re.findall('F4TCNQ', filename):
                    if re.findall('redo', filename.lower()):
                        second_level.append('REDO')
                        third_level.append('CONC')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('doublecheck', filename):
                        second_level.append('DOUBLECHECK')
                        third_level.append('CONC')
                        print(second_level[-1], third_level[-1])
                    else:
                        second_level.append(re.findall('F4TCNQ_\S+', filename)[0][7:])
                        third_level.append('CONC')
                        print(second_level[-1], third_level[-1])
                else:
                    if re.findall('doublecheck', filename):
                        second_level.append('DOUBLECHECK')
                        third_level.append('BEFORE')
                        print(second_level[-1], third_level[-1])
                    elif re.findall('redo', filename.lower()):
                        second_level.append('REDO')
                        third_level.append('BEFORE')
                        print(second_level[-1], third_level[-1])
                    else:
                        before_name = re.findall('\S+\s\S+\s\S+\s\S+', filename)[0]
                        for file in file_list:
                            file = re.sub(r'/', '_',file)
                            if file.endswith('OC.csv'):
                                if re.findall('\S+\s\S+\s\S+\s\S+', file)[0] == before_name and file != filename:
                                    if re.findall("\+-1V", file):
                                        second_level.append(re.findall("\S\d+V", file)[0][:-1])
                                        if re.findall('ONLY', file):
                                            third_level.append('BEFORE_ONLY')
                                        else:
                                            third_level.append('BEFORE')
                                        print(second_level[-1], third_level[-1])
                                    elif re.findall("\S\d+\.\d+V", file):
                                        second_level.append(re.findall("\S\d+\.\d+", file)[0])
                                        if re.findall('ONLY', file):
                                            third_level.append('BEFORE_ONLY')
                                        else:
                                            third_level.append('BEFORE')
                                        print(second_level[-1], third_level[-1])
                                    elif re.findall('post-ACN', file):
                                        second_level.append('post-ACN')
                                        third_level.append('BEFORE')
                                        print(second_level[-1], third_level[-1])
                                    elif re.findall('doublecheck', file):
                                        continue
                                    elif re.findall('redo', file.lower()):
                                        continue
                                    # Solo case takes care of this.
                                    # else:
                                    #     second_level.append('no pair found')
                                    #     third_level.append('BEFORE')
                                    #     print(second_level[-1], third_level[-1])
                                    foreveralone = False
                                    break
                        if foreveralone == True:
                            second_level.append('no pair found')
                            third_level.append('SOLO')
                            print(second_level[-1], third_level[-1])
        
        print('\nTotal unique devices:', len(set(first_level)))
        
        column_names = ['Length', 'Resistance']
        
        row_index = [first_level, second_level, third_level]

        sanity_check = pd.DataFrame({'Device': temp_filenames, 'Voltage': second_level, 'Condition': third_level})
        sanity_check.to_csv('sanity_check_OC.csv', index = None)

        tuples = list(zip(*row_index))
        
        index = pd.MultiIndex.from_tuples(tuples) # names=['Device', 'Voltage', 'Condition']
        OC_master = pd.DataFrame(np.zeros((len(tuples), 2)), index = index, columns = column_names).sort_index(level=0)

        return OC_master
            
    def current_row(self, file_list, filename):
        filename = re.sub(r'/', '_',filename)

        # Identify correct row
        foreveralone = True
        # This section deals with the afters
        self.device_name = re.findall('\S+\s\S+\s\S+\s\S+', filename)[0]
        if re.findall("\+-1V", filename):
            if re.findall('redo', filename.lower()):
                self.file_voltage = 'REDO'
                self.file_condition = 'AFTER'
            elif re.findall('doublecheck', filename):
                self.file_voltage = 'DOUBLECHECK'
                self.file_condition = 'AFTER'
            elif re.findall('ONLY', filename):
                self.file_voltage = re.findall("\S\d+V", filename)[0][:-1]
                self.file_condition = 'ONLY'
            elif re.findall('x2extra', filename):
                self.file_voltage = re.findall("\S\d+V", filename)[0][:-1]
                self.file_condition = 'CYCLE'
            else:
                self.file_voltage = re.findall("\S\d+V", filename)[0][:-1]
                self.file_condition = 'AFTER'
        elif re.findall("\S\d+\.\d+V", filename):
            if re.findall('redo', filename.lower()):
                self.file_voltage = 'REDO'
                self.file_condition = 'AFTER'
            elif re.findall('doublecheck', filename):
                self.file_voltage = 'DOUBLECHECK'
                self.file_condition = 'AFTER'
            elif re.findall('ONLY', filename):
                self.file_voltage = re.findall("\S\d+\.\d+", filename)[0]
                self.file_condition = 'AFTER_ONLY'
            elif re.findall('x2extra', filename):
                self.file_voltage = re.findall("\S\d+\.\d+", filename)[0]
                self.file_condition = 'CYCLE'
            else:
                self.file_voltage = re.findall("\S\d+\.\d+", filename)[0]
                self.file_condition = 'AFTER'
        elif re.findall('post-ACN', filename):
            self.file_voltage = 'post-ACN'
            self.file_condition = 'AFTER'
        elif re.findall('F4TCNQ', filename):
            if re.findall('redo', filename.lower()):
                self.file_voltage = 'REDO'
                self.file_condition = 'CONC'
            elif re.findall('doublecheck', filename):
                self.file_voltage = 'DOUBLECHECK'
                self.file_condition = 'CONC'
            else:
                self.file_voltage = re.findall('F4TCNQ_\S+', filename)[0][7:]
                self.file_condition = 'CONC'
        else:
            if re.findall('doublecheck', filename):
                self.file_voltage = 'DOUBLECHECK'
                self.file_condition = 'BEFORE'
            elif re.findall('redo', filename.lower()):
                self.file_voltage = 'REDO'
                self.file_condition = 'BEFORE'
            else:
                before_name = re.findall('\S+\s\S+\s\S+\s\S+', filename)[0]
                for file in file_list:
                    file = re.sub(r'/', '_',file)
                    if file.endswith('OC.csv'):
                        if re.findall('\S+\s\S+\s\S+\s\S+', file)[0] == before_name and file != filename:
                            if re.findall("\+-1V", file):
                                self.file_voltage = re.findall("\S\d+V", file)[0][:-1]
                                if re.findall('ONLY', file):
                                    self.file_condition = 'BEFORE_ONLY'
                                else:
                                    self.file_condition = 'BEFORE'
                            elif re.findall("\S\d+\.\d+V", file):
                                self.file_voltage = re.findall("\S\d+\.\d+", file)[0]
                                if re.findall('ONLY', file):
                                    self.file_condition = 'BEFORE_ONLY'
                                else:
                                    self.file_condition = 'BEFORE'
                            elif re.findall('post-ACN', file):
                                self.file_voltage = 'post-ACN'
                                self.file_condition = 'BEFORE'
                            elif re.findall('doublecheck', file):
                                continue
                            elif re.findall('redo', file.lower()):
                                continue
                            # Solo case takes care of this.
                            # else:
                            #     self.file_voltage = 'no pair found'
                            #     self.file_condition = 'BEFORE'
                            foreveralone = False
                            break
                if foreveralone == True:
                    self.file_voltage = 'no pair found'
                    self.file_condition = 'SOLO'

        return self.device_name, self.file_voltage, self.file_condition

    def generate_r2_lines(self, data_folder, data, plot_id, oc_master_RcontactFit):
        # Filter data
        if plot_id == 'voltage':
            df = data.iloc[(data.index.get_level_values(2) == 'BEFORE') | (data.index.get_level_values(2) == 'AFTER')]
            df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']

        elif plot_id == 'only':
            df = data.iloc[(data.index.get_level_values(2) == 'BEFORE_ONLY') | (data.index.get_level_values(2) == 'AFTER_ONLY')]
            df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']

        elif plot_id == 'post-ACN':
            df = data.iloc[data.index.get_level_values(1) == 'post-ACN']

        elif plot_id == 'conc':
            df = data.iloc[data.index.get_level_values(2) == 'CONC']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']

        elif plot_id == 'cycle':
            temp = data.iloc[data.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
            test = pd.DataFrame()
            for i in temp:
                test = test.append(data.iloc[data.index.get_level_values(0) == i])
            
            df = test
            df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']

        print('Total data for this subset:', df.shape)

        # Do the plots
        for device_name in df.index.get_level_values(0).unique():
            temp = df.loc[df.index.get_level_values(0) == device_name]

            for device_voltage in temp.index.get_level_values(1).unique():
                temp2 = temp.loc[temp.index.get_level_values(1) == device_voltage].sort_values(by=['Length'])
                
                for device_condition in temp2.index.get_level_values(2).unique():
                    temp3 = temp2.loc[temp2.index.get_level_values(2) == device_condition]
                    if temp3.shape[0] < 3:
                        print('\n{}_{}_{} does not have enough data'.format(device_name, device_voltage, device_condition))
                    else:
                        x = temp3['Length'].tolist()
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, temp3['Resistance'].tolist())
                        intercept_error = std_err * math.sqrt((1/len(x)) * np.sum(np.square(x)))
                        x_line = np.arange(0, temp3['Length'].tolist()[-1] + 10,10)
                        y_line = (np.array(x_line) * slope + intercept).tolist()

                        plt.figure(figsize=(14,10))
                        # Main
                        sc = plt.scatter(temp3['Length'].tolist(), temp3['Resistance'].tolist(), edgecolors='r', lw=2, marker='o', label = 'Before')
                        sc.set_facecolor("none")
                        # fitted
                        plt.plot(x_line, y_line, c='B', label = 'FITTED')
                        min_min = 0 if min([min(y_line), min(temp3['Resistance'])]) > 0 else min([min(y_line), min(temp3['Resistance'])]) + min([min(y_line), min(temp3['Resistance'])])
                        plt.ylim([min_min, max(temp3['Resistance']) + (max(temp3['Resistance']) * 0.2)])
                        plt.xlabel('Channel Length (nm)')
                        plt.ylabel(r'$Resistance_{measured}\ (\Omega)$')
                        plt.legend()

                        plt.savefig('{}/Results/RcontactFit_{}_{}_{}_{}.png'.format(data_folder, re.findall('\S+\s\S+\s\S+', device_name)[0], device_voltage, device_condition, plot_id), bbox_inches="tight")
                        plt.savefig('{}/Results/SVG_plots/RcontactFit_{}_{}_{}_{}.svg'.format(data_folder, re.findall('\S+\s\S+\s\S+', device_name)[0], device_voltage, device_condition, plot_id), bbox_inches="tight")

                        plt.gcf().clear()

                        oc_master_RcontactFit['device_name'].append(device_name)
                        oc_master_RcontactFit['device_voltage'].append(device_voltage)
                        oc_master_RcontactFit['device_condition'].append(device_condition)
                        oc_master_RcontactFit['intercept'].append(intercept/2)
                        oc_master_RcontactFit['r_value'].append(r_value**2)
                        oc_master_RcontactFit['std_err'].append(intercept_error)

        return oc_master_RcontactFit

    def plot_oc_master(data, data_folder, plot_id):
        # Filter data
        if plot_id == 'voltage':
            df = data.iloc[(data.index.get_level_values(2) == 'BEFORE') | (data.index.get_level_values(2) == 'AFTER')]
            df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']
            df = np.multiply(df, 1000000000)

            # Separete data
            before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
            before.reset_index(level=2, drop=True, inplace=True)
            after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
            after.reset_index(level=2, drop=True, inplace=True)

            # Delta plots
            delta = after - before
            delta['R_Contact_ERROR'] = np.sqrt(np.square(after['R_Contact_ERROR']) + np.square(before['R_Contact_ERROR']))
            delta_percentage = (after - before) / np.absolute(before)
            delta_percentage['R_Contact_ERROR'] = abs(delta_percentage['R_Contact']) * np.sqrt(np.square(delta['R_Contact_ERROR']/delta['R_Contact']) + np.square(before['R_Contact_ERROR']/before['R_Contact']))
            delta_percentage = np.multiply(delta_percentage, 100)

            plt.figure(figsize=(14,10))

            plt.errorbar(before.index.get_level_values(1).values, before['R_Contact'].values, yerr=before['R_Contact_ERROR'].values, fmt='o', color='b', label = 'BEFORE')
            plt.errorbar(after.index.get_level_values(1).values, after['R_Contact'].values, yerr=after['R_Contact_ERROR'].values, fmt='o', color='r', label = 'AFTER')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'Contact Resistance, $R_C$ (G$\Omega)$')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig('{}/Results/Master_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Master_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Plot delta
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta.index.get_level_values(1).values, delta['R_Contact'], yerr=delta['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (G$\Omega)$')
            plt.savefig('{}/Results/Delta_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Delta percentage
            
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta_percentage.index.get_level_values(1).values, delta_percentage['R_Contact'], yerr=delta_percentage['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (%)')
            plt.savefig('{}/Results/Delta_percentage_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_percentage_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

        elif plot_id == 'only':
            df = data.iloc[(data.index.get_level_values(2) == 'BEFORE_ONLY') | (data.index.get_level_values(2) == 'AFTER_ONLY')]
            df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']
            df = np.multiply(df, 1000000000)

            # Separete data
            before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', :]
            before.reset_index(level=2, drop=True, inplace=True)
            after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', :]
            after.reset_index(level=2, drop=True, inplace=True)

            # Delta plots
            delta = after - before
            delta['R_Contact_ERROR'] = np.sqrt(np.square(after['R_Contact_ERROR']) + np.square(before['R_Contact_ERROR']))
            delta_percentage = (after - before) / np.absolute(before)
            delta_percentage['R_Contact_ERROR'] = abs(delta_percentage['R_Contact']) * np.sqrt(np.square(delta['R_Contact_ERROR']/delta['R_Contact']) + np.square(before['R_Contact_ERROR']/before['R_Contact']))
            delta_percentage = np.multiply(delta_percentage, 100)

            plt.figure(figsize=(14,10))

            plt.errorbar(before.index.get_level_values(1).values, before['R_Contact'].values, yerr=before['R_Contact_ERROR'].values, fmt='o', color='b', label = 'BEFORE')
            plt.errorbar(after.index.get_level_values(1).values, after['R_Contact'].values, yerr=after['R_Contact_ERROR'].values, fmt='o', color='r', label = 'AFTER')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'Contact Resistance, $R_C$ (G$\Omega)$')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig('{}/Results/Master_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Master_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Plot delta
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta.index.get_level_values(1).values, delta['R_Contact'], yerr=delta['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (G$\Omega)$')
            plt.savefig('{}/Results/Delta_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Delta percentage
            
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta_percentage.index.get_level_values(1).values, delta_percentage['R_Contact'], yerr=delta_percentage['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (%)')
            plt.savefig('{}/Results/Delta_percentage_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_percentage_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

        elif plot_id == 'post-ACN':
            df = data.iloc[data.index.get_level_values(1) == 'post-ACN']
            df = np.multiply(df, 1000000000)

            # Separete data
            before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
            before.reset_index(level=2, drop=True, inplace=True)
            after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
            after.reset_index(level=2, drop=True, inplace=True)

            # Delta plots
            delta = after - before
            delta['R_Contact_ERROR'] = np.sqrt(np.square(after['R_Contact_ERROR']) + np.square(before['R_Contact_ERROR']))
            delta_percentage = (after - before) / np.absolute(before)
            delta_percentage['R_Contact_ERROR'] = abs(delta_percentage['R_Contact']) * np.sqrt(np.square(delta['R_Contact_ERROR']/delta['R_Contact']) + np.square(before['R_Contact_ERROR']/before['R_Contact']))
            delta_percentage = np.multiply(delta_percentage, 100)

            plt.figure(figsize=(14,10))

            plt.errorbar(before.index.get_level_values(0).values, before['R_Contact'].values, yerr=before['R_Contact_ERROR'].values, fmt='o', color='b', label = 'BEFORE')
            plt.errorbar(after.index.get_level_values(0).values, after['R_Contact'].values, yerr=after['R_Contact_ERROR'].values, fmt='o', color='r', label = 'AFTER')
            plt.xlabel('Device name')
            plt.xticks(rotation=45)
            plt.ylabel(r'Contact Resistance, $R_C$ (G$\Omega)$')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig('{}/Results/Master_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Master_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Plot delta
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta.index.get_level_values(0).values, delta['R_Contact'], yerr=delta['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Device name')
            plt.xticks(rotation=45)
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (G$\Omega)$')
            plt.savefig('{}/Results/Delta_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

            # Delta percentage
            
            plt.figure(figsize=(14,10))
            
            plt.errorbar(delta_percentage.index.get_level_values(0).values, delta_percentage['R_Contact'], yerr=delta_percentage['R_Contact_ERROR'], fmt='o', color='b')
            plt.xlabel('Device name')
            plt.xticks(rotation=45)
            plt.ylabel(r'$\Delta$ Contact Resistance, $R_C$ (%)')
            plt.savefig('{}/Results/Delta_percentage_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Delta_percentage_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

        elif plot_id == 'conc':
            df = data.iloc[data.index.get_level_values(2) == 'CONC']
            df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
            df = df.iloc[df.index.get_level_values(1) != 'REDO']
            df = np.multiply(df, 1000000000)

            plt.figure(figsize=(14,10))

            plt.errorbar(df.index.get_level_values(1).values.astype(float), df['R_Contact'].values, yerr=df['R_Contact_ERROR'].values, fmt='o', color='b')
            plt.xlabel('F4TCNQ Concentration (mg/mL)')
            plt.ylabel(r'Contact Resistance, $R_C$ (G$\Omega)$')
            plt.savefig('{}/Results/Master_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Master_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

        elif plot_id == 'cycle':
            temp = data.iloc[data.index.get_level_values(2) == 'CYCLE'].index.get_level_values(0).tolist()
            test = pd.DataFrame()
            for i in temp:
                test = test.append(data.iloc[data.index.get_level_values(0) == i])
            
            df = test
            df = np.multiply(df, 1000000000)

            # Separete data
            before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
            before.reset_index(level=2, drop=True, inplace=True)
            after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
            after.reset_index(level=2, drop=True, inplace=True)
            cycle = df.iloc[df.index.get_level_values(2) == 'CYCLE', :]
            cycle.reset_index(level=2, drop=True, inplace=True)

            plt.figure(figsize=(14,10))

            plt.errorbar(before.index.get_level_values(1).values, before['R_Contact'].values, yerr=before['R_Contact_ERROR'].values, fmt='o', color='b', label = 'BEFORE')
            plt.errorbar(after.index.get_level_values(1).values, after['R_Contact'].values, yerr=after['R_Contact_ERROR'].values, fmt='o', color='g', label = 'AFTER')
            plt.errorbar(cycle.index.get_level_values(1).values, cycle['R_Contact'].values, yerr=cycle['R_Contact_ERROR'].values, fmt='o', color='r', label = '2X DOPED')
            plt.xlabel('Voltage applied (V)')
            plt.ylabel(r'Contact Resistance, $R_C$ (G$\Omega)$')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig('{}/Results/Master_RcontactFit_plot_{}.png'.format(data_folder, plot_id), bbox_inches="tight")
            plt.savefig('{}/Results/SVG_plots/Master_RcontactFit_plot_{}.svg'.format(data_folder, plot_id), bbox_inches="tight")

            plt.gcf().clear()

        
