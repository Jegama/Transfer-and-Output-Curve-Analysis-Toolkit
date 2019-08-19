import numpy as np
import pandas as pd
import seaborn as sns
import os, re, math
from scipy import stats

sns.set_style('whitegrid')
sns.set_context('talk')
        
def create_intercept_line(x, slope, intercept):
    return (np.array(x) * slope + intercept).tolist()
    # return [slope * i + intercept for i in x]


class tc_analysis:
    def __init__(self):
        self.Ci = 1.72575*(10**(-8))

    def get_l_and_w(self, filename):
        self.file_l = int(re.sub('\D', '', re.findall('L\d+W', filename)[0]))
        self.file_w = int(re.sub('\D', '', re.findall('W\d+', filename)[0]))

    def generate_empty_dataframes(self, file_list):
        # Go through all the files and get the main DataFrame to analyze TC
        first_level = []
        second_level = []
        third_level = []
        temp_filenames = []

        for filename in file_list:
            filename = re.sub(r'/', '_',filename)
            if filename.endswith('TC.csv'):
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
                            if file.endswith('TC.csv'):
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
                                    elif re.findall('doublecheck', file.lower()):
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
        
        column_names = [('Linear (ID)', 'Vd1', 'FWD'),
                        ('Linear (ID)', 'Vd1', 'REV'), 
                        ('Linear (ID)', 'Vd2', 'FWD'), 
                        ('Linear (ID)', 'Vd2', 'REV'),
                        ('Saturation (ID)', 'Vd(-2)', 'FWD'), 
                        ('Saturation (ID)', 'Vd(-2)', 'REV'), 
                        ('Saturation (ID)', 'Vd(-1)', 'FWD'), 
                        ('Saturation (ID)', 'Vd(-1)', 'REV')]
        
        row_index = [first_level, second_level, third_level]

        sanity_check = pd.DataFrame({'Device': temp_filenames, 'Voltage': second_level, 'Condition': third_level})
        sanity_check.to_csv('sanity_check.csv', index = None)

        tuples = list(zip(*row_index))
        
        index = pd.MultiIndex.from_tuples(tuples) # names=['Device', 'Voltage', 'Condition']
        column_index = pd.MultiIndex.from_tuples(column_names) # names=['Region', 'Vd', 'Sweep_dir']
        TC_master = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        
        self.TC_slope = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_intercept = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_mobility = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_reliability = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_mu_eff = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)

        self.TC_min_id = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_id_at_0 = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_max_id = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)
        self.TC_ratio_id = pd.DataFrame(np.zeros((len(tuples), 8)), index = index, columns = column_index).sort_index(level=0)

        # Hysteresis

        column_names_Hysteresis = [('Linear (ID)', 'Vd1'),
                                    ('Linear (ID)', 'Vd2'), 
                                    ('Saturation (ID)', 'Vd(-2)'), 
                                    ('Saturation (ID)', 'Vd(-1)')]

        column_index_Hysteresis = pd.MultiIndex.from_tuples(column_names_Hysteresis) # names=['Region', 'Vd']

        self.max_Hysteresis = pd.DataFrame(np.zeros((len(tuples), 4)), index = index, columns = column_index_Hysteresis).sort_index(level=0)

        return TC_master
            
    def separate_data(self, data_folder, filename):
        main = pd.read_csv(os.path.join(data_folder,filename))
            
        # Get parameters for TC legend (VDs)
        temp_tc = pd.read_csv(os.path.join(data_folder,filename[:-4] + '-s.csv'), error_bad_lines=False)
        columns_TC = temp_tc.columns
        start = float(temp_tc.loc[temp_tc[columns_TC[0]] == 'Measurement.Secondary.Start', columns_TC[1]])
        count = float(temp_tc.loc[temp_tc[columns_TC[0]] == 'Measurement.Secondary.Count', columns_TC[1]])
        step = float(temp_tc.loc[temp_tc[columns_TC[0]] == 'Measurement.Secondary.Step', columns_TC[1]])
        divisions = main.shape[0] / count
        
        # Add VD to the dataframe
        self.VD = []
        VD_values = start
        for i in range(int(count)):
            self.VD.extend([VD_values]*int(divisions))
            VD_values += step
            
        main['VD'] = self.VD
        
        # Rectify sing
        main.loc[main['ID']<0, 'sqrtID'] *= -1
        
        self.main0 = main.loc[main['VD'] == 0]
        
        # Get VG for the x axis
        self.df_TC = pd.DataFrame(index = self.main0['VG'])
        self.df_TC_sqrt = pd.DataFrame(index = self.main0['VG'])
        self.df_TC_lines = pd.DataFrame(index = self.main0['VG'])
        self.df_TC_sqrt_lines = pd.DataFrame(index = self.main0['VG'])
        
        # Separete ID for plotting by VG into columns
        for n in sorted(list(set(self.VD))):
            self.df_TC[n] = list(main.loc[main['VD'] == n, 'ID'])
            self.df_TC_sqrt[n] = list(main.loc[main['VD'] == n, 'sqrtID'])
            self.df_TC_lines[n] = list(main.loc[main['VD'] == n, 'ID'])
            self.df_TC_sqrt_lines[n] = list(main.loc[main['VD'] == n, 'sqrtID'])

        return self.main0, self.VD, self.df_TC, self.df_TC_sqrt

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
                    if file.endswith('TC.csv'):
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
                            elif re.findall('doublecheck', file.lower()):
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

    def update_lines(self, intercept_line_column, middle_point, raw, slope, intercept, data_type, direction):
        if data_type:
            self.df_TC_lines[intercept_line_column] = 0
            if direction == 'FWD':
                self.df_TC_lines[intercept_line_column][:middle_point] = create_intercept_line(raw, slope, intercept)
            else:
                self.df_TC_lines[intercept_line_column][middle_point:] = create_intercept_line(raw, slope, intercept)
            self.df_TC_lines.loc[self.df_TC_lines[intercept_line_column]>0, intercept_line_column] = 0
        else:
            self.df_TC_sqrt_lines[intercept_line_column] = 0
            if direction == 'FWD':
                self.df_TC_sqrt_lines[intercept_line_column][:middle_point] = create_intercept_line(raw, slope, intercept)
            else:
                self.df_TC_sqrt_lines[intercept_line_column][middle_point:] = create_intercept_line(raw, slope, intercept)
            self.df_TC_sqrt_lines.loc[self.df_TC_sqrt_lines[intercept_line_column]>0, intercept_line_column] = 0

    def remove_zeros_from_lines(self):
        self.df_TC_lines.replace(0, np.nan, inplace=True)
        self.df_TC_sqrt_lines.replace(0, np.nan, inplace=True)

        return self.df_TC_lines, self.df_TC_sqrt_lines

    def populate_dataframes(self, column, slope, intercept, vd_ = 0, linear = True):
        """
            - Reliability factor
            - Mobility
            - mu_eff = r(reliability factor calculated above) * m(mobility calculated above from slope)
        """
        self.TC_slope.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = slope
        if linear:
            mobility_temp = ((self.file_l / (self.file_w * self.Ci * abs(vd_))) * slope) * 0.001
            reliability_temp = ((abs(self.max_id) - abs(self.id_at_0))/abs(min(self.df_TC.index))) / slope
            self.TC_intercept.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = (-intercept / slope) - abs(vd_ / 2)
            self.TC_mobility.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = mobility_temp
            self.TC_reliability.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = reliability_temp
            self.TC_mu_eff.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = reliability_temp * mobility_temp
        else:
            mobility_temp = (((2 * self.file_l) / (self.file_w * self.Ci)) * (slope ** 2)) * 0.001
            reliability_temp = math.pow(((np.sqrt(abs(self.max_id)) - np.sqrt(abs(self.id_at_0))) / abs(min(self.df_TC.index))), 2) / math.pow(slope, 2)
            self.TC_intercept.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = -intercept / slope 
            self.TC_mobility.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = mobility_temp
            self.TC_reliability.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = reliability_temp
            self.TC_mu_eff.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = reliability_temp * mobility_temp

    def export_dataframes(self, data_folder):
        self.TC_slope.to_csv('%s/Results/Slopes.csv' % data_folder)
        self.TC_intercept.to_csv('%s/Results/Intercepts.csv' % data_folder)
        self.TC_mobility.to_csv('%s/Results/Mobility.csv' % data_folder)
        self.TC_reliability.to_csv('%s/Results/Reliability_factor.csv' % data_folder)
        self.TC_mu_eff.to_csv('%s/Results/MU_eff.csv' % data_folder)
        self.TC_min_id.to_csv('%s/Results/Min_ID.csv' % data_folder)
        self.TC_max_id.to_csv('%s/Results/Max_ID.csv' % data_folder)
        self.TC_id_at_0.to_csv('%s/Results/ID_at_VG_0.csv' % data_folder)
        self.TC_ratio_id.to_csv('%s/Results/Ratio_ID.csv' % data_folder)
        self.max_Hysteresis.to_csv('%s/Results/Max_Hysteresis.csv' % data_folder)

    def populate_min_max_ratio_files(self, df_temp, first_iloc_max, first_iloc_min, second_iloc, column, middle_point):
        '''
            Fancy files with 3 levels index and 3 level columns

            For max:
                df_TC.iloc[middle_point-1, -2] < fwd_id1
                df_TC.iloc[middle_point, -2] < rev_id1

            For min:
                df_TC.iloc[0, -2] < fwd_id1
                df_TC.iloc[-1, -2] < rev_id1

            For id_at_0:
                Use VG == 0

            For ratio:
                max/min
        '''

        if column[0] == 'Linear (ID)':
            self.max_id = df_temp.iloc[first_iloc_max, second_iloc]
            if column[2] == 'FWD':
                self.id_at_0 = df_temp[:middle_point].loc[0, df_temp.columns[second_iloc]]
            else:
                self.id_at_0 = df_temp[middle_point:].loc[0, df_temp.columns[second_iloc]]
            min_id = df_temp.iloc[first_iloc_min, second_iloc]
        else:
            self.max_id = math.pow(df_temp.iloc[first_iloc_max, second_iloc], 2) * (-1)
            if column[2] == 'FWD':
                self.id_at_0 = math.pow(df_temp[:middle_point].loc[0, df_temp.columns[second_iloc]], 2) * (-1)
            else:
                self.id_at_0 = math.pow(df_temp[middle_point:].loc[0, df_temp.columns[second_iloc]], 2) * (-1)
            min_id = math.pow(df_temp.iloc[first_iloc_min, second_iloc], 2) * (-1)
        ratio_id = abs(self.max_id/min_id)

        self.TC_max_id.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = self.max_id
        self.TC_id_at_0.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = self.id_at_0
        self.TC_min_id.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = min_id
        self.TC_ratio_id.loc[self.device_name, self.file_voltage, self.file_condition][column[0], column[1], column[2]] = ratio_id

    def generate_hysteresis(self, data_folder, filename, middle_point):
        """
            Hysteresis of transfer curves = abs(Id_rev - Id_fwd)
            abs(np.subtract(reverse, forward))
        """

        filename = re.sub(r'/', '_',filename)

        temp_hysteresis = pd.DataFrame(index = self.main0['VG'][:middle_point])

        temp_hysteresis[self.df_TC.columns[-2]] = abs(np.subtract(list(self.df_TC[self.df_TC.columns[-2]][middle_point:])[::-1], list(self.df_TC[self.df_TC.columns[-2]][:middle_point])))
        self.max_Hysteresis.loc[self.device_name, self.file_voltage, self.file_condition]['Linear (ID)', 'Vd1'] = max(temp_hysteresis[self.df_TC.columns[-2]])

        temp_hysteresis[self.df_TC.columns[-3]] = abs(np.subtract(list(self.df_TC[self.df_TC.columns[-3]][middle_point:])[::-1], list(self.df_TC[self.df_TC.columns[-3]][:middle_point])))
        self.max_Hysteresis.loc[self.device_name, self.file_voltage, self.file_condition]['Linear (ID)', 'Vd2'] = max(temp_hysteresis[self.df_TC.columns[-3]])

        temp_hysteresis[self.df_TC.columns[1]] = abs(np.subtract(list(self.df_TC[self.df_TC.columns[1]][middle_point:])[::-1], list(self.df_TC[self.df_TC.columns[1]][:middle_point])))
        self.max_Hysteresis.loc[self.device_name, self.file_voltage, self.file_condition]['Saturation (ID)', 'Vd(-2)'] = max(temp_hysteresis[self.df_TC.columns[1]])

        temp_hysteresis[self.df_TC.columns[0]] = abs(np.subtract(list(self.df_TC[self.df_TC.columns[0]][middle_point:])[::-1], list(self.df_TC[self.df_TC.columns[0]][:middle_point])))
        self.max_Hysteresis.loc[self.device_name, self.file_voltage, self.file_condition]['Saturation (ID)', 'Vd(-1)'] = max(temp_hysteresis[self.df_TC.columns[0]])

        temp_hysteresis.to_csv('{}/Results/Hysteresis/{}_Hysteresis.csv'.format(data_folder, filename[:-4]))

        # Generate plot for linear ID
        pltTC = temp_hysteresis.plot(figsize=(14,10), colormap = 'inferno')
        pltTC.set_ylabel('ID (A)')
        pltTC.set_xlabel('VG (V)')
        
        # Invert plot to first quadrant 1
        pltTC.invert_xaxis()
        
        # Save plots
        linear_tc = pltTC.get_figure();
        linear_tc.savefig(os.path.join(data_folder, 'Results/Hysteresis',filename[:-4] + '.png'))


    def generate_mobility_curves(self, data_folder, filename, vg_temp, id_temp, direction, vd_ = 0, linear = True):
        '''
            Generates mobility curve for entire VG range (x-axis)
            from TC file.
            
            Outputs 8 plots, one for each:

            fwd_id1
            rev_id1
            fwd_id2
            rev_id2
            fwd_idminus2
            rev_idminus2
            fwd_idminus1
            rev_idminus1
        '''

        filename = re.sub(r'/', '_',filename)

        temp_TC = pd.DataFrame(index = self.main0['VG']).iloc[5:-1]
        temp_mobility = []

        for i in range(5, len(vg_temp)):
            slope, intercept, r_value, p_value, std_err = stats.linregress(vg_temp[i-5:i], id_temp[i-5:i])
            if linear:
                temp_mobility.append(((self.file_l / (self.file_w * self.Ci * abs(vd_))) * slope) * 0.001)
                type_ = 'lin'
            else:
                temp_mobility.append((((2 * self.file_l) / (self.file_w * self.Ci)) * (slope ** 2)) * 0.001)
                type_ = 'sat'

        df_to_plot = pd.DataFrame({
            'VG': vg_temp[5:],
            'Mobility': temp_mobility
        }).set_index('VG')

        pltTC = df_to_plot.plot(figsize=(14,10), colormap = 'inferno')
        pltTC.set_ylabel('Mobility')
        pltTC.set_xlabel('VG (V)')

        # pltTC.invert_yaxis()
        pltTC.invert_xaxis()

        linear_tc = pltTC.get_figure();
        linear_tc.savefig(os.path.join(data_folder, 'Results/Mobility_curves', filename[:-4] + type_ + ' Vd_' + str(vd_) + ' ' + direction + ' mu.png'))


    def generate_avg_df(self, data_folder, data_name):
        print('Generating avg_df for %s' % data_name)
        if data_name == 'Max_Hysteresis':
            df_temp = pd.read_csv('{}/Results/{}.csv'.format(data_folder, data_name), index_col=[0,1,2], header=[0,1]).sort_index(level=0)
        else:
            df_temp = pd.read_csv('{}/Results/{}.csv'.format(data_folder, data_name), index_col=[0,1,2], header=[0,1,2]).sort_index(level=0)

        # Voltage

        df = df_temp.iloc[(df_temp.index.get_level_values(2) == 'BEFORE') | (df_temp.index.get_level_values(2) == 'AFTER')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu.dropna(inplace = True)
        delta_mu_percentage = (after - before) / np.absolute(before)
        delta_mu_percentage.dropna(inplace = True)

        lin_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1']
        lin_vd1.columns = lin_vd1.columns.droplevel(1)
        lin_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2']
        lin_vd2.columns = lin_vd2.columns.droplevel(1)
        lin_master = pd.concat([lin_vd1, lin_vd2])
        sat_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)']
        sat_vd1.columns = sat_vd1.columns.droplevel(1)
        sat_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)']
        sat_vd2.columns = sat_vd2.columns.droplevel(1)
        sat_master = pd.concat([sat_vd1, sat_vd2])
        master_voltage = pd.concat([lin_master, sat_master], axis = 1)

        master_voltage = master_voltage.groupby(master_voltage.index.get_level_values(1)).describe()
        try:
            master_voltage.loc[:, (master_voltage.columns.get_level_values(2) == 'mean') | (master_voltage.columns.get_level_values(2) == 'std')].to_csv('{}Results/{}_avg_master_voltage.csv'.format(data_folder, data_name))
        except IndexError:
            master_voltage.loc[:, (master_voltage.columns.get_level_values(1) == 'mean') | (master_voltage.columns.get_level_values(1) == 'std')].to_csv('{}Results/{}_avg_master_voltage.csv'.format(data_folder, data_name))

        # Only

        df = df_temp.iloc[(df_temp.index.get_level_values(2) == 'BEFORE_ONLY') | (df_temp.index.get_level_values(2) == 'AFTER_ONLY')]
        df = df.iloc[df.index.get_level_values(1) != 'post-ACN']
        df = df.iloc[df.index.get_level_values(1) != 'DOUBLECHECK']
        df = df.iloc[df.index.get_level_values(1) != 'REDO']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE_ONLY', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER_ONLY', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu.dropna(inplace = True)
        delta_mu_percentage = (after - before) / np.absolute(before)
        delta_mu_percentage.dropna(inplace = True)

        lin_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1']
        lin_vd1.columns = lin_vd1.columns.droplevel(1)
        lin_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2']
        lin_vd2.columns = lin_vd2.columns.droplevel(1)
        lin_master = pd.concat([lin_vd1, lin_vd2])
        sat_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)']
        sat_vd1.columns = sat_vd1.columns.droplevel(1)
        sat_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)']
        sat_vd2.columns = sat_vd2.columns.droplevel(1)
        sat_master = pd.concat([sat_vd1, sat_vd2])
        master_only = pd.concat([lin_master, sat_master], axis = 1)

        master_only = master_only.groupby(master_only.index.get_level_values(1)).describe()
        try:
            master_only.loc[:, (master_only.columns.get_level_values(2) == 'mean') | (master_only.columns.get_level_values(2) == 'std')].to_csv('{}Results/{}_avg_master_only.csv'.format(data_folder, data_name))
        except IndexError:
            master_only.loc[:, (master_only.columns.get_level_values(1) == 'mean') | (master_only.columns.get_level_values(1) == 'std')].to_csv('{}Results/{}_avg_master_only.csv'.format(data_folder, data_name))

        # Post-ACN

        df = df_temp.iloc[df_temp.index.get_level_values(1) == 'post-ACN']

        before = df.iloc[df.index.get_level_values(2) == 'BEFORE', :]
        before.reset_index(level=2, drop=True, inplace=True)
        after = df.iloc[df.index.get_level_values(2) == 'AFTER', :]
        after.reset_index(level=2, drop=True, inplace=True)
        delta_mu = after - before
        delta_mu.dropna(inplace = True)
        delta_mu_percentage = (after - before) / np.absolute(before)
        delta_mu_percentage.dropna(inplace = True)

        lin_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd1']
        lin_vd1.columns = lin_vd1.columns.droplevel(1)
        lin_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd2']
        lin_vd2.columns = lin_vd2.columns.droplevel(1)
        lin_master = pd.concat([lin_vd1, lin_vd2])
        sat_vd1 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-2)']
        sat_vd1.columns = sat_vd1.columns.droplevel(1)
        sat_vd2 = delta_mu.iloc[:, delta_mu.columns.get_level_values(1) == 'Vd(-1)']
        sat_vd2.columns = sat_vd2.columns.droplevel(1)
        sat_master = pd.concat([sat_vd1, sat_vd2])
        master_post_ACN = pd.concat([lin_master, sat_master], axis = 1)

        master_post_ACN = master_post_ACN.groupby(master_post_ACN.index.get_level_values(1)).describe()
        try:
            master_post_ACN.loc[:, (master_post_ACN.columns.get_level_values(2) == 'mean') | (master_post_ACN.columns.get_level_values(2) == 'std')].to_csv('{}Results/{}_avg_master_post-ACN.csv'.format(data_folder, data_name))
        except IndexError:
            master_post_ACN.loc[:, (master_post_ACN.columns.get_level_values(1) == 'mean') | (master_post_ACN.columns.get_level_values(1) == 'std')].to_csv('{}Results/{}_avg_master_post-ACN.csv'.format(data_folder, data_name))