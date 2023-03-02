
from integration_2_copy import *

"""
file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/1m_2/1m_2_10.csv')
"""

"""
file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/3m_2/3m_2_10.csv')
"""


#"""
file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_10.csv')
#"""

########################################################################################
########################################################################################

"""
###### -> NOTE: PROCESSED SIGNAL IS CORRECTED_SIGNAL (POLYNOMIAL APPROXIMATION)

Functionality: Do the signal processing to all the files and get the distance from the
target according to all csv files.
Input: 
Output: Distance from the target according to the different frequency methods. It
presents the distance from each range period of the channelB and the average of each
signals.
"""

l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]
print(l_filename)

#"""
def procesado_conjunto_Average(l_filename):
    l_distances_aux = []
    l_distances_aux_average = []
    l_distances_aux_CF = []
    l_distances_aux_average_CF = []
    
    #l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]

    Time_aux = []
    channelB_aux = []
    corrected_aux = []
    corrected_aux_Sig = []
    
    period_time = []
    fiit_R = []
    
    for i in range(len(l_filename)):
        ######Call functions
        (Time, channelA, channelB) = read_csv(l_filename[i])
        Time_aux.append(Time)
        channelB_aux.append(channelB)
        
        samples = samples_IntervalTime(Time)

        peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)

        peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)

        interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

        (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)
        period_time.append(periods_Time)

        (fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)
        fiit_R.append(fiit_real)

        (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
        #(corrected_Sig, model_full_ran, polyline_full_ran) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
        #corrected_aux_Sig.append(corrected_Sig)
        #(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)
        corrected_aux.append(corrected_Signals_periods)

        size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)

        (output_def_periods, x_union) = interpolationSignal_where_crossZero(corrected_Signals_periods)

        (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, corrected_Signals_periods, periods_Time, x_union)
    
        output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)

        (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

        frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

        (freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, corrected_Signals_periods)

        fft_method(periods_channelB, samples, valor)

        (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

        l_distances_aux.append(l_distance_periods) #ALL THREE PERIODS
        l_distances_aux_average.append(average_distance) #AVERAGE

        l_distances_aux_CF.append(l_distance_CF_periods)
        l_distances_aux_average_CF.append(average_CF_distance)

    l_std_deviation = []
    l_std_deviation_average = []

    for r in range(len(l_distances_aux)):
        l_std_deviation.append(np.std(l_distances_aux[r]))
        print("Distance with Zero Crossing (Average) - file " + str(r+1) + " in the first period is: " + str(l_distances_aux[r][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux[r][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux[r][2]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
        
    for l in range(len(l_std_deviation)):
        print("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation[l]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    for m in range(len(l_distances_aux_average)):
        print("Distance with Zero Crossing (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_distances_aux_average_ZC_allt = []
    l_distances_aux_average_ZC_allt = statistics.mean(l_distances_aux_average)
    print("Aaverage of distances with Zero Crossing method: " + str(l_distances_aux_average_ZC_allt))

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_std_deviation_average.append(np.std(l_distances_aux_average))
    print("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average) + "\n")

    print("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---")

    l_std_deviation_CF = []
    l_std_deviation_average_CF = []

    for f in range(len(l_distances_aux_CF)):
        l_std_deviation_CF.append(np.std(l_distances_aux_CF[f]))
        print("Distance with Curve Fitting (Average) - file " + str(f+1) + " in the first period is: " + str(l_distances_aux_CF[f][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux_CF[f][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux_CF[f][2]) + "\n")
    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
        
    for l in range(len(l_std_deviation_CF)):
        print("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation_CF[l]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    for m in range(len(l_distances_aux_average_CF)):
        print("Distance with Curve Fitting (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average_CF[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_distances_aux_average_CF_allt = []
    l_distances_aux_average_CF_allt = statistics.mean(l_distances_aux_average_CF)
    print("Average of distances with Curve Fitting method: " + str(l_distances_aux_average_CF_allt))

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_std_deviation_average_CF.append(np.std(l_distances_aux_average_CF))
    print("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average_CF) + "\n")
    
    
    representation_Figures(periods_Time, periods_channelA, periods_channelB,
                       Time, channelA, channelB,
                       corrected_Signals_periods, model_full_range, polyline_full_range,
                       valor,
                       peaks_Time_B, peaks_channelB,
                       #peaks_Time_B_PC, peaks_channelB_1m_PC,
                       #output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis,
                       #output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis,
                       #output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC,
                       popt, tt)

    return(Time_aux, channelB_aux, corrected_aux, period_time, fiit_R, corrected_aux_Sig)
   
#"""

"""
###### -> NOTE: PROCESSED SIGNAL IS CORRECTED_SIGNAL (POLYNOMIAL APPROXIMATION)

Functionality: Do the signal processing to the file selected and get the distance
from the target according to the csv file.
Input: File to analyze.
Output: Distance from the target according to the different frequency methods. It
presents the distance from each range period of the channelB and the average of each
signals.
"""

#"""
def procesado_selectivo_instant_andRepresentation(file):
    ######Call functions
    (Time, channelA, channelB) = read_csv(file)

    samples = samples_IntervalTime(Time)

    peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)

    peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)

    interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

    (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)

    (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)

    size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)

    (output_def_periods, x_union) = interpolationSignal_where_crossZero(corrected_Signals_periods)

    (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, corrected_Signals_periods, periods_Time, x_union)
    
    output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)

    (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

    frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

    (freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, corrected_Signals_periods)

    fft_method(periods_channelB, samples, valor)

    (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

    l_std_deviation = []
    l_std_deviation_CF = []

    for m in range(len(l_distance_periods)):
        print("Distance with Zero Crossing for file " + str(file) + " in the " +str(m+1) +  " period is: " + str(l_distance_periods[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_std_deviation.append(np.std(l_distance_periods))
    print("Standar deviation of all periods for file " + str(file) + " is:" + str(l_std_deviation) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    print("Distance with Zero Crossing (Average) - file " + str(file) + " is: " + str(average_distance) + "\n")

    print("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---")

    for m in range(len(l_distance_CF_periods)):
        print("Distance with Curve Fitting method for file " + str(file) + " in the " + str(m+1) + " period is: " + str(l_distance_CF_periods[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    l_std_deviation_CF.append(np.std(l_distance_CF_periods))
    print("Standar deviation of all periods for file " + str(file) + " is:" + str(l_std_deviation_CF) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")

    print("Distance with Curve Fitting method (Average) - file " + str(file) + " is: " + str(average_CF_distance) + "\n")

########################################################################################
########################################################################################
#"""


"""
###### -> NOTE: PROCESSED SIGNAL IS OUTPUT (CHANNEL B) WITHOUT POLYNOMIAL APPROXIMATION

Functionality: Do the signal processing to all the files and get the distance from the
target according to all csv files.
Input: 
Output: Distance from the target according to the different frequency methods. It
presents the distance from each range period of the channelB and the average of each
signals.
"""
#"""
def procesado_conjunto_Average_channelB(l_filename):
    l_distances_aux = []
    l_distances_aux_average = []
    l_distances_aux_CF = []
    l_distances_aux_average_CF = []

    Time_aux = []
    channelB_aux = []
    corrected_aux = []

    periods_Tim = []
    periods_chaB = []

    #l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]
    
    for i in range(len(l_filename)):
        ######Call functions
        (Time, channelA, channelB) = read_csv(l_filename[i])

        Time_aux.append(Time)
        channelB_aux.append(channelB)

        samples = samples_IntervalTime(Time)

        peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)

        peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)

        interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

        (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)
        periods_Tim.append(periods_Time)
        periods_chaB.append(periods_channelB)

        (fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)

        #(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
        (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)
        corrected_aux.append(corrected_Signals_periods)

        #size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)
        size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, fiit_real)

        #(output_def_periods, x_union) = interpolationSignal_where_crossZero(periods_channelB)
        (output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

        #(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, periods_channelB, periods_Time, x_union)
        (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, fiit_real, periods_Time, x_union)
        
        output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)

        (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

        frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

        #(freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, periods_channelB)
        (freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, fiit_real)

        fft_method(periods_channelB, samples, valor)

        (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

        l_distances_aux.append(l_distance_periods) #ALL THREE PERIODS
        l_distances_aux_average.append(average_distance) #AVERAGE

        l_std_deviation = []
        l_std_deviation_average = []

        l_distances_aux_CF.append(l_distance_CF_periods)
        l_distances_aux_average_CF.append(average_CF_distance)

        ### NEW MODIFICATION - Median process###
        l_median = []

    df = open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information.txt','w')

    for r in range(len(l_distances_aux)):
        l_std_deviation.append(np.std(l_distances_aux[r]))
        l_median.append(np.median(l_distances_aux[r]))
        print("Distance with Zero Crossing (Average) - file " + str(r+1) + " in the first period is: " + str(l_distances_aux[r][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux[r][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux[r][2]) + "\n")
        ### NEW MODIFICATION - First mode - Create txt file###
        
        df.write("Distance with Zero Crossing (Average) - file " + str(r+1) + " in the first period is: " + str(l_distances_aux[r][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux[r][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux[r][2]) + "\n")

    ### ERROR COEFFICIENT ###
    print(l_median)
    ec = 1.05
    l_median_Ec = []
    for e in range(len(l_median)):
        l_median_Ec.append(ec*l_median[e])
        
    print("Error coefficient")
    print(l_median_Ec)

    print("Standar deviation")
    st_median = np.std(l_median_Ec)
    
    print(st_median)

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    for l in range(len(l_std_deviation)):
        print("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation[l]) + "\n")
        df.write("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation[l]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")
    
    for m in range(len(l_distances_aux_average)):
        print("Distance with Zero Crossing (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average[m]))
        df.write("Distance with Zero Crossing (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_std_deviation_average.append(np.std(l_distances_aux_average))
    print("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average) + "\n")
    df.write("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_distances_aux_average_ZC_allt = []
    l_distances_aux_average_ZC_allt = statistics.mean(l_distances_aux_average)
    print("Average of distances with Zero crossing method: " + str(l_distances_aux_average_ZC_allt))
    df.write("Average of distances with Zero crossing method: " + str(l_distances_aux_average_ZC_allt) + "\n")

    print("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_std_deviation_CF = []
    l_std_deviation_average_CF = []

    for f in range(len(l_distances_aux_CF)):
        l_std_deviation_CF.append(np.std(l_distances_aux_CF[f]))
        print("Distance with Curve Fitting (Average) - file " + str(f+1) + " in the first period is: " + str(l_distances_aux_CF[f][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux_CF[f][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux_CF[f][2]) + "\n")

        df.write("Distance with Curve Fitting (Average) - file " + str(f+1) + " in the first period is: " + str(l_distances_aux_CF[f][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux_CF[f][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux_CF[f][2]) + "\n")
        
    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")
        
    for l in range(len(l_std_deviation_CF)):
        print("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation_CF[l]) + "\n")
        df.write("Standar deviation of all periods - file " + str(l+1) + " is: " + str(l_std_deviation_CF[l]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    for m in range(len(l_distances_aux_average_CF)):
        print("Distance with Curve Fitting (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average_CF[m]) + "\n")
        df.write("Distance with Curve Fitting (Average) - file " + str(m+1) + ": " + str(l_distances_aux_average_CF[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_std_deviation_average_CF.append(np.std(l_distances_aux_average_CF))
    print("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average_CF) + "\n")
    df.write("Standar deviation of all distances (Average) - all files: " + str(l_std_deviation_average_CF) + "\n")

    df.close()

    return(Time_aux, channelB_aux, corrected_aux, periods_Tim, periods_chaB)

#"""
###### -> NOTE: PROCESSED SIGNAL IS OUTPUT (CHANNEL B) WITHOUT POLYNOMIAL APPROXIMATION

"""
Functionality: Do the signal processing to the file selected and get the distance
from the target according to the csv file.
Input: File to analyze.
Output: Distance from the target according to the different frequency methods. It
presents the distance from each range period of the channelB and the average of each
signals.
"""
#"""
def procesado_selectivo_instant_andRepresentation_channelB(file):
    ######Call functions
    (Time, channelA, channelB) = read_csv(file)

    samples = samples_IntervalTime(Time)

    peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)

    peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)

    interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

    (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)

    (fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)

    (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
    #(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)

    size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)
    #size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, fiit_real)

    (output_def_periods, x_union) = interpolationSignal_where_crossZero(periods_channelB)
    #(output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

    (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, periods_channelB, periods_Time, x_union)
    #(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, fiit_real, periods_Time, x_union)
    
    output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)

    (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

    frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

    (freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, periods_channelB)
    #(freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, fiit_real)

    fft_method(periods_channelB, samples, valor)

    (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

    representation_Figures(periods_Time, periods_channelA, periods_channelB,
                       Time, channelA, channelB,
                       corrected_Signals_periods, model_full_range, polyline_full_range,
                       valor,
                       peaks_Time_B, peaks_channelB,
                       #peaks_Time_B_PC, peaks_channelB_1m_PC,
                       #output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis,
                       #output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis,
                       #output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC,
                       popt, tt)


    l_std_deviation = []
    l_std_deviation_CF = []

    df = open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information_select_instant.txt', 'w')
    
    for m in range(len(l_distance_periods)):
        print("Distance with Zero Crossing for file " + str(file) + " in the " +str(m+1) +  " period is: " + str(l_distance_periods[m]) + "\n")
        df.write("Distance with Zero Crossing for file " + str(file) + " in the " +str(m+1) +  " period is: " + str(l_distance_periods[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_std_deviation.append(np.std(l_distance_periods))
    print("Standar deviation of all periods for file " + str(file) + " is:" + str(l_std_deviation) + "\n")
    df.write("Standar deviation of all periods for file " + str(file) + " is:" + str(l_std_deviation) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    print("Distance with Zero Crossing (Average) - file " + str(file) + " is: " + str(average_distance) + "\n")
    df.write("Distance with Zero Crossing (Average) - file " + str(file) + " is: " + str(average_distance) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    print("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- CURVE FITTING METHOD --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    for m in range(len(l_distance_CF_periods)):
        print("Distance with Curve Fitting method for file " + str(file) + " in the " + str(m+1) + " period is: " + str(l_distance_CF_periods[m]))
        df.write("Distance with Curve Fitting method for file " + str(file) + " in the " + str(m+1) + " period is: " + str(l_distance_CF_periods[m]) + "\n")

    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n")

    l_std_deviation_CF.append(np.std(l_distance_CF_periods))
    print("Standar deviation of all periods for file " + str(file) + " is: " + str(average_CF_distance))
    df.write("Standar deviation of all periods for file " + str(file) + " is: " + str(average_CF_distance) + "\n")
             
    print("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---")
    df.write("--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---" + "\n") 

    print("Distance with Curve Fitting method (Average) - file " + str(file) + " is: " + str(average_CF_distance))
    df.write("Distance with Curve Fitting method (Average) - file " + str(file) + " is: " + str(average_CF_distance) + "\n")

    df.close()
#"""

########################################################################################
########################################################################################

"""
(Time, channelA, channelB) = read_csv(file1)

samples = samples_IntervalTime(Time)

peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)

peaks_Time_B_PC, peaks_channelB_1m_PC = peaks_section_channel_ParticularCase(Time, channelB)

interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

(periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)

(fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)

#(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)

(output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis,
 output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = peaks_periods_section(periods_Time, periods_channelB, corrected_Signals_periods)

(output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC) = peaks_periods_section_PC(periods_Time, periods_channelB)

size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)

(output_def_periods, x_union) = interpolationSignal_where_crossZero(corrected_Signals_periods)

(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, corrected_Signals_periods, periods_Time, x_union)

output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)

(average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

(freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, corrected_Signals_periods)

#fft_method(corrected_Signals_periods, samples, valor)
fft_method(periods_channelB, samples, valor)

(l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

representation_Figures(periods_Time, periods_channelA, periods_channelB,
                       Time, channelA, channelB,
                       corrected_Signals_periods, model_full_range, polyline_full_range,
                       valor,
                       peaks_Time_B, peaks_channelB,
                       #peaks_Time_B_PC, peaks_channelB_1m_PC,
                       #output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis,
                       #output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis,
                       #output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC,
                       popt, tt)
"""

"""
plt.figure()
plt.plot(freq_c[L_c], PSD[L_c])
plt.show()
"""


########################################################################################
########################################################################################

##Calls important functions
#procesado_conjunto_Average(l_filename)
#procesado_selectivo_instant_andRepresentation(file5)

#procesado_conjunto_Average_channelB(l_filename)
#procesado_selectivo_instant_andRepresentation_channelB(file1)

########################################################################################
########################################################################################
print("---------------------")

#"""
#NEW TASK --> 
#(time, channelB, corrected, period_time, fiit_R, corrected_aux_Sig) = procesado_conjunto_Average(l_filename)
(time, channelB, corrected, periods_Time, periods_channelB) = procesado_conjunto_Average_channelB(l_filename)

plt.figure(1)
#plt.plot(time[1], channelB[1])
#plt.plot(period_time[1][0], fiit_R[1][0])
#plt.plot(period_time[1][0], corrected[1][0])
#plt.plot(period_time[1][0], corrected_aux_Sig[1][0])
#plt.plot(period_time[1], corrected[1])

# --> Images : correction process
#plt.plot(time[1], channelB[1])
#plt.plot(time[7], channelB[7], color = 'orange')
#plt.plot(period_time[1][0], corrected[1][0], color = 'green')
#plt.plot(period_time[7][0], corrected[7][0], color = 'black')

# --> Images: no correction process
#plt.plot(time[0], channelB[0], label = 'frame 1')
plt.plot(periods_Time[0][2], periods_channelB[0][2], color = 'black', label = 'First period - frame 1')
#plt.plot(time[1], channelB[1], label = 'frame 2')
#plt.plot(periods_Time[1][0], periods_channelB[1][0], color = 'black', label = 'First period - frame 2')
#plt.plot(periods_Time[1][0], periods_channelB[1][0], color = 'blue')
#plt.plot(time[2], channelB[2])
#plt.plot(time[3], channelB[3], label = ' frame 4')
#plt.plot(periods_Time[3][2], periods_channelB[3][2], color = 'blue', label = 'First period - frame 4')
#plt.plot(time[4], channelB[4])
#plt.plot(time[5], channelB[5], label = 'instant 5')
#plt.plot(time[6], channelB[6], label = 'frame 7')
#plt.plot(periods_Time[6][0], periods_channelB[6][0], color = 'green', label = 'First period - frame 7')
#plt.plot(periods_Time[6][0], periods_channelB[6][0], color = 'black')
#plt.plot(time[7], channelB[7], label = 'frame 8')
#plt.plot(periods_Time[7][0], periods_channelB[7][0], color = 'blue', label = 'First period - frame 8')
#plt.plot(time[8], channelB[8], label = 'frame 9')
#plt.plot(periods_Time[8][2], periods_channelB[8][2], color = 'green', label = 'First period - frame 9')
#plt.plot(time[8][1], channelB[8][1])
#plt.plot(time[9], channelB[9], label = 'frame 10')
#plt.plot(periods_Time[9][0], periods_channelB[9][0], color = 'orange', label = 'First period - frame 10')
#plt.plot(time[9][1], channelB[9][1])

#plt.legend()
plt.title('Comparison between frames')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (mV)')
plt.show()
#"""



    
