from process_new_measurements import *

file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_10.csv')
file11 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_11.csv')
file12 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements_fifthTime/modulo 3 - 5ta medida/1m - modulo 3/1m - modulo 3_12.csv')


l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10, file11, file12]
#print(l_filename)
#l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]


def procesado_conjunto(l_filename):
    l_distances_aux = []
    l_distances_aux_average = []
    l_distances_aux_CF = []
    l_distances_aux_average_CF = []
    
    Time_aux = []
    channelB_aux = []
    corrected_aux = []
    corrected_aux_Sig = []
    
    period_time = []
    fiit_R = []

    frequency_est = []
    freq_CF_cB = []
    freq_CF_fii = []

    peaks_cB_full = []
    peaks_cB_periods = []
    
    for i in range(len(l_filename)):
        ### Call functions ###
        (Time, channelA, channelB) = read_csv(l_filename[i])

        samples = samples_IntervalTime(Time)

        (peaks_Time_A, peaks_Time_B, peaks_channelB) = peaks_section_channel(Time, channelA, channelB)
        peaks_cB_full.append(len(peaks_channelB))

        (peaks_TimeB_1m_PC, peaks_channelB_1m_PC) = peaks_section_channel_ParticularCase(Time, channelB)

        output_range = interval_periods(peaks_Time_A, Time, samples)

        (l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB) = concordance_periods_withSignals(output_range, Time, channelA, channelB)

        (fiit_real, freq_c, PSD, L_c) = denoising_signal(l_period_analyze_Time, l_period_analyze_channelB)

        (subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB)
        #(subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, fiit_real)
        
        (output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = peaks_periods_section(l_period_analyze_Time, l_period_analyze_channelB, subtract_l_full_range)
        peaks_cB_periods.append(len(output_peaks_channelB_periods[i]))
        
        (output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC) = peaks_periods_section_PC(l_period_analyze_Time, l_period_analyze_channelB)

        (size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, subtract_l_full_range)
        #(size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, fiit_real)

        (output_def_periods, x_union, l_cross) = interpolationSignal_where_crossZero(l_period_analyze_channelB)
        #(output_def_periods, x_union) = interpolationSignal_where_crossZero(subtract_l_full_range)
        #(output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

        (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, l_period_analyze_channelB, l_period_analyze_Time, x_union)
        #(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, subtract_l_full_range, l_period_analyze_Time, x_union)

        (output_interpolation) = interpolation_calculus_2part(output_subtract_div, output_Time_div)

        (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)
        print(average_periods_interpolation_section)
        print(sum_period)
        print(output_interpolation_periods)

        (frequency_estimation_ZC) = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)
        frequency_est.append(frequency_estimation_ZC)
        
        (freq_sineFunction_full, popt_full_index, tt_full) = curveFitting_sineFunction(l_period_analyze_Time, l_period_analyze_channelB)
        freq_CF_cB.append(freq_sineFunction_full)
        #(freq_sineFunction_full_sub, popt_full_index_sub, tt_full_sub) = curveFitting_sineFunction(l_period_analyze_Time, subtract_l_full_range)
        (freq_SineFunction_full_fii, popt_full_index_fii, tt_full_fii) = curveFitting_sineFunction(l_period_analyze_Time, fiit_real)
        freq_CF_fii.append(freq_SineFunction_full_fii)

    ### SHOW ###
    for i in range(len(peaks_cB_full)):
        print("Peaks full in channelB: " + str(peaks_cB_full[i]) + "\n")

    for t in range(len(peaks_cB_periods)):
        print("Peaks for every period in channelB: " + str(peaks_cB_periods[t]) + "\n")

    for r in range(len(frequency_est)):
        print("Frequency estimation ZC for every period: " + str(frequency_est[r]) + "\n")

    for g in range(len(freq_CF_cB)):
        print("Frequency estimation CF for every period in channelB original: " + str(freq_CF_cB[g]) + "\n")

    for w in range(len(freq_CF_fii)):
        print("Frequency estimation CF for every period in denoising channelB: " + str(freq_CF_fii[w]) + "\n")

    return(Time, channelA, channelB, l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis)

def procesado_selectivo(file):
    
    (Time, channelA, channelB) = read_csv(file)

    samples = samples_IntervalTime(Time)

    (peaks_Time_A, peaks_Time_B, peaks_channelB) = peaks_section_channel(Time, channelA, channelB)

    (peaks_TimeB_1m_PC, peaks_channelB_1m_PC) = peaks_section_channel_ParticularCase(Time, channelB)

    output_range = interval_periods(peaks_Time_A, Time, samples)

    (l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB) = concordance_periods_withSignals(output_range, Time, channelA, channelB)

    (fiit_real, freq_c, PSD, L_c) = denoising_signal(l_period_analyze_Time, l_period_analyze_channelB)

    (subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB)
    #(subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, fiit_real)
        
    (output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = peaks_periods_section(l_period_analyze_Time, l_period_analyze_channelB, subtract_l_full_range)
        
    (output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC) = peaks_periods_section_PC(l_period_analyze_Time, l_period_analyze_channelB)

    (size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, subtract_l_full_range)
    #(size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, fiit_real)

    (output_def_periods, x_union, l_cross) = interpolationSignal_where_crossZero(l_period_analyze_channelB)
    #(output_def_periods, x_union) = interpolationSignal_where_crossZero(subtract_l_full_range)
    #(output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

    (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, l_period_analyze_channelB, l_period_analyze_Time, x_union)
    #(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, subtract_l_full_range, l_period_analyze_Time, x_union)

    (output_interpolation) = interpolation_calculus_2part(output_subtract_div, output_Time_div)

    representation_Figures(Time, channelA, channelB, l_period_analyze_Time, l_period_analyze_channelB, subtract_l_full_range)

    (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

    (frequency_estimation_ZC) = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)
        
    (freq_sineFunction_full, popt_full_index, tt_full) = curveFitting_sineFunction(l_period_analyze_Time, l_period_analyze_channelB)
    #(freq_sineFunction_full_sub, popt_full_index_sub, tt_full_sub) = curveFitting_sineFunction(l_period_analyze_Time, subtract_l_full_range)
    (freq_SineFunction_full_fii, popt_full_index_fii, tt_full_fii) = curveFitting_sineFunction(l_period_analyze_Time, fiit_real)

    print("----- Compact process measurements ------")
    print(len(peaks_channelB))

    #print(len(output_peaks_channelB_periods[0]))
    for i in range(len(output_peaks_channelB_periods)):
        print(len(output_peaks_channelB_periods[i]))
        print(output_peaks_channelB_periods[i])
    
    print(frequency_estimation_ZC)

    print(freq_sineFunction_full)

    print(freq_SineFunction_full_fii)

    return (Time, channelA, channelB, l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis)

#(Time, channelA, channelB, l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = procesado_conjunto(l_filename)
(Time, channelA,
channelB, l_period_analyze_Time,
l_period_analyze_channelA, l_period_analyze_channelB,
output_peaks_channelB_periods,
 output_peaks_channelB_periods_yaxis) = procesado_selectivo(file1)

plt.figure(1)
plt.plot(Time, channelA)
plt.plot(Time, channelB)
plt.plot(l_period_analyze_Time[1], l_period_analyze_channelB[1], color = 'black')

plt.show()
