from integration_2_copy import *
import telebot
import datetime
import requests

#Telegram's Bot.
TOKEN = '5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4'
chat_id = '9284473'
bot =telebot.TeleBot(TOKEN)
bot.config['api_key'] = TOKEN
#Bot status.
#print(bot.get_me())

"""
Functionality: 
Input:
Output:
"""

"""
def process_selective_Instant(time, channelA, channelB, file, actualSamplingInterval):
    samples = samples_IntervalTime(time, actualSamplingInterval)
    #print(samples)

    peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(time, channelA, channelB)
    print(peaks_Time_A)
    #print(peaks_Time_B)
    #print(peaks_channelB)

    peaks_Time_B_PC = peaks_section_channel_ParticularCase(time, channelB)

    interval_periods_ChannelA = interval_periods(peaks_Time_A, time, samples)

    (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, time, channelA, channelB)
    #print(len(periods_Time))
    #print(len(periods_channelA))

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
    #print(l_distance_periods)

    return(periods_Time, periods_channelA, periods_channelB)
"""

"""
Functionality: 
Input: 
Output:
"""

#"""
def process_selective_Instant(Time, channelA, channelB, file, actualSamplingInterval):

    #print(Time)
    Time_mV = []
    for i in range(len(Time)):
        Time_mV.append(Time[i]*1000)
    #print(Time_mV)

    #print(channelB)
    channelB_mV = []
    for t in range(len(channelB)):
        channelB_mV.append(channelB[t]*1000)
    #print(channelB_mV)
    
    #samples = samples_IntervalTime(Time, actualSamplingInterval)
    samples = samples_IntervalTime(Time_mV, actualSamplingInterval)

    #peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)
    peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time_mV, channelA, channelB_mV)
    #print(len(peaks_Time_B))

    #peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)
    peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time_mV, channelB_mV)

    #interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)
    interval_periods_ChannelA = interval_periods(peaks_Time_A, Time_mV, samples)

    #(periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)
    (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time_mV, channelA, channelB_mV)

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

    #fft_method(periods_channelB, samples, valor)

    (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

    l_std_deviation = []
    l_std_deviation_CF = []

    df = open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information_select_instant_def.txt', 'w')
    
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

    #Message to Telegram Bot. Information about the data.
    bot.send_message('9284473', 'Welcome to Radar Information bot. Here, you can see the obtained results.' +
                     ' You have chosen the first operating mode.')
    
    #Send the document to telegram bot.
    today= datetime.datetime.now()
    files = {'document':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information_select_instant_def.txt','rb')}
    resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendDocument?chat_id=9284473&caption=File txt'.format(today), files=files)
                
    return(periods_Time, periods_channelA, periods_channelB, model_full_range, polyline_full_range, Time_mV, channelB_mV, channelA, peaks_Time_B, peaks_channelB)
#"""

"""
Functionality: 
Input: 
Output:
"""
def process_Average_simult(files, actualSamplingInterval):
    l_distances_aux = []
    l_distances_aux_average = []
    l_distances_aux_CF = []
    l_distances_aux_average_CF = []

    Time_aux = []
    channelB_aux = []
    corrected_aux = []

    periods_Tim = []
    periods_chaB = []
    periods_chaA = []

    name_rows_measurement = ['Time', 'Channel A', 'Channel B']

    list_data_LTime = []
    list_data_lChannelA = []
    list_data_lChannelB = []

    Time_mV = []
    channelB_mV = []
    channelA = []

    for t in range(len(files)):
        df_measu = pd.read_csv(files[t], sep = ',', error_bad_lines = False, names = name_rows_measurement)
        #list_data_LTime.append(list(df_measu['Time']))
        list_data_LTime = list(df_measu['Time'])
                
        #list_data_lChannelA.append(list(df_measu['Channel A']))
        list_data_lChannelA = list(df_measu['Channel A'])
        channelA.append(list_data_lChannelA)
                
        #list_data_lChannelB.append(list(df_measu['Channel B']))
        list_data_lChannelB = list(df_measu['Channel B'])

        ### To mV (Time and channelB) ###
        for i in range(len(list_data_LTime)):
            Time_mV.append(list_data_LTime[i]*1000)

        for t in range(len(list_data_lChannelB)):
            channelB_mV.append(list_data_lChannelB[t]*1000)
        
        ### ADD FUNCTIONS ###
        #Time_aux.append(list_data_LTime)
        Time_aux.append(Time_mV)
        #channelB_aux.append(list_data_lChannelB)
        channelB_aux.append(channelB_mV)

        #samples = samples_IntervalTime(list_data_LTime, actualSamplingInterval)
        samples = samples_IntervalTime(Time_mV, actualSamplingInterval)

        #peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(list_data_LTime, list_data_lChannelA, list_data_lChannelB)
        (peaks_Time_A, peaks_Time_B, peaks_channelB) = peaks_section_channel(Time_mV, list_data_lChannelA, channelB_mV)
        #print("----")
        #print(len(peaks_channelB))
        #print(len(peaks_Time_A))

        #peaks_Time_B_PC = peaks_section_channel_ParticularCase(list_data_LTime, list_data_lChannelB)
        peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time_mV, channelB_mV)
        #print(len(peaks_Time_B_PC))
        #print(len(peaks_Time_B_PC[0]))
        #print("----")
        
        #interval_periods_ChannelA = interval_periods(peaks_Time_A, list_data_LTime, samples)
        interval_periods_ChannelA = interval_periods(peaks_Time_A, Time_mV, samples)

        #(periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, list_data_LTime, list_data_lChannelA, list_data_lChannelB)
        (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time_mV, list_data_lChannelA, channelB_mV)
        
        periods_Tim.append(periods_Time)
        periods_chaB.append(periods_channelB)
        periods_chaA.append(periods_channelA)

        #(fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)

        (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
        #(corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)
        corrected_aux.append(corrected_Signals_periods)

        (output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = peaks_periods_section(periods_Time, periods_channelB, corrected_Signals_periods) 
        #print("--------")
        #print(len(output_peaks_channelB_periods))
        #print(len(output_peaks_channelB_periods[1]))

        (output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC) = peaks_periods_section_PC(periods_Time, periods_channelB)
        #print(len(output_peaks_channelB_periods_yaxis_PC))
        #print(len(output_peaks_channelB_periods_yaxis_PC[1]))
        #print("--------")

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

        #fft_method(periods_channelB, samples, valor)

        (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

        l_distances_aux.append(l_distance_periods) #ALL THREE PERIODS
        l_distances_aux_average.append(average_distance) #AVERAGE

        l_std_deviation = []
        l_std_deviation_average = []

        l_distances_aux_CF.append(l_distance_CF_periods)
        l_distances_aux_average_CF.append(average_CF_distance)

        ### NEW MODIFICATION - Median process ###
        l_median = []

    df = open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information_simul_def.txt','w')

    for r in range(len(l_distances_aux)):
        l_std_deviation.append(np.std(l_distances_aux[r]))
        l_median.append(np.median(l_distances_aux[r]))
        print("Distance with Zero Crossing (Average) - file " + str(r+1) + " in the first period is: " + str(l_distances_aux[r][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux[r][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux[r][2]) + "\n")
        ### NEW MODIFICATION - First mode - Create txt file ###
        df.write("Distance with Zero Crossing (Average) - file " + str(r+1) + " in the first period is: " + str(l_distances_aux[r][0]) + "\n"
                                                                          " in the second period is: " + str(l_distances_aux[r][1]) + "\n"
                                                                          " in the third period is: " + str(l_distances_aux[r][2]) + "\n")

    ### ERROR COEFFICIENT ###
    #print(l_median)
    ec = 1.05
    l_median_Ec = []
    for e in range(len(l_median)):
        l_median_Ec.append(ec*l_median[e])
    #print("Error coefficient")
    #print(l_median_Ec)

    #print("Standar deviation")
    st_median = np.std(l_median_Ec)
    #print(st_median)

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

    #Message to Telegram Bot. Information about the data.
    bot.send_message('9284473', 'Welcome to Radar Information bot. Here, you can see the obtained results in your measurement.' +
                     ' You have chosen the first operating mode.')
    
    #Send the document to telegram bot.
    today= datetime.datetime.now()
    files = {'document':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/file_Information_simul_def.txt','rb')}
    resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendDocument?chat_id=9284473&caption=File txt'.format(today), files=files)

    return(Time_aux, channelA, channelB_aux, periods_Tim, periods_chaB, periods_chaA)
    
"""
Functionality: 
Input: 
Output:
"""
def process_LiveTime(Time, channelA, channelB, actualSamplingInterval):

    Time_mV = []
    for i in range(len(Time)):
        Time_mV.append(Time[i]*1000)

    channelB_mV = []
    for t in range(len(channelB)):
        channelB_mV.append(channelB[t]*1000)
    
    #samples = samples_IntervalTime(Time, actualSamplingInterval)
    samples = samples_IntervalTime(Time_mV, actualSamplingInterval)

    #peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time, channelA, channelB)
    peaks_Time_A, peaks_Time_B, peaks_channelB = peaks_section_channel(Time_mV, channelA, channelB_mV)

    #peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)
    peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time_mV, channelB_mV)

    #interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)
    interval_periods_ChannelA = interval_periods(peaks_Time_A, Time_mV, samples)
    
    #(periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)
    (periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time_mV, channelA, channelB_mV)
    
    #(fiit_real, freq_c, PSD, L_c) = denoising_signal(periods_Time, periods_channelB)

    (corrected_Signals_periods, model_full_range, polyline_full_range) = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)
    #(corrected_Signals_periods_fiit, model_full_range_fiit, polyline_full_range_fiit) = polynomial_approximation(periods_Time, periods_channelA, fiit_real)

    size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, corrected_Signals_periods)
    #size_list_intersect = find_Intersection(periods_Time, periods_channelA, periods_channelB, fiit_real)

    (output_def_periods, x_union) = interpolationSignal_where_crossZero(periods_channelB)
    #(output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

    (output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, periods_channelB, periods_Time, x_union)
    #print(output_subtract_div)
    #print(output_Time_div)
    #(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, fiit_real, periods_Time, x_union)
        
    output_interpolation = interpolation_calculus_2part(output_subtract_div, output_Time_div)
    #print(output_interpolation)

    (average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)

    frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)

    (freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, periods_channelB)
    #(freq_SineFunction_Fitting_R, popt, tt) = curveFitting_sineFunction(periods_Time, fiit_real)

    #fft_method(periods_channelB, samples, valor)

    (l_distance_periods, average_distance, l_distance_CF_periods, average_CF_distance) = calculation(peaks_Time_B, peaks_Time_B_PC, size_list_intersect, peaks_Time_A, frequency_estimation_ZC, freq_SineFunction_Fitting_R)

    l_distances_aux=[]
    l_distances_aux.append(l_distance_periods) #ALL THREE PERIODS
    #l_distances_aux_average.append(average_distance) #AVERAGE

    l_std_deviation = []
    l_std_deviation.append(np.std(l_distances_aux))

    l_median = []
    l_median.append(np.median(l_distances_aux))
    ec = 1.05
    l_median_Ec = []
    for i in range(len(l_median)):
        l_median_Ec.append(ec*l_median[i])

    return(l_median, l_median_Ec, l_std_deviation, periods_Time, periods_channelB, corrected_Signals_periods, Time_mV, channelB_mV)
    #return (periods_Time, periods_channelB, corrected_Signals_periods, Time_mV, channelB_mV)
