import sys
from scipy.fftpack import fft
import scipy.fft
from scipy.signal import find_peaks
from numpy.fft import rfft
import matplotlib.pyplot as plt
import statistics
from intersect import intersection
from statistics import StatisticsError
import pandas as pd
import numpy as np

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

#Constants
f = 20
T = 1/f
T_halfPeriod = T/2
speed_ligth = 2.99e8
point_FreqHigh = 24.196e9
point_FreqLow = 24.082e9
deltaFreq = point_FreqHigh - point_FreqLow
#deltaFreq = 140e6

#To analyze period number two.
valor = 1

def read_csv(filename):
    data = pd.read_csv(filename, header = 0)
    columns_names = data.columns.values
    #Data processing of the excel file.
    listTime = list(data['Time'])
    listAuxTime = listTime.copy()
    #print(len(listAuxTime))
    listAuxTime.pop(0)
    listAuxTimeReal = list(float(i) for i in listAuxTime)
    #print(len(listAuxTimeReal))
    #print(listAuxTimeReal)

    listChannelA = list(data['Channel A'])
    listAuxChannelA = listChannelA.copy()
    listAuxChannelA.pop(0)
    listAuxChannelAReal = list(float(i) for i in listAuxChannelA)
    #print(len(listAuxChannelAReal))
    #print(listAuxChannelAReal)

    listChannelB = list(data['Channel B'])
    listAuxChannelB = listChannelB.copy()
    listAuxChannelB.pop(0)
    listAuxChannelBReal = list(float(i) for i in listAuxChannelB)
    #print(len(listAuxChannelBReal))
    #print(listAuxChannelBReal)
    
    return (listAuxTimeReal, listAuxChannelAReal, listAuxChannelBReal)

def samples_IntervalTime(timeReal):
    #First part (Samples and period).
    intervalTime = abs(timeReal[10]-timeReal[9])
    #print(intervalTime)
    intervalTime_seg = intervalTime/1000
    #print(intervalTime_seg)
    frequency_sampling = 1/intervalTime_seg
    intervalT_halfPeriod = intervalTime_seg + T_halfPeriod
    samples = T_halfPeriod/intervalTime_seg

    samples_Round = round(samples)
    print(samples_Round)
    samples_half = round(samples_Round/2)+1
    print(samples_half)
    #samples_half_Round = round(samples_half)+1
    #print(samples_half_Round)

    return(samples_Round)

def peaks_section_channel(timeReal, channelA, channelB):
    peaks_A, _ = find_peaks(channelA, height = 7.8) #Channel A
    l_peaks_A = []
    l_peaks_A = peaks_A.tolist()
    #print(l_peaks_A)

    #####TRY TO ERASE MULTIPLE PEAKS (PEAKS THAT ARE REALLY CLOSE TO EACH OTHER
    l_peaks_A_indexSamples = []
    l_peaks_A_indexSamples = l_peaks_A.copy()
    #print(l_peaks_A_indexSamples)

    index = 1
    while index < len(l_peaks_A_indexSamples):
        if l_peaks_A_indexSamples[index] - l_peaks_A_indexSamples[index-1] < 40:
            del l_peaks_A_indexSamples[index]
        else:
            index+=1
    #print(l_peaks_A_indexSamples)
    ######WE USE "L_PEAKS_A_INDEXSAMPLES" TO GET THE MEASUREMENT.
    #####

    peaks_Time_A = []
    for s in l_peaks_A_indexSamples:
        peaks_Time_A.append(timeReal[s])
    #print(peaks_Time_A)
    #print(len(peaks_Time_A))

    peaks_channelA = []
    for r in l_peaks_A_indexSamples:
        peaks_channelA.append(channelA[r])

    ######################################CHANNELB##########################################
    peaks_B, _ = find_peaks(channelB, height = 4, distance = 15) #Channel B (for cases != 1m).
    l_peaks_B = []
    l_peaks_B = peaks_B.tolist()

    peaks_Time_B = []
    for w in l_peaks_B:
        peaks_Time_B.append(timeReal[w])

    peaks_channelB = []
    for n in l_peaks_B:
        peaks_channelB.append(channelB[n])

    return (peaks_Time_A, peaks_Time_B)

def peaks_section_channel_ParticularCase(timeReal, channelB):
    peaks_B_1m, _ = find_peaks(channelB, height = 5, distance=35)
    l_peaks_B_1m = []
    l_peaks_B_1m = peaks_B_1m.tolist()

    peaks_TimeB_1m_CP = []
    for f in l_peaks_B_1m:
        peaks_TimeB_1m_CP.append(timeReal[f])

    peaks_channelB_1m_CP = []
    for r in l_peaks_B_1m:
        peaks_channelB_1m_CP.append(channelB[r])

    return (peaks_TimeB_1m_CP)


def interval_periods(peaks_Time_A, timeReal, samples):
    ############CONCORDANCE PEAKS IN CHANNEL A WITH THE LENGTH OF NUMBER OF SAMPLES
    list_index_peaks = []
    for t in range(len(peaks_Time_A)):
        list_index_peaks.append(timeReal.index(peaks_Time_A[t]))
    #print(list_index_peaks)

    ############CREATION THE RANGE OF THE MEASUREMENT -(3 HALF HIGH PERIOD).
    value = 0
    for g in range(len(list_index_peaks)):
        value = list_index_peaks[g] + samples
        list_index_peaks.append(value)

    #print(list_index_peaks)
    list_index_peaks.sort()
    #print(list_index_peaks)
    list_index_peaks.pop(0)
    list_index_peaks.pop(len(list_index_peaks)-1)
    #print(list_index_peaks)

    ##############DIVERSIFICATION/SEPARATION BETWEEN RANGE OF PERIODS.
    sep = 2
    final_list_peaks_range = lambda list_index_peaks, sep: [list_index_peaks[i:i+sep] for i in range(0, len(list_index_peaks), sep)]
    output_range = final_list_peaks_range(list_index_peaks, sep)
    #print(output_range)
    #print(len(output_range))

    return (output_range)

def concordance_periods_withSignals(output_range, timeReal, channelA, channelB):
    ##############PERIODS TO ANALYZE
    l_period_analyze_Time = []
    l_period_analyze_channelA = []
    l_period_analyze_channelB = []
    for j in range(len(output_range)):
        l_period_analyze_Time.append(timeReal[output_range[j][0]:output_range[j][1]])
        l_period_analyze_channelA.append(channelA[output_range[j][0]:output_range[j][1]])
        l_period_analyze_channelB.append(channelB[output_range[j][0]:output_range[j][1]])

    """
    print("-")
    print(len(l_period_analyze_Time))
    print(len(l_period_analyze_Time[0]))
    print(len(l_period_analyze_Time[1]))
    print(len(l_period_analyze_Time[2]))

    print(".")
    print(len(l_period_analyze_channelA[0]))
    print(len(l_period_analyze_channelA[1]))
    print(len(l_period_analyze_channelA[2]))
    print("-")
    
    print(len(l_period_analyze_channelB[0]))
    print(len(l_period_analyze_channelB[1]))
    print(len(l_period_analyze_channelB[2]))
    """
    
    #######COPY#######
    l_period_analyze_Time_lengthPro = []
    l_period_analyze_channelA_lengthPro = []
    l_period_analyze_channelB_lengthPro = []

    l_period_analyze_Time_lengthPro = l_period_analyze_Time.copy()
    l_period_analyze_channelA_lengthPro = l_period_analyze_channelA.copy()
    l_period_analyze_channelB_lengthPro = l_period_analyze_channelB.copy()

    #######CHECK SAME LENGTHS
    value = []
    for g in range(len(l_period_analyze_Time)):
        value.append(len(l_period_analyze_Time[g]))
    min_value = min(value)

    #print("..")
    for t in range(len(l_period_analyze_Time)):
        del l_period_analyze_Time[t][min_value:]
    #print(len(l_period_analyze_Time[0]))
    #print(len(l_period_analyze_Time[1]))
    #print(len(l_period_analyze_Time[2]))

    for g in range(len(l_period_analyze_channelA)):
        del l_period_analyze_channelA[g][min_value:]
    #print("-")
    #print(len(l_period_analyze_channelA[0]))
    #print(len(l_period_analyze_channelA[1]))
    #print(len(l_period_analyze_channelA[2]))

    for k in range(len(l_period_analyze_channelB)):
        del l_period_analyze_channelB[k][min_value:]
    #print(".")
    #print(len(l_period_analyze_channelB[0]))
    #print(len(l_period_analyze_channelB[1]))
    #print(len(l_period_analyze_channelB[2]))

    return (l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB)

def polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB):
    ##############POLYNOMIAL
    model_full_range = []
    polyline_full_range = []
    for t, h in zip(range(len(l_period_analyze_Time)), range(len(l_period_analyze_channelB))):
        model_full_range.append(np.poly1d(np.polyfit(l_period_analyze_Time[t], l_period_analyze_channelB[h], 3)))
        polyline_full_range.append(np.linspace(l_period_analyze_Time[t][0], l_period_analyze_Time[t], len(l_period_analyze_Time[t])))

    #print(model_full_range)
    #print(len(model_full_range))
    #print(len(model_full_range[0]))
    #print(len(model_full_range[1]))
    #print(len(model_full_range[2]))
    #print("polyline")
    #print(polyline_full_range)
    #print(len(polyline_full_range[0]))
    #print(len(polyline_full_range[1]))
    #print(len(polyline_full_range[2]))

    ###############POLYNOMIAL - 2STEP.
    l_model_polyline_full_range = []
    for g, m in zip(range(len(model_full_range)), range(len(polyline_full_range))):
        l_model_polyline_full_range.append(model_full_range[g](polyline_full_range[m]))

    #print(l_model_polyline_full_range)
    #print(len(l_model_polyline_full_range))
    #print(len(l_model_polyline_full_range[0]))
    #print(len(l_model_polyline_full_range[1]))
    #print(len(l_model_polyline_full_range[2]))

    ###############POLYNOMIAL - 3STEP
    l_agrupation_listModel_poly_full = []
    for g in range(len(l_model_polyline_full_range)):
        l_agrupation_listModel_poly_full.append(l_model_polyline_full_range[g][len(l_model_polyline_full_range[g])-1])

    #print(len(l_agrupation_listModel_poly_full[0]))
    #print(len(l_agrupation_listModel_poly_full[1]))
    #print(len(l_agrupation_listModel_poly_full[2]))

    ################POLYNOMIAL - SUBTRACT - 4STEP.
    subtract_l_full_range = []
    for g, f in zip(range(len(l_period_analyze_channelB)), range(len(l_agrupation_listModel_poly_full))):
        subtract_l_full_range.append([t1 - t2 for t1, t2 in zip(l_period_analyze_channelB[g], l_agrupation_listModel_poly_full[f])])

    #print(len(subtract_l_full_range[0]))
    #print(len(subtract_l_full_range[1]))
    #print(len(subtract_l_full_range[2]))
    #print(subtract_l_full_range)

    return (subtract_l_full_range)

def interpolationSignal_where_crossZero(subtract_l_full_range):
    zero_cross_full = []
    for h in range(len(subtract_l_full_range)):
        zero_cross_full.append(np.where(np.diff(np.sign(subtract_l_full_range[h])))[0])

    print(type(zero_cross_full))
    print(len(zero_cross_full))
    print(len(zero_cross_full[0]))
    print(len(zero_cross_full[1]))
    print(len(zero_cross_full[2]))

    ###############################################################################
    """####CHECK IF THE LENGTHS ARE THE SAME.
    value = []
    for g in range(len(zero_cross_full)):
        value.append(len(zero_cross_full[g]))
    min_value = min(value)

    for t in range(len(zero_cross_full)):
        del zero_cross_full[t][min_value:]

    print(len(zero_cross_full[0]))
    print(len(zero_cross_full[1]))
    print(len(zero_cross_full[2]))"""
    ###############################################################################
    
    l_polyline = []
    for d in range(len(zero_cross_full)):
        l_polyline.append(zero_cross_full[d]+1)

    print(l_polyline)
    print(len(l_polyline))
    print(len(l_polyline[0]))
    print(len(l_polyline[1]))
    print(len(l_polyline[2]))
    
    union_periods = []
    for t, g in zip(range(len(zero_cross_full)), range(len(l_polyline))):
        union_periods.append(zero_cross_full[t])
        union_periods.append(l_polyline[g])
        union_periods_def = [item for sublist in union_periods for item in sublist]

    print(union_periods)
    print(union_periods_def)
    ####CORRECTO

    ##########################SEPARATION 
    x2 = 2*len(l_polyline[0]) #Separation the list in 3 list (One for each period).
    def_l_periods = lambda union_periods_def, x2: [union_periods_def[i:i+x2] for i in range(0, len(union_periods_def), x2)]
    output_def_periods = def_l_periods(union_periods_def, x2)
    ##########################ORDENACION DE LA LISTA.
    for j in range(len(output_def_periods)):
        output_def_periods[j].sort()

    print(output_def_periods)
    print(len(output_def_periods[0]))
    print(len(output_def_periods[1]))
    print(len(output_def_periods[2]))

    return (output_def_periods)

def interpolation_calculus_1part(output_def_periods, subtract_l_full_range, l_period_analyze_Time):
    #####INTERPOLATION - FIRST PART - GET POINTS OF ZERO CROSS.
    pos_subtractSignal_full_def = []
    for t in range(len(output_def_periods)):
        for f in range(len(output_def_periods[t])):
            try:
                pos_subtractSignal_full_def.append(subtract_l_full_range[t][output_def_periods[t][f]])
            except IndexError:
                pass

    print("Desbordamiento")
    print(pos_subtractSignal_full_def)
    print(len(pos_subtractSignal_full_def))

    #####UNION EN SUBLISTA DE LONGITUD 3
    g = len(output_def_periods[0])
    final_list_union = lambda pos_subtractSignal_full_def, g: [pos_subtractSignal_full_def[i:i+g] for i in range(0, len(pos_subtractSignal_full_def), g)]
    output_1part_period = final_list_union(pos_subtractSignal_full_def, g)

    print(output_1part_period)
    print(len(output_1part_period))
    
    pos_Time_full_def = []
    for h in range(len(output_def_periods)):
        for l in range(len(output_def_periods[h])):
            try:
                pos_Time_full_def.append(l_period_analyze_Time[h][output_def_periods[h][l]])
            except IndexError:
                pass
    print(pos_Time_full_def)
    
    #####UNION EN SUBLISTA DE LONGITUD 3
    final_list_union_Time = lambda pos_Time_full_def, g: [pos_Time_full_def[i:i+g] for i in range(0, len(pos_Time_full_def), g)]
    output_1part_union_Time = final_list_union_Time(pos_Time_full_def, g)
    print(len(output_1part_union_Time))
    print(len(output_1part_union_Time[0]))
    print(len(output_1part_union_Time[1]))
    print(len(output_1part_union_Time[2]))
    print("Comprobacion desbordamiento")

    return(output_1part_period, output_1part_union_Time)

def interpolation_calculus_2part(output_1part_period, output_1part_union_Time):
    list_Interpolation_ZC = []
    for t,g in zip(range(len(output_1part_period)), range(len(output_1part_union_Time))):
        for d,h in zip(range(len(output_1part_period[t])-1), range(len(output_1part_union_Time[g])-1)):
            try:
                pointst1 = output_1part_union_Time[g][h] - ((output_1part_period[t][d]/(output_1part_period[t][d+1] - output_1part_period[t][d]))*(output_1part_union_Time[g][h+1]-output_1part_union_Time[g][h]))
            except ZeroDivisionError:
               pointst1 = 0
            list_Interpolation_ZC.append(pointst1)

    #UNION EN SUBLISTA DE LONGITUD 3
    #print(list_Interpolation_ZC)
    f = len(output_1part_period[0])-1
    f_interpolation = lambda list_Interpolation_ZC, f:[list_Interpolation_ZC[i:i+f] for i in range(0, len(list_Interpolation_ZC), f)]
    output_f_interpolation = f_interpolation(list_Interpolation_ZC, f)
    print(output_f_interpolation)
    print(len(output_f_interpolation))

    #####REAL INTERPOLATION
    for d in range(len(output_f_interpolation)):
        del output_f_interpolation[d][1:len(output_f_interpolation[d]):2]
    print(output_f_interpolation)

    output_f_interpolation_allPeriod = []
    output_f_interpolation_allPeriod = output_f_interpolation.copy()
    print(output_f_interpolation_allPeriod)

    #########TO GET THE COMPLETE PERIOD OF THE REAL INTERPOLATION######
    for t in range(len(output_f_interpolation)):
        del output_f_interpolation[t][1:len(output_f_interpolation[t]):2]
    print(output_f_interpolation)

    ###################CHECK: CALCULUS OF ALL PERIODS (IN THIS CASE, SECTION PERIOD = 1##################
    list_proof = []
    for g in range(len(output_f_interpolation[1])-1):
        list_proof.append(output_f_interpolation[1][g+1]-output_f_interpolation[1][g])
    print(list_proof)

    return(output_f_interpolation)

def calculus_periods_Interpolation(output_f_interpolation):
    #####AT THE SAME TIME
    l_subtraction_periods = []
    for g in range(len(output_f_interpolation)):
        for d in range(len(output_f_interpolation[g])-1):
            l_subtraction_periods.append(output_f_interpolation[g][d+1]-output_f_interpolation[g][d])

    print(l_subtraction_periods)
    #print(len(l_subtraction_periods))

    average_periods_interpolation_allTog = statistics.mean(l_subtraction_periods)
    #print(average_periods_interpolation_allTog)

    #####LIST'S DIVISION BY 3.
    division_average_interpolation = len(output_f_interpolation[0])-1
    f_periods_interpolation_average = lambda l_subtraction_periods, division_average_interpolation:[l_subtraction_periods[i:i+division_average_interpolation] for i in range(0, len(l_subtraction_periods), division_average_interpolation)]
    output_f_periods_interp_average = f_periods_interpolation_average(l_subtraction_periods, division_average_interpolation)

    print("-")
    print(output_f_periods_interp_average)
    #print(len(output_f_periods_interp_average[0]))

    #####AVERAGE AND STANDARD DEVIATION.
    average_periods_interpolation_section = []
    sum_period = []
    for t in range(len(output_f_periods_interp_average)):
        average_periods_interpolation_section.append(statistics.mean(output_f_periods_interp_average[t]))
        sum_period.append(sum(output_f_periods_interp_average[t]))

    print(average_periods_interpolation_section)
    print(sum_period)

    return(average_periods_interpolation_section, sum_period, output_f_periods_interp_average)

def ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_f_periods_interp_average):
    frequency_estimation_ZC = []
    for r, g in zip(range(len(output_f_periods_interp_average)), range(len(sum_period))):
        #print(len(output_f_periods_interp_average[r]))
        frequency_estimation_ZC.append(0.5*(len(output_f_periods_interp_average[r])/sum_period[g]*1000))
    print(frequency_estimation_ZC)

    return (frequency_estimation_ZC)

def method_ZC(frequency_estimation_ZC):
    R_ZC_all_together = []
    for g in range(len(frequency_estimation_ZC)):
        R_ZC_all_together.append((speed_ligth*frequency_estimation_ZC[g]*T)/(2*2*deltaFreq))
    print(R_ZC_all_together)


def representation_Figures(timeRange, channelA_Range, channelB_Range, timeReal, channelA_Real, channelB_Real, corrected_Signal, valor):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)', color = color)
    ax1.plot(timeReal, channelA_Real, color = color, label = "Channel A")
    #ax1.plot(timeRange[valor], channelA_Range[valor], label = "Half-period Channel A")
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.legend(loc = "best")

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Voltage (mV)', color = color)
    ax2.plot(timeReal, channelB_Real, color = color, label = "Channel B")
    #ax2.plot(timeRange[valor], channelB_Range[valor], label = "Half-period Channel B")
    ax2.plot(timeRange[valor], corrected_Signal[valor], color = 'purple', label = "Corrected signal - Channel B")

    #ax2.plot(tt, sinfunc(tt, *popt), color = 'red', label = "Sine curve fitting")
    #ax2.plot(polyline, modelo(polyline), color = 'green')

    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.legend(loc = "best")

    plt.tight_layout()
    plt.grid()
    #plt.ion()
    plt.show()

    
"""
def processing_automatic_Polynomial_correction(peaks_Time_A, timeReal, samples, channelA, channelB):
    ############CONCORDANCE PEAKS IN CHANNEL A WITH THE LENGTH OF NUMBER OF SAMPLES
    list_index_peaks = []
    for t in range(len(peaks_Time_A)):
        list_index_peaks.append(timeReal.index(peaks_Time_A[t]))
    print(list_index_peaks)

    ############CREATION THE RANGE OF THE MEASUREMENT -(3 HALF HIGH PERIOD).
    value = 0
    for g in range(len(list_index_peaks)):
        value = list_index_peaks[g] + samples
        list_index_peaks.append(value)

    print(list_index_peaks)
    list_index_peaks.sort()
    print(list_index_peaks)
    list_index_peaks.pop(0)
    list_index_peaks.pop(len(list_index_peaks)-1)
    print(list_index_peaks)

    ##############DIVERSIFICATION/SEPARATION BETWEEN RANGE OF PERIODS.
    sep = 2
    final_list_peaks_range = lambda list_index_peaks, sep: [list_index_peaks[i:i+sep] for i in range(0, len(list_index_peaks), sep)]
    output_range = final_list_peaks_range(list_index_peaks, sep)
    print(output_range)
    print(len(output_range))

    ##############PERIODS TO ANALYZE
    l_period_analyze_Time = []
    l_period_analyze_channelA = []
    l_period_analyze_channelB = []
    for j in range(len(output_range)):
        l_period_analyze_Time.append(timeReal[output_range[j][0]:output_range[j][1]])
        l_period_analyze_channelA.append(channelA[output_range[j][0]:output_range[j][1]])
        l_period_analyze_channelB.append(channelB[output_range[j][0]:output_range[j][1]])

    print("LLLLLL")
    print(len(l_period_analyze_Time))
    print(len(l_period_analyze_Time[0]))
    print(len(l_period_analyze_Time[1]))
    print(len(l_period_analyze_Time[2]))

    print(".")
    print(len(l_period_analyze_channelA[0]))
    print(len(l_period_analyze_channelA[1]))
    print(len(l_period_analyze_channelA[2]))
    print(".")
    
    print(len(l_period_analyze_channelB[0]))
    print(len(l_period_analyze_channelB[1]))
    print(len(l_period_analyze_channelB[2]))

    #######COPY#######
    l_period_analyze_Time_lengthPro = []
    l_period_analyze_channelA_lengthPro = []
    l_period_analyze_channelB_lengthPro = []

    l_period_analyze_Time_lengthPro = l_period_analyze_Time.copy()
    l_period_analyze_channelA_lengthPro = l_period_analyze_channelA.copy()
    l_period_analyze_channelB_lengthPro = l_period_analyze_channelB.copy()

    #######CHECK SAME LENGTHS
    value = []
    for g in range(len(l_period_analyze_Time)):
        value.append(len(l_period_analyze_Time[g]))
    min_value = min(value)

    print("..")
    for t in range(len(l_period_analyze_Time)):
        del l_period_analyze_Time[t][min_value:]
    print(len(l_period_analyze_Time[0]))
    print(len(l_period_analyze_Time[1]))
    print(len(l_period_analyze_Time[2]))

    for g in range(len(l_period_analyze_channelA)):
        del l_period_analyze_channelA[g][min_value:]
    print("-")
    print(len(l_period_analyze_channelA[0]))
    print(len(l_period_analyze_channelA[1]))
    print(len(l_period_analyze_channelA[2]))

    for k in range(len(l_period_analyze_channelB)):
        del l_period_analyze_channelB[k][min_value:]
    print(".")
    print(len(l_period_analyze_channelB[0]))
    print(len(l_period_analyze_channelB[1]))
    print(len(l_period_analyze_channelB[2]))
    #######CHECK SAME LENGTHS --> DONE.

    ##############POLYNOMIAL
    model_full_range = []
    polyline_full_range = []
    for t, h in zip(range(len(l_period_analyze_Time)), range(len(l_period_analyze_channelB))):
        model_full_range.append(np.poly1d(np.polyfit(l_period_analyze_Time[t], l_period_analyze_channelB[h], 3)))
        polyline_full_range.append(np.linspace(l_period_analyze_Time[t][0], l_period_analyze_Time[t], len(l_period_analyze_Time[t])))
    print(len(model_full_range))
    print(len(model_full_range[0]))
    print(len(model_full_range[1]))
    print(len(model_full_range[2]))
    print("polyline")
    print(len(polyline_full_range[0]))
    print(len(polyline_full_range[1]))
    print(len(polyline_full_range[2]))

    ###############POLYNOMIAL - 2STEP.
    l_model_polyline_full_range = []
    for g, m in zip(range(len(model_full_range)), range(len(polyline_full_range))):
        l_model_polyline_full_range.append(model_full_range[g](polyline_full_range[m]))
    print(len(l_model_polyline_full_range[0]))
    print(len(l_model_polyline_full_range[1]))
    print(len(l_model_polyline_full_range[2]))

    ###############POLYNOMIAL - 3STEP
    l_agrupation_model_polyline_full_range = []
    for t in range(len(l_model_polyline_full_range)):
        ####### -> CHECK LENGTH OF LISTS.
        try:
            l_agrupation_model_polyline_full_range.append(l_model_polyline_full_range[t][len(l_model_polyline_full_range[g])-1])
        except IndexError:
            pass
        ####### -> CHECK LENGTH OF LISTS.
    #print(len(l_agrupation_model_polyline_full_range[0]))
    #print(len(l_agrupation_model_polyline_full_range[1]))
    #print(len(l_agrupation_model_polyline_full_range[2]))

    ################POLYNOMIAL - SUBTRACT - 4STEP.
    subtract_full_range = []
    for g, f in zip(range(len(l_period_analyze_channelB)), range(len(l_agrupation_model_polyline_full_range))):
        subtract_full_range.append([t1-t2 for t1, t2 in zip(l_period_analyze_channelB[g], l_agrupation_model_polyline_full_range[f])])
    #print(len(subtract_full_range[0]))
    #print(len(subtract_full_range[1]))
    #print(len(subtract_full_range[2]))

    return(l_period_analyze_Time, l_period_analyze_channelB, l_period_analyze_channelA, subtract_full_range) #subtract_full_range -> correctedSignal_range


def find_Intersection(time_period_range, channelA_period_range, channelB_period_range, correctedSignal_range, valor):
    x_axis_Real, y_axis_Real = intersection(time_period_range[valor], channelA_period_range[valor], time_period_range[valor], channelB_period_range[valor]) #Intersection original signal section (ChannelB)
    x_axis_corrected, y_axis_corrected = intersection(time_period_range[valor], channelA_period_range[valor], time_period_range[valor], correctedSignal_range[valor]) #Intersection corrected signal section (polynomial)
    x_axisP = []
    x_axisP = x_axis_corrected.tolist()
    y_axisP = y_axis_corrected.tolist()
    x_axisR = []
    x_axisR = x_axis_Real.tolist()
    #print("..")
    #print(x_axisR)
    #print(len(x_axisR))
    y_axisR = y_axis_Real.tolist()

    return (x_axisR)

def where_ZC(correctedSignal_range, time_period_range, channelB_period_range, valor):
    #First - Know where zero crossings occur
    zero_cross_CS = np.where(np.diff(np.sign(correctedSignal_range[valor])))[0]
    zeroCross_NP_CS = []
    zeroCross_NP_CS = zero_cross_CS.tolist() #Channel B Corrected signal.

    zero_cross_CB = np.where(np.diff(np.sign(channelB_period_range[valor])))[0]
    zeroCross_NP_CB = []
    zeroCross_NP_CB = zero_cross_CB.tolist() #Channel B without correction.
    #print(zeroCross_NP_CB)

    #####
    ###Corrected signal.
    zeroCross_aux_def_CS = []
    zeroCross_aux_def_CS = zeroCross_NP_CS.copy()

    laux_zeroCross_def_CS = []
    for p in range(len(zeroCross_NP_CS)):
        laux_zeroCross_def_CS.append(zeroCross_NP_CS[p]+1)
        zeroCross_NP_CS.append(zeroCross_NP_CS[p]+1)

    zeroCross_NP_CS.sort() #Ordenar.
    print(zeroCross_NP_CS)

    ###ChannelB
    zeroCross_aux_def_CB = []
    zeroCross_aux_def_CB = zeroCross_NP_CB.copy()

    laux_zeroCross_def_CB = []
    for t in range(len(zeroCross_NP_CB)):
        laux_zeroCross_def_CB.append(zeroCross_NP_CB[t]+1)
        zeroCross_NP_CB.append(zeroCross_NP_CB[t]+1)
    #print(zeroCross_NP_CB)

    zeroCross_NP_CB.sort()
    print(zeroCross_NP_CB)
    #####
    return (zeroCross_NP_CS, zeroCross_NP_CB)

def where_ZC_toInterpolation(zeroCross_def_CB, zeroCross_def_CS, correctedSignal_range, time_period_range, channelB_period_range, valor):
    l_CS_ZC = [] #Corrected signal
    for y in zeroCross_def_CS:
        l_CS_ZC.append(correctedSignal_range[valor][y])
    #print(l_CS_ZC)
    #print(len(l_CS_ZC))
    #print(type(l_CS_ZC))

    l_CB_ZC = [] #Channel B signal.
    for h in zeroCross_def_CB:
        l_CB_ZC.append(channelB_period_range[valor][h])
    #print(l_CB_ZC)
    #print(len(l_CB_ZC))
    #print(type(l_CB_ZC))
    

    l_Time_CS_ZC = [] #Corrected signal
    for q in zeroCross_def_CS:
        l_Time_CS_ZC.append(time_period_range[valor][q])
    #print(l_Time_CS_ZC)
    #print(len(l_Time_CS_ZC))
    #print(type(l_Time_CS_ZC))

    l_Time_CB_ZC = [] #ChannelB signal.
    for d in zeroCross_def_CB:
        l_Time_CB_ZC.append(time_period_range[valor][d])
    #print(l_Time_CB_ZC)
    #print(len(l_Time_CB_ZC))
    #print(type(l_Time_CB_ZC))
    
    return (l_CS_ZC, l_Time_CS_ZC, l_CB_ZC, l_Time_CB_ZC)

def Interpolation_process(CS_ZC, Time_CS_ZC, CB_ZC, Time_CB_ZC):
    ###Corrected Signal
    laux_Interpolation_CS = []
    for k, l in zip(range(len(CS_ZC)-1), range(len(Time_CS_ZC)-1)):
        try:
            point_t1_CS = Time_CS_ZC[l] - ((CS_ZC[k]/(CS_ZC[k+1] - CS_ZC[k]))*(Time_CS_ZC[l+1] - Time_CS_ZC[l]))                                       
        except ZeroDivisionError:
            point_t1_CS = 0
        laux_Interpolation_CS.append(point_t1_CS)
    print("--")
    #print(laux_Interpolation_CS)
    del laux_Interpolation_CS[1:len(laux_Interpolation_CS):2] #Interpolation list Correct signal
    #print(laux_Interpolation_CS)

    ###ChannelB
    laux_Interpolation_CB = []
    for u, m in zip(range(len(CB_ZC)-1), range(len(Time_CB_ZC)-1)):
        try:
            point_t1_CB = Time_CB_ZC[m] - ((CB_ZC[u]/(CB_ZC[u+1] - CB_ZC[u]))*(Time_CB_ZC[m+1] - Time_CB_ZC[m]))
        except ZeroDivisionError:
            point_t1_CB = 0
        laux_Interpolation_CB.append(point_t1_CB)
    #print(laux_Interpolation_CB)
    del laux_Interpolation_CB[1:len(laux_Interpolation_CB):2] #Interpolation list ChannelB
    #print(laux_Interpolation_CB)
    #print(len(laux_Interpolation_CB))

    #Eliminar posiciones impares de la onda.
    ###Corrected signal
    laux_Inteporlation_CS_def = []
    laux_Interpolation_CS_def = laux_Interpolation_CS.copy()
    del laux_Interpolation_CS_def[1::2]
    #print(laux_Interpolation_CS_def)
    #print(len(laux_Interpolation_CS_def))
    
    ###ChannelB
    laux_Interpolation_CB_def = []
    laux_Interpolation_CB_def = laux_Interpolation_CB.copy()
    del laux_Interpolation_CB_def[1::2]
    #print(laux_Interpolation_CB_def)
    #print(len(laux_Interpolation_CB_def))

    return (laux_Interpolation_CS_def, laux_Interpolation_CB_def)

def method_ZC_IntervalTime(l_Interpolation_CS, l_Interpolation_CB):
    ###Corrected signal
    laux_intervalTime_ZC = []
    for b in range(len(l_Interpolation_CS)-1):
        laux_intervalTime_ZC.append(l_Interpolation_CS[b+1] - l_Interpolation_CS[b])
    #print(laux_intervalTime_ZC)
    #print(len(laux_intervalTime_ZC))

    ###Channel B
    laux_intervalTime_ZC_CB = []
    for t in range(len(l_Interpolation_CB)-1):
        laux_intervalTime_ZC_CB.append(l_Interpolation_CB[t+1] - l_Interpolation_CB[t])
    #print(laux_intervalTime_ZC_CB)
    #print(len(laux_intervalTime_ZC_CB))

    return (laux_intervalTime_ZC, laux_intervalTime_ZC_CB)

############################################################################################
##### - SECOND METHOD TO ESTIMATE A FREQUENCY: CURVE FITTING METHOD - SINUSOID APPROXIMATION.
#Sinusoid function
def sinfunc(t, A, w, p, c):
    return A*np.sin(w*t + p) + c

def curveFitting_sineFunction(time_period_range, correctedSignal_range, valor):
    tt = np.array(time_period_range[valor])
    yy = np.array(correctedSignal_range[valor])
    ff = np.fft.fftfreq(len(tt), abs(tt[1]-tt[0]))
    Fyy = abs(np.fft.fft(yy))
    guess_freq = abs(ff[np.argmax(Fyy[1:])+1])
    guess_amp = np.std(yy) * 2.**0.5
    guess_offset = np.mean(yy)
    guess = np.array([guess_amp, 2.*np.pi*guess_freq, 0., guess_offset])

    popt, pcov = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=15000)
    A, w, p, c = popt
    f=w/(2.*np.pi) #Frequencia Angular.
    f_R = f*1000 #Without two times multiplication.
    fitfunc = lambda t: A * np.sin(w*t + p) + c
    
    #Improve next lines (Check how to present data).
    T_sineFunction_Fitting = 1/f
    T_sineFunction_Fitting_R = 2*T_sineFunction_Fitting
    freq_SineFunction_Fitting = 1/T_sineFunction_Fitting_R
    freq_SineFunction_Fitting_R = freq_SineFunction_Fitting*1000

    return(freq_SineFunction_Fitting_R, f_R)
############################################################################################

############################################################################################
##### - THIRD METHOD TO ESTIMATE A FREQUENCY: FFT
def fft_method(channelB_period_range, samples, valor):
    samplingFrequency = samples/T_halfPeriod
    Y_k = np.fft.fft(channelB_period_range[valor])[0:int(samples/2)]/samples
    Y_k[1:] = 2*Y_k[1:]
    Pxx = np.abs(Y_k)
    f_axis = samplingFrequency*np.arange((samples/2))/samples
    f_axis_l = []
    f_axis_l = f_axis.tolist()
    f_axis_l.pop(76)

    #Find peaks
    peaksFFT_halfPeriod, _ = find_peaks(Pxx, height = 2.5)
    peaksFFT_halfPeriod_l = []
    peaksFFT_halfPeriod_l = peaksFFT_halfPeriod.tolist()

    peaksFFT_l_Pxx = []
    for e in peaksFFT_halfPeriod_l:
        peaksFFT_l_Pxx.append(Pxx[e])

    peaksFFT_l_f_axis = []
    for r in peaksFFT_halfPeriod_l:
        peaksFFT_l_f_axis.append(f_axis_l)
    #Process to acquire peak. (Review to increase the accuracy).
    # Now: -> It works with the sum of the list.
    #value_sum_frequency_FFT = sum(peaksFFT_l_f_axis)
    value_frequency_FFT_average = statistics.mean(peaksFFT_l_f_axis)
    return (value_frequency_FFT_average)

def calculation(peaks_Time_B, peaks_Time_B_PC, intersection_axisX_Signal, peaks_Time_A, valor, ):
    if (len(peaks_Time_B) <= 29 and len(peaks_Time_B_PC) <= 15 and len(intersection_axisX_Signal) <= 5):
        print("Caso 1 meter")
        l_ParticularCase=[]
        for t in range(len(peaks_Time_A)-1):
            l_ParticularCase.append((peaks_Time_A[t+1]- peaks_Time_A[t])-15)
        value_IntervalTime_mean = statistics.mean(l_ParticularCase)
        #print(value_IntervalTime_mean)
        freq_PC = 1/(value_IntervalTime_mean)*1000
        l_value_freq_PC = []
        l_value_freq_PC.append(freq_PC)
        l_frequencies_aux_PC = []
        for g in range(len(l_value_freq_PC)):
            distance_PC = (speed_ligth*T*l_value_freq_PC[g])/(2*2*deltaFreq)
            l_frequencies_aux_PC.append(distance_PC)
        print("Distance particular case (1 meter): " + str(l_frequencies_aux_PC[0]))
        return(l_frequencies_aux_PC)

    #elif (len(peaks_Time_B) >= 30):
        #print("Caso different to 1 meter")
"""

############################################################################################
(Time, channelA, channelB) = read_csv(file2)

samples = samples_IntervalTime(Time)

peaks_Time_A, peaks_Time_B = peaks_section_channel(Time, channelA, channelB)

peaks_Time_B_PC = peaks_section_channel_ParticularCase(Time, channelB)

interval_periods_ChannelA = interval_periods(peaks_Time_A, Time, samples)

(periods_Time, periods_channelA, periods_channelB) = concordance_periods_withSignals(interval_periods_ChannelA, Time, channelA, channelB)

corrected_Signals_periods = polynomial_approximation(periods_Time, periods_channelA, periods_channelB)                                                                                       

output_def_periods = interpolationSignal_where_crossZero(corrected_Signals_periods)

(output_1part_period, output_1part_union_Time) = interpolation_calculus_1part(output_def_periods, corrected_Signals_periods, periods_Time)

output_f_interpolation = interpolation_calculus_2part(output_1part_period, output_1part_union_Time)

(average_periods_interpolation_section, sum_period, output_f_periods_interp_average) = calculus_periods_Interpolation(output_f_interpolation)

frequency_estimation_ZC = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_f_periods_interp_average)

method_ZC(frequency_estimation_ZC)

representation_Figures(periods_Time, periods_channelA, periods_channelB, Time, channelA, channelB, corrected_Signals_periods, valor)

"""
(Time_range, channelB_range, channelA_range, subtract_range) = processing_automatic_Polynomial_correction(peaks_Time_A, Time, samples, channelA, channelB)

x_axisR = find_Intersection(Time_range, channelA_range, channelB_range, subtract_range, valor)

(zeroCross_CS, zeroCross_CB) = where_ZC(subtract_range, Time_range, channelB_range, valor)

(l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc) = where_ZC_toInterpolation(zeroCross_CB, zeroCross_CS, subtract_range, Time_range, channelB_range, valor)

(l_interpolation_CS, l_interpolation_CB) = Interpolation_process(l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc)

(intervalTime_ZC_CS, intervalTime_ZC_CB) = method_ZC_IntervalTime(l_interpolation_CS, l_interpolation_CB)

(freq_CV_sineFunction, freq_CV_R) = curveFitting_sineFunction(Time_range, subtract_range, valor)

(distance) = calculation(peaks_Time_B, peaks_Time_B_PC, x_axisR, peaks_Time_A, valor)

#### --> FFT method.

representation_Figures(Time_range, channelA_range, channelB_range, Time, channelA, channelB, subtract_range, valor)
"""

