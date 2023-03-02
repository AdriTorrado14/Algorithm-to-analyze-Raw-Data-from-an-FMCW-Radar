from __future__ import division
import xlrd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import scipy.fft
from intersect import intersection
from scipy.signal import find_peaks

import statistics
from numpy.fft import rfft
from numpy import argmax, mean, diff, log, nonzero
from scipy.signal import blackmanharris, correlate, chirp, peak_widths
from time import time
import sys

from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

from itertools import islice


#Constants
freq = 20
T = 1/freq
T_halfPeriod= T/2
speed_ligth = 2.99e8
point_FreqHigh = 24.196e9
point_FreqLow = 24.082e9
deltaFreq = point_FreqHigh - point_FreqLow

valor = 1

"""
Functionality: Read and process the csv file.
Input: File to read
Output: Processed signals (Time, channelA and channelB)
"""
def read_csv(filename):
    print("Read_csv")
    data = pd.read_csv(filename, header = 0, sep = ";", decimal = ",")

    time = data['Tiempo']
    time.pop(0)
    lTime = [float(x.replace(',','.')) for x in time]
    print("lTime")
    print(lTime)
    print(len(lTime))

    channelA = data['Canal A']
    channelA.pop(0)
    lChannelA = [float(x.replace(',', '.')) for x in channelA]
    print("lChannelA")
    print(lChannelA)
    print(len(lChannelA))

    channelB = data['Canal B']
    channelB.pop(0)
    lChannelB = [float(x.replace(',', '.')) for x in channelB]
    print("lChannelB")
    print(lChannelB)
    print(len(lChannelB))
    return (lTime, lChannelA, lChannelB)

"""
Functionality: Get the number of samples corresponding to each period.
Input: Time signal (Complete signal)
Output: Number of samples (Round).
"""
def samples_IntervalTime(timeReal):
    #First part (Samples and period).
    intervalTime = abs(timeReal[10]-timeReal[9]) #In ms.
    #print(intervalTime)
    intervalTime_seg = intervalTime/1000 #In seconds.
    #print(intervalTime_seg)
    frequency_sampling = 1/intervalTime_seg
    intervalT_halfPeriod = intervalTime_seg + T_halfPeriod
    samples = T_halfPeriod/intervalTime_seg
    #print(samples)
    samples_Round = round(samples) #Round the number of samples.
    #print(samples_Round)
    samples_half = round(samples_Round/2)+1
    #print(samples_half)
    return(samples_Round)

"""
Functionality: Get the number of peaks. We applied the function to channelA and channelB.
To channelA because we want to know the number of periods comprising the signal.
To channelB because we want to know the number of peaks to see the differences
between different distances.
Input: Processed signals in 'read_csv' (Time, channelA, channelB)
Output: Asosiation of the signal peak over time (peaks_Time_A and peaks_Time_B) and
asosiation of the signal peak over amplitude (peaks_channel_B)
"""
def peaks_section_channel(timeReal, channelA, channelB):
    peaks_A, _ = find_peaks(channelA, height = 7.8) #Channel A
    l_peaks_A = []
    l_peaks_A = peaks_A.tolist()
    #print(l_peaks_A)

    #####TRY TO ERASE MULTIPLE PEAKS (PEAKS THAT ARE REALLY CLOSE TO EACH OTHER)
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
            
    ######WE USE "l_peaks_A_indexSamples" TO GET THE MEASUREMENT.
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
    #print(peaks_B)
    l_peaks_B = []
    l_peaks_B = peaks_B.tolist()

    peaks_Time_B = []
    for w in l_peaks_B:
        peaks_Time_B.append(timeReal[w])

    peaks_channelB = []
    for n in l_peaks_B:
        peaks_channelB.append(channelB[n])
    return (peaks_Time_A, peaks_Time_B, peaks_channelB)

"""
Functionality: Same function as the module before but with a different application.
We use this to see 1 meter particular case.
Input: Processed signals (Time and channelB)
Output: Asosiation of the signal peak over time (peaks_TimeB_1m_CP) and
asosiation of the signal peak over amplitude (peaks_channelB_1m_CP)
"""
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
    return (peaks_TimeB_1m_CP, peaks_channelB_1m_CP)

"""
Functionality: Get the samples range of each period of the signal channelA.
(Range in samples: Start point - End point)
Input: Asosiation of the signal peak over time (peaks_Time_A), processed
signal (Time) and the number of samples.
Output: Sample range of the different periods.
"""
def interval_periods(peaks_Time_A, timeReal, samples):
    print("interval_periods")
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
    print("output_range")
    print(output_range)
    #print(len(output_range))
    
    return (output_range)

"""
Functionality: Asign the samples range from the module 'interval_periods' with the
processed signals (Time, channelA, channelB)
Input: Samples range and the processed signals (time, channelA, channelB)
Output: Different range list of the processed signals. 
"""
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
    #######COPY INFORMATION (JUST IN CASE)#######
    l_period_analyze_Time_lengthPro = []
    l_period_analyze_channelA_lengthPro = []
    l_period_analyze_channelB_lengthPro = []
    l_period_analyze_Time_lengthPro = l_period_analyze_Time.copy()
    l_period_analyze_channelA_lengthPro = l_period_analyze_channelA.copy()
    l_period_analyze_channelB_lengthPro = l_period_analyze_channelB.copy()

    #######CHECK IF THE LENGTHS ARE THE SAME FOR THE DIFFERENT SIGNALS.
    value = []
    for g in range(len(l_period_analyze_Time)):
        value.append(len(l_period_analyze_Time[g]))
    min_value = min(value)
    for t in range(len(l_period_analyze_Time)):
        del l_period_analyze_Time[t][min_value:]
    """
    print("-Time-")
    print(len(l_period_analyze_Time[0]))
    print(len(l_period_analyze_Time[1]))
    print(len(l_period_analyze_Time[2]))
    """
    for g in range(len(l_period_analyze_channelA)):
        del l_period_analyze_channelA[g][min_value:]
    """
    print("- channel A -")
    print(len(l_period_analyze_channelA[0]))
    print(len(l_period_analyze_channelA[1]))
    print(len(l_period_analyze_channelA[2]))
    """
    for k in range(len(l_period_analyze_channelB)):
        del l_period_analyze_channelB[k][min_value:]
    """
    print(". channelB .")
    print(len(l_period_analyze_channelB[0]))
    print(len(l_period_analyze_channelB[1]))
    print(len(l_period_analyze_channelB[2]))
    """
    return (l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB)

"""
Functionality: Apply polynomial approximation to correct the signal output (ChannelB).
Input: Different ranges of the processed signals (Time, channelB, channelA)
Output: Corrected signal. We use this signal to get the distance from the target.
"""
def polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB):
    ###############POLYNOMIAL APPROXIMATION - 1STEP
    #print("Polinomial")
    model_full_range = []
    polyline_full_range = []
    for t, h in zip(range(len(l_period_analyze_Time)), range(len(l_period_analyze_channelB))):
        model_full_range.append(np.poly1d(np.polyfit(l_period_analyze_Time[t], l_period_analyze_channelB[h], 5)))
        polyline_full_range.append(np.linspace(l_period_analyze_Time[t][0], l_period_analyze_Time[t], len(l_period_analyze_Time[t])))
    """
    print(model_full_range)
    print(len(model_full_range))
    print(len(model_full_range[0]))
    print(len(model_full_range[1]))
    print(len(model_full_range[2]))
    print("polyline")
    print(polyline_full_range)
    print(len(polyline_full_range))
    print(len(polyline_full_range[0]))
    print(len(polyline_full_range[1]))
    print(len(polyline_full_range[2]))
    """
    ###############POLYNOMIAL APPROXIMATION - 2STEP.
    l_model_polyline_full_range = []
    for g, m in zip(range(len(model_full_range)), range(len(polyline_full_range))):
        l_model_polyline_full_range.append(model_full_range[g](polyline_full_range[m]))
    """
    print(l_model_polyline_full_range)
    print(len(l_model_polyline_full_range))
    print(len(l_model_polyline_full_range[0]))
    print(len(l_model_polyline_full_range[1]))
    print(len(l_model_polyline_full_range[2]))
    """
    ###############POLYNOMIAL APPROXIMATION - 3STEP
    l_agrupation_listModel_poly_full = []
    for g in range(len(l_model_polyline_full_range)):
        l_agrupation_listModel_poly_full.append(l_model_polyline_full_range[g][len(l_model_polyline_full_range[g])-1])
    """
    print(len(l_agrupation_listModel_poly_full[0]))
    print(len(l_agrupation_listModel_poly_full[1]))
    print(len(l_agrupation_listModel_poly_full[2]))
    """
    ################POLYNOMIAL APPROXIMATION - SUBTRACT - 4STEP.
    subtract_l_full_range = []
    for g, f in zip(range(len(l_period_analyze_channelB)), range(len(l_agrupation_listModel_poly_full))):
        subtract_l_full_range.append([t1 - t2 for t1, t2 in zip(l_period_analyze_channelB[g], l_agrupation_listModel_poly_full[f])])
    """
    print(len(subtract_l_full_range[0]))
    print(len(subtract_l_full_range[1]))
    print(len(subtract_l_full_range[2]))
    print(subtract_l_full_range[0])
    print(subtract_l_full_range[1])
    print(subtract_l_full_range[2])
    """
    return (subtract_l_full_range, model_full_range[valor], polyline_full_range[valor])

"""
Functionality:
Input:
Output:
"""
def peaks_periods_section(l_period_analyze_Time, l_period_analyze_channelB, subtract_l_full_range):
    ######PROCESSING FOR SUBTRACT SIGNAL#######
    l_peaks = []
    for t in range(len(subtract_l_full_range)):
        peaks_subtract, _ = find_peaks(subtract_l_full_range[t], height = 1, distance = 4)
        l_peaks.append(peaks_subtract)
    """
    print(l_peaks)
    print(len(l_peaks[0]))
    print(len(l_peaks[1]))
    print(len(l_peaks[2]))
    """
    ######"X-AXIS" - TIME
    peaks_Time_subtract = []
    union = []
    for r in range(len(l_peaks)):
        union.append(len(l_peaks[r]))
        for t in range(len(l_peaks[r])):
            peaks_Time_subtract.append(l_period_analyze_Time[r][l_peaks[r][t]])
    """
    print(peaks_Time_subtract)
    print(union)
    print(len(peaks_Time_subtract))
    """
    #######UNION DIFFERENT PERIODS.
    Input = iter(peaks_Time_subtract)
    output_peaks_subtract_periods = [list(islice(Input,elem)) for elem in union]
    """
    print(len(output_peaks_subtract_periods[0]))
    print(len(output_peaks_subtract_periods[1]))
    print(len(output_peaks_subtract_periods[2]))
    """
    #######Y-AXIS - SUBTRACT.
    peaks_yaxis_subtract = []
    for k in range(len(l_peaks)):
        for q in range(len(l_peaks[k])):
            peaks_yaxis_subtract.append(subtract_l_full_range[k][l_peaks[k][q]])
    #print(peaks_yaxis_subtract)
    #print(len(peaks_yaxis_subtract))

    #######UNION DIFFERENT PERIODS - YAXIS.
    Input_yaxis = iter(peaks_yaxis_subtract)
    output_peaks_subtract_periods_yaxis = [list(islice(Input_yaxis, elem)) for elem in union]
    """
    print(len(output_peaks_subtract_periods_yaxis[0]))
    print(len(output_peaks_subtract_periods_yaxis[1]))
    print(len(output_peaks_subtract_periods_yaxis[2]))
    """
    ##########################################################################################
    ######PROCESSING FOR CHANNELB SIGNAL#######
    l_peaks_channelB = []
    l_peaks_B = []
    index = 1
    l_peaks_B_indexSamples = []
    for g in range(len(l_period_analyze_channelB)):
        peaks_channelB, _ = find_peaks(l_period_analyze_channelB[g], height = 5.5, distance = 10)
        l_peaks_B = peaks_channelB.tolist()
        l_peaks_B_indexSamples.append(l_peaks_B)
        #print(l_peaks_B)
        l_peaks_channelB.append(peaks_channelB)

    #print(l_peaks_channelB)
    #print(l_peaks_B_indexSamples)

    """
    for r in range(len(l_peaks_B_indexSamples)):
        while index < len(l_peaks_B_indexSamples[r]):
            if l_peaks_B_indexSamples[index] - l_peaks_B_indexSamples[index-1] < 20:
                del l_peaks_B_indexSamples[index]
            else:
                index+=1
    print(l_peaks_B_indexSamples)
    """

    """
    index = 1
    while index < len(l_peaks_A_indexSamples):
        if l_peaks_A_indexSamples[index] - l_peaks_A_indexSamples[index-1] < 40:
            del l_peaks_A_indexSamples[index]
        else:
            index+=1
    """
    
    """
    #print(l_peaks_channelB)
    #print(len(l_peaks_channelB[0]))
    #print(len(l_peaks_channelB[1]))
    #print(len(l_peaks_channelB[2]))
    """
    peaks_Time_channelB = []
    union_cB = []
    for e in range(len(l_peaks_channelB)):
        union_cB.append(len(l_peaks_channelB[e]))
        for w in range(len(l_peaks_channelB[e])):
            peaks_Time_channelB.append(l_period_analyze_Time[e][l_peaks_channelB[e][w]])
    """
    #print(peaks_Time_channelB)
    #print(union_cB)
    #print(len(peaks_Time_channelB))
    """
    Input2 = iter(peaks_Time_channelB)
    output_peaks_channelB_periods = [list(islice(Input2, elem)) for elem in union_cB]
    """
    print(len(output_peaks_channelB_periods[0]))
    print(len(output_peaks_channelB_periods[1]))
    print(len(output_peaks_channelB_periods[2]))
    """
    ######Y-AXIS CHANNELB
    peaks_yaxis_channelB = []
    for l in range(len(l_peaks_channelB)):
        for f in range(len(l_peaks_channelB[l])):
            peaks_yaxis_channelB.append(l_period_analyze_channelB[l][l_peaks_channelB[l][f]])
    #######UNION DIFFERENT PERIODS - YAXIS.
    Input_yaxis_channelB = iter(peaks_yaxis_channelB)
    output_peaks_channelB_periods_yaxis = [list(islice(Input_yaxis_channelB, elem)) for elem in union_cB]
    """
    print(len(output_peaks_channelB_periods_yaxis[0]))
    print(len(output_peaks_channelB_periods_yaxis[1]))
    print(len(output_peaks_channelB_periods_yaxis[2]))
    """
    return(output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis)

"""
Functionality:
Input:
Output:
"""
def peaks_periods_section_PC(l_period_analyze_Time, l_period_analyze_channelB):
    ######PROCESSING FOR CHANNELB SIGNAL#######
    l_peaks_channelB_PC = []
    for g in range(len(l_period_analyze_channelB)):
        peaks_channelB_PC, _ = find_peaks(l_period_analyze_channelB[g], height = 5.5, distance = 10)
        l_peaks_channelB_PC.append(peaks_channelB_PC)
    """
    #print(l_peaks_channelB_PC)
    #print(len(l_peaks_channelB_PC[0]))
    #print(len(l_peaks_channelB_PC[1]))
    #print(len(l_peaks_channelB_PC[2]))
    """
    peaks_Time_channelB_PC = []
    union_cB = []
    for e in range(len(l_peaks_channelB_PC)):
        union_cB.append(len(l_peaks_channelB_PC[e]))
        for w in range(len(l_peaks_channelB_PC[e])):
            peaks_Time_channelB_PC.append(l_period_analyze_Time[e][l_peaks_channelB_PC[e][w]])
    """
    #print(peaks_Time_channelB_PC)
    #print(union_cB)
    #print(len(peaks_Time_channelB_PC))
    """
    Input2 = iter(peaks_Time_channelB_PC)
    output_peaks_channelB_periods_PC = [list(islice(Input2, elem)) for elem in union_cB]
    """
    print(len(output_peaks_channelB_periods_PC[0]))
    print(len(output_peaks_channelB_periods_PC[1]))
    print(len(output_peaks_channelB_periods_PC[2]))
    """
    ######Y-AXIS CHANNELB
    peaks_yaxis_channelB_PC = []
    for l in range(len(l_peaks_channelB_PC)):
        for f in range(len(l_peaks_channelB_PC[l])):
            peaks_yaxis_channelB_PC.append(l_period_analyze_channelB[l][l_peaks_channelB_PC[l][f]])
    #######UNION DIFFERENT PERIODS - YAXIS.
    Input_yaxis_channelB_PC = iter(peaks_yaxis_channelB_PC)
    output_peaks_channelB_periods_yaxis_PC = [list(islice(Input_yaxis_channelB_PC, elem)) for elem in union_cB]
    """
    print(len(output_peaks_channelB_periods_yaxis_PC[0]))
    print(len(output_peaks_channelB_periods_yaxis_PC[1]))
    print(len(output_peaks_channelB_periods_yaxis_PC[2]))
    """
    return(output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC)

"""
Functionality: Get the number of intersection between the input signal (channelA) and
the output signal (channelB)
Input: Range of the section of the processed signals (Time, channelA, channelB
and the corrected signal).
Output: Number of the intersection elements from each period.
"""
def find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, subtract_l_full_range):
    print("find_intersection")
    x_axis_Real = []
    y_axis_Real = []
    for t,g,r in zip(range(len(l_period_analyze_Time)), range(len(l_period_analyze_channelA)), range(len(l_period_analyze_channelB))):
        x_axis_Real.append(intersection(l_period_analyze_Time[t], l_period_analyze_channelA[g], l_period_analyze_Time[t], l_period_analyze_channelB[r]))
    #print(x_axis_Real)
    union_intersect_long = []
    union_intersect = []
    for r in range(len(x_axis_Real)):
        for t in range(len(x_axis_Real[r])):
            union_intersect_long.append(len(x_axis_Real[r][t]))
            union_intersect.append(x_axis_Real[r][t])
            union_intersect_def = [item for sublist in union_intersect for item in sublist]
    #print(union_intersect_long)
    #print(union_intersect_def)

    #####DIVISION OF THE LIST.
    Input = iter(union_intersect_def)
    output_intersect = [list(islice(Input, elem)) for elem in union_intersect_long]
    #print(output_intersect)

    #####ERASE "Y AXIS" POSITION IN THE LIST.
    del output_intersect[1::2]
    #print(output_intersect)
    
    #####SIZE OF THE LIST.
    size_list_intersect = []
    for r in range(len(output_intersect)):
        size_list_intersect.append(len(output_intersect[r]))
    print("size_list_intersect")
    print(size_list_intersect)

    return (size_list_intersect)
    ######AMPLIAR PARA CHANNELB SIN CORREGIR (SEÑAL ORIGINAL DE SALIDA)######

"""
Functionality: We get the number of sample when a zero cross occur in the corrected signal.
Also, we create a list with the following number of samples to get the completed range.
Input: Period section of the corrected signal (Polynomial approximation).
Output: Number of zero crossing elements in each periods (x_union), samples range (list)
of the zero-crossing in the corrected signal from each period.
"""
def interpolationSignal_where_crossZero(subtract_l_full_range):
    print("InterpolationSignal_where_CrossZero")
    zero_cross_full = []
    for h in range(len(subtract_l_full_range)):
        zero_cross_full.append(np.where(np.diff(np.sign(subtract_l_full_range[h])))[0])
    print("zero_cross_full")
    print(zero_cross_full)
    
    """
    print(zero_cross_full)
    print(len(zero_cross_full))
    print(len(zero_cross_full[0]))
    print(len(zero_cross_full[1]))
    print(len(zero_cross_full[2]))
    """

    l_cross = []
    l_polyline = []
    for d in range(len(zero_cross_full)):
        l_polyline.append(zero_cross_full[d]+1)
        l_cross.append(len(zero_cross_full[d]))
    print("l_cross")
    print(l_cross)
                    
    """
    print("polyline")
    print(l_polyline)
    print(len(l_polyline))
    print(len(l_polyline[0]))
    print(len(l_polyline[1]))
    print(len(l_polyline[2]))
    """
    union_periods = []
    for t, g in zip(range(len(zero_cross_full)), range(len(l_polyline))):
        union_periods.append(zero_cross_full[t])
        union_periods.append(l_polyline[g])
        union_periods_def = [item for sublist in union_periods for item in sublist]
    """
    print("Union")
    print(union_periods)
    print(union_periods_def) #ALL THE CROSS ZERO TOGETHER.
    """
    ##################IMPROVE THE METHOD --> DONE##################
    x_union = []
    for r in range(len(l_polyline)):
        x_union.append(2*len(l_polyline[r]))
    print("x_union")
    print(x_union)

    Input = iter(union_periods_def)
    output_def_periods = [list(islice(Input, elem)) for elem in x_union]
    #print(output_def_periods)
    for j in range(len(output_def_periods)):
        output_def_periods[j].sort()
    """   
    print(output_def_periods)
    print(len(output_def_periods[0]))
    print(len(output_def_periods[1]))
    print(len(output_def_periods[2]))
    """
    print("output_def_periods")
    print(output_def_periods)
    
    return (output_def_periods, x_union, l_cross)

"""
Functionality: Get the correspondence from the processed signals period range
over the samples range. We get the correspondence from the corrected
signal (polynomial approximation).
Input: Samples range of the zero crossing from each signal period (output_def_periods),
corrected signal (polynomial approximation) and processed signal (Time).
Output: List that shows the exact time and amplitude over the samples range
using the Zero-crossing from the interpolation.
"""
def interpolation_calculus_1part(output_def_periods, subtract_l_full_range, l_period_analyze_Time, x_union):
    print("interpolation_calculus_1part")
    #####INTERPOLATION - FIRST PART - GET POINTS OF ZERO CROSS.
    pos_subtractSignal_full_def = []
    for t in range(len(output_def_periods)):
        for f in range(len(output_def_periods[t])):
            try:
                pos_subtractSignal_full_def.append(subtract_l_full_range[t][output_def_periods[t][f]])
            except IndexError:
                pass

    #print("Desbordamiento")
    print("pos_subtractSignal_full_def")
    print(pos_subtractSignal_full_def)
    #print(len(pos_subtractSignal_full_def))

    Input = iter(pos_subtractSignal_full_def)
    output_subtract_div = [list(islice(Input, elem)) for elem in x_union]
    print("output_subtract_div")
    print(output_subtract_div)
    
    """
    print(output_subtract_div)
    print(len(output_subtract_div[0]))
    print(len(output_subtract_div[1]))
    print(len(output_subtract_div[2]))
    """
    #####CORRESPONDENCE WITH TIME SIGNAL.
    pos_Time_full_def = []
    for h in range(len(output_def_periods)):
        for l in range(len(output_def_periods[h])):
            try:
                pos_Time_full_def.append(l_period_analyze_Time[h][output_def_periods[h][l]])
            except IndexError:
                pass
    #print(pos_Time_full_def)

    Input_Time = iter(pos_Time_full_def)
    output_Time_div = [list(islice(Input_Time, elem)) for elem in x_union]
    print("output_Time_div")
    print(output_Time_div)
    """
    print(output_Time_div)
    print(len(output_Time_div[0]))
    print(len(output_Time_div[1]))
    print(len(output_Time_div[2]))
    """
    return(output_subtract_div, output_Time_div)

"""
Functionality: We do the interpolation calculus. We have the time and amplitude from
the module 'interpolation_calculus_1part' and we use the formula 1 to do the
interpolation. Also, we do a signal process because we need to erase some points
of the signal.
Input: List that shows the exact time and amplitude over the samples range using
the Zero-crossing from the interpolation.
Output: List that contains the interpolation of the corrected
signal (polynomial approximation).
"""
def interpolation_calculus_2part(output_subtract_div, output_Time_div):
    print("Interpolation_calculus_2part")
    list_Interpolation_ZC = []
    for t,g in zip(range(len(output_subtract_div)), range(len(output_Time_div))):
        for d,h in zip(range(len(output_subtract_div[t])-1), range(len(output_Time_div[g])-1)):
            try:
                pointst1 = output_Time_div[g][h] - ((output_subtract_div[t][d]/(output_subtract_div[t][d+1] - output_subtract_div[t][d]))*(output_Time_div[g][h+1]-output_Time_div[g][h]))
            except ZeroDivisionError:
               pointst1 = 0
            list_Interpolation_ZC.append(pointst1)
            
    print("list_interpolation_ZC")
    print(list_Interpolation_ZC)
    print("len lista_interpolation_ZC")
    print(len(list_Interpolation_ZC))

    ######UNION EN SUBLISTA DE LONGITUD 3.
    union_div = []
    for i in range(len(output_subtract_div)):
        #print(len(output_subtract_div[i]))
        union_div.append(len(output_subtract_div[i])-1)
    print("union div")
    print(union_div)

    Input_interpolation = iter(list_Interpolation_ZC)
    output_interpolation = [list(islice(Input_interpolation, elem)) for elem in union_div]
    
    print("output interpolation")
    print(output_interpolation)
    
    """
    print(output_interpolation)
    print(len(output_interpolation[0]))
    print(len(output_interpolation[1]))
    print(len(output_interpolation[2]))
    """
    #####REAL INTERPOLATION(ERASE WRONG RESULTS)
    for d in range(len(output_interpolation)):
        del output_interpolation[d][1:len(output_interpolation[d]):2]
        
    print("output_interpolation_wrong results")
    print(output_interpolation)

    output_interpolation_allPeriod = []
    output_interpolation_allPeriod = output_interpolation.copy()
    #print(output_interpolation_allPeriod)

    #########TO GET THE COMPLETE PERIOD OF THE REAL INTERPOLATION######
    for t in range(len(output_interpolation)):
        del output_interpolation[t][1:len(output_interpolation[t]):2]

    print("Output interpolation")
    print(output_interpolation)

    """
    ###################CHECK: CALCULUS OF ALL PERIODS (IN THIS CASE, SECTION PERIOD = 1##################
    list_proof = []
    for g in range(len(output_interpolation[1])-1):
        list_proof.append(output_interpolation[1][g+1]-output_interpolation[1][g])
    print(list_proof)
    """
    return(output_interpolation)

"""
Functionality: We get some data from the interpolation of the corrected signal (polynomial
approximation), period time of the different number of wave of the corrected signal,
period time (sum) of each period.
Input: List that contains the interpolation of the corrected signal
(polynomial approximation).
Output: Period time (sum) of each period and some data from the interpolation (period time
average of each period, subtract value from the different sine waveform).
"""
def calculus_periods_Interpolation(output_interpolation):
    print("calculus_periods_Interpolation")
    #####AT THE SAME TIME
    l_subtraction_periods = []
    for g in range(len(output_interpolation)):
        for d in range(len(output_interpolation[g])-1):
            l_subtraction_periods.append(output_interpolation[g][d+1]-output_interpolation[g][d])

    print("Subtraction periods")
    print(l_subtraction_periods)
    print("Longitud")
    print(len(l_subtraction_periods))
    
    #average_periods_interpolation_allTog = statistics.mean(l_subtraction_periods)
    #print(average_periods_interpolation_allTog)
    union_interpolation_periods = []
    for l in range(len(output_interpolation)):
        union_interpolation_periods.append(len(output_interpolation[l])-1)

    print("union_interpolation_periods")
    print(union_interpolation_periods)

    Input = iter(l_subtraction_periods)
    output_interpolation_periods = [list(islice(Input, elem)) for elem in union_interpolation_periods]

    print("output_interpolation_periods")
    print(output_interpolation_periods)
    print("len output_interpolation_periods")
    print(len(output_interpolation_periods))

    """
    print(output_interpolation_periods)
    print(len(output_interpolation_periods[0]))
    print(len(output_interpolation_periods[1]))
    print(len(output_interpolation_periods[2]))
    """
    #####AVERAGE AND STANDARD DEVIATION.
    
    average_periods_interpolation_section = []
    sum_period = []

    for t in range(len(output_interpolation_periods)):
        average_periods_interpolation_section.append(statistics.mean(output_interpolation_periods[t]))
        sum_period.append(sum(output_interpolation_periods[t]))
    
    print(average_periods_interpolation_section)
    print(sum_period)
    
    return(average_periods_interpolation_section, sum_period, output_interpolation_periods)
    #return(output_interpolation_periods)

"""
Functionality: Get the frequency using Zero-Crossing method.
Input: Interpolation range of the corrected signal (polynomial
approximation -- > corrected signal, period time of the interpolation section
(in ms) and the output interpolation range of the corrected signal
Output: Frequency of the different periods using Zero-Crossing methods.
"""
def ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods):
#def ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods):
    frequency_estimation_ZC = []
    for r, g in zip(range(len(output_interpolation_periods)), range(len(sum_period))):
        frequency_estimation_ZC.append(0.5*(len(output_interpolation_periods[r])/sum_period[g]*1000))
    #print(frequency_estimation_ZC)
    average_frequency_estimation_periods_ZC = statistics.mean(frequency_estimation_ZC)
    #print(average_frequency_estimation_periods_ZC)
    return (frequency_estimation_ZC)

"""
Functionality: 
Input: 
Output:
"""
def sinfunc(t, A, w, p, c):
    return A*np.sin(w*t + p) + c

"""
Functionality: 
Input: 
Output:
"""
def curveFitting_sineFunction(l_period_analyze_Time, subtract_l_full_range):
    tt_full = np.array(l_period_analyze_Time)
    #print(tt_full)
    #print(len(tt_full))
    
    yy_full = np.array(subtract_l_full_range)
    #print(yy_full)
    #print(len(yy_full))
    
    ff_full = []
    for r in range(len(tt_full)):
        ff_full.append(np.fft.fftfreq(len(tt_full[r]), abs(tt_full[r][1] - tt_full[r][0])))
    #print(ff_full)
    #print(len(ff_full))

    Fyy_full = []
    guess_amp_full = []
    guess_offset_full = []
    for i in range(len(yy_full)):
        Fyy_full.append(abs(np.fft.fft(yy_full[i])))
        guess_amp_full.append(np.std(yy_full[i] * 2.**0.5))
        guess_offset_full.append(np.mean(yy_full[i]))
        
    #print(Fyy_full)
    #print(len(Fyy_full))
    #print(guess_amp_full)
    #print(guess_offset_full)

    guess_freq_full = []
    for t,p in zip(range(len(ff_full)), range(len(Fyy_full))):
        guess_freq_full.append(abs(ff_full[t][np.argmax(Fyy_full[p][1:])+1]))
    #print(guess_freq_full)

    guess_freq_l_def = []
    for w in range(len(guess_freq_full)):
        guess_freq_l_def.append(2.*np.pi*guess_freq_full[w])
    #print(guess_freq_l_def)

    guess_full = []
    list_full_0 = []
    for i in range(len(guess_amp_full)):
        list_full_0.append(0)
   
    #guess_full = np.array(guess_amp_full, guess_freq_l_def, 0., guess_offset_full)
    guess_full.append(guess_amp_full)
    guess_full.append(guess_freq_l_def)
    guess_full.append(list_full_0)
    guess_full.append(guess_offset_full)
    #print(guess_full)

    guess_full_index = [[*items] for items in zip(*guess_full)]
    #print(guess_full_index)

    popt_full_index = []
    pcov_full_index = []
    for x,v,n in zip(range(len(tt_full)), range(len(yy_full)), range(len(guess_full_index))):
        popt_full, pcov_full = scipy.optimize.curve_fit(sinfunc,tt_full[x], yy_full[v], p0 = guess_full_index[n], maxfev = 15000)
        popt_full_index.append(popt_full)
        #popt_full_index.append(pcov_full)
        #print(popt_full)
        #print(pcov_full)
    ################################### --> CHECKEAR
    #print(popt_full_index)
    ############# RETURN THIS --> CHECKEAR --> FORM: ARRAY LIST.
    #print(len(popt_full_index))
    #print(type(popt_full_index))

    popt_full_index = [[*items] for items in zip(*popt_full_index)]
    #print(popt_full_index)

    w_full = (popt_full_index[1])
    #print(w_full)

    f_full = []
    f_ang_full = []
    T_sineFunction_full = []
    T_sineFunction_R_full = []
    freq_sineFunction_full = []
    
    for r in range(len(w_full)):
        f_full.append(w_full[r]/(2.*np.pi))
        f_ang_full.append(f_full[r]*1000)
        T_sineFunction_full.append(1/f_full[r])
        T_sineFunction_R_full.append(2*T_sineFunction_full[r])
        freq_sineFunction_full.append((1/T_sineFunction_R_full[r])*1000)
        
    #print(f_full)
    #print(f_ang_full) #Frequency first variant
    #print("Second variant")
    #print(T_sineFunction_full)
    #print(T_sineFunction_R_full)
    #print(freq_sineFunction_full) #Frequency second variant.

    ###################### CHECKEAR --> FITFUNC. 
    #popt = scipy.optimize.curve_fit(sinfunc, tt, yy, p0=guess, maxfev=15000)
    #fitfunc = lambda t: A * np.sin(w*t + p) + c

    return(freq_sineFunction_full, popt_full_index, tt_full)
    #return(freq_sineFunction_full, f_ang_full, popt_full_index, tt_full)
    #return(freq_SineFunction_Fitting_R, f_R_angular, popt, tt)

### AÑADIR FFT ###

"""
Functionality: 
Input: 
Output:
"""
def denoising_signal(periods_Time, periods_channelB):
    #for l in range(len(periods_Time)):
        #print(periods_Time[l])
        #print(len(periods_Time[l]))
    n = len(periods_Time[0])
    #print(n)
    dt = periods_Time[0][1]-periods_Time[0][0]
    #print(dt)
    fhat = []
    PSD = []
    for h in range(len(periods_channelB)):
        fhat.append(np.fft.fft(periods_channelB[h], n))
        PSD.append(fhat[h] * np.conj(fhat[h]) / n)
    """
    print(fhat)
    print(len(fhat))
    print(PSD)
    print(len(PSD))
    """

    freq = (1/(dt*n)) * np.arange(n)
    #print(freq)
    #print(len(freq))
    freq_c = freq.tolist()
    #print(freq_c)
    L = np.arange(1, np.floor(n/2), dtype = 'int')
    #print(L)
    #print(len(L))
    L_c = L.tolist()
    #print(L_c)

    indices = []
    PSD_clean = []
    fhat_2 = []
    for t in range(len(PSD)):
        indices.append(PSD[t] > 500)
    for h,l in zip(range(len(indices)), range(len(PSD))):
        PSD_clean.append(indices[h]*PSD[l])
    for d,l in zip(range(len(fhat)), range(len(indices))):
        fhat_2.append(indices[l]*fhat[d])

    """
    print(indices)
    print(len(indices))

    print(PSD_clean)
    print(len(PSD_clean))

    print(fhat_2)
    print(len(fhat_2))
    """

    fiit = []
    for i in range(len(fhat_2)):
        fiit.append(np.fft.ifft(fhat_2[i]))
    """
    print(fiit)
    print(len(fiit))
    print(len(fiit[0]))
    print(len(fiit[1]))
    print(len(fiit[2]))
    """

    fiit_real = []
    for i in range(len(fiit)):
        fiit_real.append(fiit[i].real)
    
    return(fiit_real, freq_c, PSD, L_c)

"""
Functionality:
Input:
Output:
"""
def representation_Figures(lTime, lChannelA, lChannelB,
                           #peaks_Time_B, peaks_channelB,
                           #peaks_TimeB_1m_PC, peaks_channelB_1m_PC,
                           l_period_analyze_Time, l_period_analyze_channelB,
                           #output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC,
                           #output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis,
                           subtract_l_full_range):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)', color = color)
    ax1.plot(lTime, lChannelA, color = color, label = "Channel A")
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.legend(loc = "best")

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Voltage (mV)', color = color)
    ax2.plot(lTime, lChannelB, color = color, label = "Channel B")
    #ax2.plot(peaks_Time_B, peaks_channelB, "*k")
    #ax2.plot(peaks_TimeB_1m_PC, peaks_channelB_1m_PC, "*k")
    ###
    #ax2.plot(output_peaks_channelB_periods_PC[0], output_peaks_channelB_periods_yaxis_PC[0], "*k")
    #ax2.plot(output_peaks_channelB_periods_PC[1], output_peaks_channelB_periods_yaxis_PC[1], "*k")
    #ax2.plot(output_peaks_channelB_periods_PC[2], output_peaks_channelB_periods_yaxis_PC[2], "*k")
    ###
    #ax2.plot(output_peaks_channelB_periods[0], output_peaks_channelB_periods_yaxis[0], "*k")
    #ax2.plot(output_peaks_channelB_periods[1], output_peaks_channelB_periods_yaxis[1], "*k")
    #ax2.plot(output_peaks_channelB_periods[2], output_peaks_channelB_periods_yaxis[2], "*k")
    ###
    ax2.plot(l_period_analyze_Time[0], l_period_analyze_channelB[0], color = 'orange')
    ax2.plot(l_period_analyze_Time[1], l_period_analyze_channelB[1], color = 'orange')
    ax2.plot(l_period_analyze_Time[2], l_period_analyze_channelB[2], color = 'orange')
    ###
    #ax2.plot(l_period_analyze_Time[0], subtract_l_full_range[0], color = 'red')
    #ax2.plot(l_period_analyze_Time[1], subtract_l_full_range[1], color = 'red')
    #ax2.plot(l_period_analyze_Time[2], subtract_l_full_range[2], color = 'red')
    ###
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.legend(loc = "best")

    plt.tight_layout()
    plt.grid()
    #plt.ion()
    plt.show() 

print("-----------------------------------------------------------------")
"""
###
(Time, channelA, channelB) = read_csv(file6)
#print(Time)
#print(len(Time))

samples = samples_IntervalTime(Time)
#print(samples)

(peaks_Time_A, peaks_Time_B, peaks_channelB) = peaks_section_channel(Time, channelA, channelB)
#print(peaks_Time_A)
#print(peaks_Time_B)
#print(peaks_channelB)
#print(len(peaks_channelB))

(peaks_TimeB_1m_PC, peaks_channelB_1m_PC) = peaks_section_channel_ParticularCase(Time, channelB)
#print(len(peaks_channelB_1m_PC))

output_range = interval_periods(peaks_Time_A, Time, samples)
#print(output_range)

(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB) = concordance_periods_withSignals(output_range, Time, channelA, channelB)
#print(l_period_analyze_Time)
#print(len(l_period_analyze_Time))
#print(l_period_analyze_channelA)
#print(len(l_period_analyze_channelA))

(fiit_real, freq_c, PSD, L_c) = denoising_signal(l_period_analyze_Time, l_period_analyze_channelB)

(subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB)
#(subtract_l_full_range, model_full_range, polyline_full_range) = polynomial_approximation(l_period_analyze_Time, l_period_analyze_channelA, fiit_real)
#print(len(subtract_l_full_range))

###
(output_peaks_subtract_periods, output_peaks_subtract_periods_yaxis, output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis) = peaks_periods_section(l_period_analyze_Time, l_period_analyze_channelB, subtract_l_full_range)
#print(output_peaks_subtract_periods)
#print(len(output_peaks_subtract_periods))
#print(output_peaks_subtract_periods_yaxis)
#print(len(output_peaks_subtract_periods_yaxis))
#print(output_peaks_channelB_periods)
#print(len(output_peaks_channelB_periods))
#print(output_peaks_channelB_periods_yaxis)
#print(len(output_peaks_channelB_periods_yaxis))

(output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC) = peaks_periods_section_PC(l_period_analyze_Time, l_period_analyze_channelB)
#print(output_peaks_channelB_periods_PC)
#print(len(output_peaks_channelB_periods_PC))
#print(output_peaks_channelB_periods_yaxis_PC)
#print(len(output_peaks_channelB_periods_yaxis_PC))
###

(size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, subtract_l_full_range)
#(size_list_intersect) = find_Intersection(l_period_analyze_Time, l_period_analyze_channelA, l_period_analyze_channelB, fiit_real)
print(size_list_intersect)

#(output_def_periods, x_union) = interpolationSignal_where_crossZero(subtract_l_full_range)
(output_def_periods, x_union, l_cross) = interpolationSignal_where_crossZero(l_period_analyze_channelB)
print(output_def_periods)
print(x_union)
print(l_cross)
#(output_def_periods, x_union) = interpolationSignal_where_crossZero(fiit_real)

#(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, subtract_l_full_range, l_period_analyze_Time, x_union)
(output_subtract_div, output_Time_div) = interpolation_calculus_1part(output_def_periods, l_period_analyze_channelB, l_period_analyze_Time, x_union)
print(output_subtract_div)
print(output_Time_div)

(output_interpolation) = interpolation_calculus_2part(output_subtract_div, output_Time_div)
print(output_interpolation)

(average_periods_interpolation_section, sum_period, output_interpolation_periods) = calculus_periods_Interpolation(output_interpolation)
print(average_periods_interpolation_section)
print(sum_period)
print(output_interpolation_periods)

(frequency_estimation_ZC) = ZeroCrossing_frequency_withInterpolation(average_periods_interpolation_section, sum_period, output_interpolation_periods)
print(frequency_estimation_ZC)

#(freq_sineFunction_full, popt_full_index, tt_full) = curveFitting_sineFunction(l_period_analyze_Time, subtract_l_full_range)
(freq_sineFunction_full, popt_full_index, tt_full) = curveFitting_sineFunction(l_period_analyze_Time, l_period_analyze_channelB)
#(freq_SineFunction_full, popt_full_index, tt_full) = curveFitting_sineFunction(l_period_analyze_Time, fiit_real)
print(freq_sineFunction_full)

representation_Figures(Time, channelA, channelB, peaks_Time_B, peaks_channelB,
                       peaks_TimeB_1m_PC, peaks_channelB_1m_PC,
                       l_period_analyze_Time, l_period_analyze_channelB,
                       #output_peaks_channelB_periods_PC, output_peaks_channelB_periods_yaxis_PC,
                       output_peaks_channelB_periods, output_peaks_channelB_periods_yaxis,
                       subtract_l_full_range)
"""


