from __future__ import division
import pandas as pd
import numpy as np
import sys
from scipy.fftpack import fft
import scipy.fft
from scipy.signal import find_peaks
from numpy.fft import rfft
import matplotlib.pyplot as plt
import statistics
from intersect import intersection
from statistics import StatisticsError


#"""
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
#"""



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



"""file1 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_01.csv')
file2 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_02.csv')
file3 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_03.csv')
file4 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_04.csv')
file5 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_05.csv')
file6 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_06.csv')
file7 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_07.csv')
file8 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_08.csv')
file9 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_09.csv')
file10 = ('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/Radar_Measurements/5m_2/5m_2_10.csv')"""


#Constants
f = 20
T = 1/f
T_halfPeriod = T/2
speed_ligth = 2.99e8
point_FreqHigh = 24.196e9
point_FreqLow = 24.082e9
deltaFreq = point_FreqHigh - point_FreqLow
#deltaFreq = 140e6


def read_csv(filename):
    data = pd.read_csv(filename, header = 0)
    columns_names = data.columns.values
    #Data processing of the excel file.
    listTime = list(data['Time'])
    listAuxTime = listTime.copy()
    listAuxTime.pop(0)
    listAuxTimeReal = list(float(i) for i in listAuxTime)

    listChannelA = list(data['Channel A'])
    listAuxChannelA = listChannelA.copy()
    listAuxChannelA.pop(0)
    listAuxChannelAReal = list(float(i) for i in listAuxChannelA)

    listChannelB = list(data['Channel B'])
    listAuxChannelB = listChannelB.copy()
    listAuxChannelB.pop(0)
    listAuxChannelBReal = list(float(i) for i in listAuxChannelB)
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
    samples_half = samples_Round/2
    samples_half_Round = round(samples_half)+1

    return(samples_Round)

def processing_signal_section(timeReal, channelA_Real, channelB_Real, samples):
    #Time process.
    lTime = []
    lTime=timeReal.copy()
    del lTime[(samples*2)+77:len(timeReal)] #Half-period up.
    lTime_aux = []
    lTime_aux = lTime.copy()
    del lTime_aux[0:230]

    #Channel A process.
    lChannelA = []
    lChannelA = channelA_Real.copy()
    del lChannelA[(samples*2)+77:len(lChannelA)] #Half-period up.
    lChannelA_aux = []
    lChannelA_aux = lChannelA.copy()
    del lChannelA_aux[0:230]

    #Channel B process.
    lChannelB = []
    lChannelB = channelB_Real.copy()
    del lChannelB[(samples*2)+77:len(lChannelB)] #Half-period up.
    lChannelB_aux = []
    lChannelB_aux = lChannelB.copy()
    del lChannelB_aux[0:230]

    return (lTime_aux, lChannelA_aux, lChannelB_aux)    

def polynomial_Approximation(timeReal_half, channelB_Real_half, grade):
    modelo = np.poly1d(np.polyfit(timeReal_half, channelB_Real_half, grade))
    polyline = np.linspace(timeReal_half[0], timeReal_half, len(timeReal_half))

    modelo_poly = []
    modelo_poly = modelo(polyline)
    agrupation_modelo_poly = modelo_poly[len(timeReal_half)-1]
    #print(agrupation_modelo_poly)
    c = channelB_Real_half
    #print(c)

    l_agrupation_modelo_poly = agrupation_modelo_poly.tolist() #List conversion
    subtract = [t1-t2 for t1, t2 in zip(channelB_Real_half, l_agrupation_modelo_poly)]
    #print(subtract)

    return(subtract, polyline, modelo)

def find_Intersection(timeReal_half, channelA_half, channelB_half, correctedSignal):
    x_axis_Real, y_axis_Real = intersection(timeReal_half, channelA_half, timeReal_half, channelB_half) #Intersection original signal section (ChannelB)
    x_axis_corrected, y_axis_corrected = intersection(timeReal_half, channelA_half, timeReal_half, correctedSignal) #Intersection corrected signal section (polynomial)
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

def peaks_section_channel(timeReal, channelA, channelB, corrected_Signal):
    peaks_A, _ = find_peaks(channelA, height = 7.8) #Channel A
    l_peaks_A = []
    l_peaks_A = peaks_A.tolist()

    peaks_Time_A = []
    for s in l_peaks_A:
        peaks_Time_A.append(timeReal[s])
    #print(peaks_Time_A)
    #print(len(peaks_Time_A))

    peaks_channelA = []
    for r in l_peaks_A:
        peaks_channelA.append(channelA[r])

    peaks_B, _ = find_peaks(channelB, height = 4, distance = 15) #Channel B
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
    
def where_ZC(corrected_Signal, timeReal_half, channelB_Real_half):
    #First - Know where zero crossings occur
    zero_cross_CS = np.where(np.diff(np.sign(corrected_Signal)))[0]
    zeroCross_NP_CS = []
    zeroCross_NP_CS = zero_cross_CS.tolist() #Channel B Corrected signal.

    zero_cross_CB = np.where(np.diff(np.sign(channelB_Real_half)))[0]
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

    ###ChannelB
    zeroCross_aux_def_CB = []
    zeroCross_aux_def_CB = zeroCross_NP_CB.copy()

    laux_zeroCross_def_CB = []
    for t in range(len(zeroCross_NP_CB)):
        laux_zeroCross_def_CB.append(zeroCross_NP_CB[t]+1)
        zeroCross_NP_CB.append(zeroCross_NP_CB[t]+1)
    #print(zeroCross_NP_CB)

    zeroCross_NP_CB.sort()
    #print(zeroCross_NP_CB)
    #####

    return (zeroCross_NP_CS, zeroCross_NP_CB)

def where_ZC_toInterpolation(zeroCross_def_CB, zeroCross_def_CS, corrected_Signal, TimeReal_half, channelB_Real_half):
    l_CS_ZC = [] #Corrected signal
    for y in zeroCross_def_CS:
        l_CS_ZC.append(corrected_Signal[y])

    l_CB_ZC = [] #Channel B signal.
    for h in zeroCross_def_CB:
        l_CB_ZC.append(channelB_Real_half[h])

    l_Time_CS_ZC = [] #Corrected signal
    for q in zeroCross_def_CS:
        l_Time_CS_ZC.append(TimeReal_half[q])

    l_Time_CB_ZC = [] #ChannelB signal.
    for d in zeroCross_def_CB:
        l_Time_CB_ZC.append(TimeReal_half[d])
    #print(l_Time_CB_ZC)

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

    """###
    print("---")
    laux_Interpolation_CS_orig = []
    for f in range(len(laux_Interpolation_CS)-1):
        laux_Interpolation_CS_orig.append(laux_Interpolation_CS[f+1] - laux_Interpolation_CS[f])
    print(laux_Interpolation_CS_orig)
    print(len(laux_Interpolation_CS_orig))

    
    value_IntervalTime_CS = sum(laux_Interpolation_CS_orig)
    print(value_IntervalTime_CS)
    twotimes_value_IntervalTime_CS = 2*(value_IntervalTime_CS)
    Period_wave_CS = (twotimes_value_IntervalTime_CS)/(len(laux_Interpolation_CS_def)-1)
    print(Period_wave_CS)
    ###

    ###
    laux_Interpolation_CB_orig = []
    for c in range(len(laux_Interpolation_CB)-1):
        laux_Interpolation_CB_orig.append(laux_Interpolation_CB[c+1]-laux_Interpolation_CB[c])
    print(laux_Interpolation_CB_orig)
    print(len(laux_Interpolation_CB_orig))

    
    value_IntervalTime_CB = sum(laux_Interpolation_CB_orig)
    print(value_IntervalTime_CB)
    twotimes_value_IntervalTime_CB = 2*(value_IntervalTime_CB)
    Period_wave_CB = twotimes_value_IntervalTime_CB/(len(laux_Interpolation_CB_def)-1)
    print(len(laux_Interpolation_CB_orig))
    val = ((len(laux_Interpolation_CB_orig)/2))
    print(len(laux_Interpolation_CB_def))
    print(len(laux_Interpolation_CB_def)-1)
    print(val)
    print(Period_wave_CB)
    ###
    print("---")"""

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

#Sinusoid function
def sinfunc(t, A, w, p, c):
    return A*np.sin(w*t + p) + c

def fft_method(ChannelB_half, samples):
    samplingFrequency = samples/T_halfPeriod
    Y_k = np.fft.fft(ChannelB_half)[0:int(samples/2)]/samples
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
    
def curveFitting_sineFunction(TimeReal, corrected_Signal,):
    tt = np.array(TimeReal)
    yy = np.array(corrected_Signal)
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

def calculation(l_process_IntervalTime_ZC, l_process_IntervalTime_CB, l_Time_Interpolation_CS, l_Time_Interpolation_CB, peaks_TimeB_1m_CP, peaks_Time_B, intersect_axisX, peaks_Time_A, f_CF):
    if (len(peaks_Time_B) <= 29 and len(peaks_TimeB_1m_CP) <= 15 and len(intersect_axisX) <= 5):
        print("Caso 1 meter")
        l_ParticularCase=[]
        for t in range(len(peaks_Time_A)-1):
            l_ParticularCase.append((peaks_Time_A[t+1]- peaks_Time_A[t])-15)
        value_IntervalTime_mean = statistics.mean(l_ParticularCase)
        #print(value_IntervalTime_mean)
        freq_PC = 1/(value_IntervalTime_mean)*1000
        l_value_freq_PC_CF = []
        l_value_freq_PC_CF.append(freq_PC)
        l_value_freq_PC_CF.append(f_CF)
        #print(l_value_freq_PC_CF)
        l_frequencies_aux_difMethods = []
        for g in range(len(l_value_freq_PC_CF)):
            distance = (speed_ligth*T*l_value_freq_PC_CF[g])/(2*2*deltaFreq)
            l_frequencies_aux_difMethods.append(distance)
        #print("Distance particular case: " + str(l_frequencies_aux_difMethods[0]))
        #print("Distance with Curve fitting method: " + str(l_frequencies_aux_difMethods[1]))
        return(l_frequencies_aux_difMethods)
        
    elif (len(peaks_Time_B) >= 30):
        print("Caso different to 1 meter")
        ###Corrected signal - second version
        value_len_l_CS = len(l_Time_Interpolation_CS)
        time_CS = l_Time_Interpolation_CS[value_len_l_CS - 1] - l_Time_Interpolation_CS[0]
        freq_CS = 0.5*((value_len_l_CS)-1)/(time_CS*1e-3)

        ###ChannelB - second version
        value_len_l_CB = len(l_Time_Interpolation_CB)
        time_CB = l_Time_Interpolation_CB[value_len_l_CB - 1] - l_Time_Interpolation_CB[0]
        try:
            freq_CB = 0.5*((value_len_l_CB)-1)/(time_CB*1e-3)
        except ZeroDivisionError:
            freq_CB = 0
        #print(freq_CS)
        #print(freq_CB)
        #print(f_CF)

        """#Corrected signal - first version
        mean_TimeInterval = statistics.mean(l_process_IntervalTime_ZC)
        freq_CS_1v = 1/(2*mean_TimeInterval)*1000

        #ChannelB - first version
        mean_TimeInterval_CB = statistics.mean(l_process_IntervalTime_CB)
        freq_ZC_CB_1v = 1/(2*mean_TimeInterval_CB)*1000"""
        
        l_value_freq = []
        l_value_freq.append(freq_CS)
        l_value_freq.append(freq_CB)
        #l_value_freq.append(freq_CS_1v)
        #l_value_freq.append(freq_ZC_CB_1v)
        l_value_freq.append(f_CF)
        l_frequencies_aux_difMethods = []
        for m in range(len(l_value_freq)):
            distance = (speed_ligth*T*l_value_freq[m])/(2*2*deltaFreq)
            l_frequencies_aux_difMethods.append(distance)
        #print("Distance ZC - second version (Corrected signal): " + str(l_frequencies_aux_difMethods[0]))
        #print("Distance ZC - second version (ChannelB): " + str(l_frequencies_aux_difMethods[1]))
        #print("Distance ZC - first version (Corrected signal): " + str(l_frequencies_aux_difMethods[2]))
        #print("Distance ZC - first version (ChannelB): " + str(l_frequencies_aux_difMethods[3]))
        #print("Distance curve fitting method: " + str(l_frequencies_aux_difMethods[2]))
        #return (l_frequencies_aux_difMethods[0], l_frequencies_aux_difMethods[1], l_frequencies_aux_difMethods[2])

        return(l_frequencies_aux_difMethods)
        
#Check: -> Adapt to script.
def representation_Figure(lTime_aux, lChannelA_aux, lChannelB_aux, subtract_list, polyline, modelo):
    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)', color = color)
    ax1.plot(lTime_aux, lChannelA_aux, color = color, label = "Half period - Channel A")
    ax1.tick_params(axis = 'y', labelcolor = color)
    ax1.legend(loc = "best")

    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Voltage (mV)', color = color)
    ax2.plot(lTime_aux, lChannelB_aux, color = color, label = "Half period - Channel B")
    ax2.plot(lTime_aux, subtract_list, color = 'purple', label = "Corrected signal - Channel B")
    #ax2.plot(tt, sinfunc(tt, *popt), color = 'red', label = "Sine curve fitting")
    ax2.plot(polyline, modelo(polyline), color = 'green')
    ax2.tick_params(axis = 'y', labelcolor = color)
    ax2.legend(loc = "best")

    plt.tight_layout()
    plt.grid()
    #plt.ion()
    plt.show()

########################################################################################
########################################################################################

def procesado_conjunto_Average():
    l_distances_aux = []
    l_filename = [file1, file2, file3, file4, file5, file6, file7, file8, file9, file10]
    for i in range(len(l_filename)):
        #Call function
        (Time, channelA, channelB) = read_csv(l_filename[i])

        samples = samples_IntervalTime(Time)

        (Time_halfP, channelA_halfP, channelB_halfP) = processing_signal_section(Time, channelA, channelB, samples)

        (subtract_CS, polyline, model) = polynomial_Approximation(Time_halfP, channelB_halfP, 4)

        x_axisR = find_Intersection(Time_halfP, channelA_halfP, channelB_halfP, subtract_CS)

        (peaks_Time_A, peaks_Time_B) = peaks_section_channel(Time, channelA, channelB, subtract_CS)

        peaks_Time_B_CP = peaks_section_channel_ParticularCase(Time, channelB)
        
        (zeroCross_CS, zeroCross_CB) = where_ZC(subtract_CS, Time_halfP, channelB_halfP)

        (l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc) = where_ZC_toInterpolation(zeroCross_CB, zeroCross_CS, subtract_CS, Time_halfP, channelB_halfP)
    
        (l_interpolation_CS, l_interpolation_CB) = Interpolation_process(l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc)

        (intervalTime_ZC_CS, intervalTime_ZC_CB) = method_ZC_IntervalTime(l_interpolation_CS, l_interpolation_CB)

        (freq_CV_sineFunction, freq_CV_R) = curveFitting_sineFunction(Time_halfP, subtract_CS)

        l_distances = calculation(intervalTime_ZC_CS, intervalTime_ZC_CB, l_interpolation_CS, l_interpolation_CB, peaks_Time_B_CP, peaks_Time_B, x_axisR, peaks_Time_A, freq_CV_sineFunction)
    
        l_distances_aux.append(l_distances)

    #A partir de aqui --> Posibilidad de dividir el modulo.
    
    laux_distance_1 = [] #Lista auxiliar
    laux_distance_2 = [] #Lista auxiliar
    laux_distance_3 = [] #Lista auxiliar
    laux_distance_2type = [] #Lista auxiliar.

    for m in range(len(l_distances_aux)):
        #print(l_distances_aux[m])
        if len(l_distances_aux[0]) >= 3:
            laux_distance_1.append(l_distances_aux[m][0])
            laux_distance_2.append(l_distances_aux[m][1])
            laux_distance_3.append(l_distances_aux[m][2])
            ###
            average_distance1 = statistics.mean(laux_distance_1)
            average_distance2 = statistics.mean(laux_distance_2)
            average_distance3 = statistics.mean(laux_distance_3)
            ###
            std_deviation1 = np.std(laux_distance_1)
            std_deviation2 = np.std(laux_distance_2)
            std_deviation3 = np.std(laux_distance_3)
        else:
            laux_distance_1.append(l_distances_aux[m][0])
            laux_distance_2.append(l_distances_aux[m][1])
            ###
            average_distance1 = statistics.mean(laux_distance_1)
            average_distance2 = statistics.mean(laux_distance_2)
            average_distance3 = "There is no distance in this mode"
            ###
            std_deviation1 = np.std(laux_distance_1)
            std_deviation2 = np.std(laux_distance_2)
            std_deviation3 = "There is no distance in this mode"
            
    print("Average distance 1 : " + str(average_distance1))
    print("Average distance 2 : " + str(average_distance2))
    print("Average distance 3 : " + str(average_distance3))
    ##
    print("Standar deviation distance 1 : " + str(std_deviation1))
    print("Standar deviation distance 2 : " + str(std_deviation2))
    print("Standar deviation distance 3:  " + str(std_deviation3))

def procesado_selectivo_instant_andRepresentation(file):
    #Call functions
    (Time, channelA, channelB) = read_csv(file)

    samples = samples_IntervalTime(Time)

    (Time_halfP, channelA_halfP, channelB_halfP) = processing_signal_section(Time, channelA, channelB, samples)

    (subtract_CS, polyline, model) = polynomial_Approximation(Time_halfP, channelB_halfP, 4)

    x_axisR = find_Intersection(Time_halfP, channelA_halfP, channelB_halfP, subtract_CS)

    (peaks_Time_A, peaks_Time_B) = peaks_section_channel(Time, channelA, channelB, subtract_CS)

    peaks_Time_B_CP = peaks_section_channel_ParticularCase(Time, channelB)
        
    (zeroCross_CS, zeroCross_CB) = where_ZC(subtract_CS, Time_halfP, channelB_halfP)

    (l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc) = where_ZC_toInterpolation(zeroCross_CB, zeroCross_CS, subtract_CS, Time_halfP, channelB_halfP)
    
    (l_interpolation_CS, l_interpolation_CB) = Interpolation_process(l_CS_zc, Time_CS_zc, l_CB_zc, Time_CB_zc)

    (intervalTime_ZC_CS, intervalTime_ZC_CB) = method_ZC_IntervalTime(l_interpolation_CS, l_interpolation_CB)

    (freq_CV_sineFunction, freq_CV_R) = curveFitting_sineFunction(Time_halfP, subtract_CS)

    l_distances = calculation(intervalTime_ZC_CS, intervalTime_ZC_CB, l_interpolation_CS, l_interpolation_CB, peaks_Time_B_CP, peaks_Time_B, x_axisR, peaks_Time_A, freq_CV_sineFunction)
                    
    representation_Figure(Time_halfP, channelA_halfP, channelB_halfP, subtract_CS, polyline, model)

####################################################################################
##Calls important functions
#procesado_conjunto_Average()
procesado_selectivo_instant_andRepresentation(file1)

                
