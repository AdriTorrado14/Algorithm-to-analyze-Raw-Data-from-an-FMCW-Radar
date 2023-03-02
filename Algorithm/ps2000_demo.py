"""
PS2000 Demo.

By: Colin O'Flynn, based on Mark Harfouche's software

This is a demo of how to use AWG with the Picoscope 2204 along with capture
It was tested with the PS2204A USB2.0 version

The AWG is connected to Channel A.
Nothing else is required.

NOTE: Must change line below to use with "A" and "B" series PS2000 models

See http://www.picotech.com/document/pdf/ps2000pg.en-10.pdf for PS2000 models:
PicoScope 2104
PicoScope 2105
PicoScope 2202
PicoScope 2203
PicoScope 2204
PicoScope 2205
PicoScope 2204A
PicoScope 2205A
"""

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals

from picoscope import ps2000

import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import sys
import pandas as pd
from itertools import accumulate
import os
import glob

from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QDialog, QLabel, QMainWindow)
from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
import tkinter as tk

from modules_ps2000 import *

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
-> class Ayuda_Dialog
Functionality: Menu that appears when we run the script. It shows how to enter the different variables
to do the measurement. When we close this window, it will appear the interface.
"""
class Ayuda_Dialog:
    def __init__(self, parent):
        text = ("Information to be taken into account:\n"
                "\n"
                "To exit the box help and be able to enter the data, you must click \n"
                "the tab 'X'. \n"
                "\n"
                "Different steps: \n"
                "Step 1: Insert the value of the frequency. Use a dot ( . ), instead of a \n"
                "coma ( , ).\n" 
                "Step 2: Insert the value of the amplitude. Use a dot ( . ), instead of a \n"
                "coma ( , ).\n"
                "Step 3: Insert the type of the signal. The different types are: Sine, Square, \n"
                "Triangle, RampUp, RampDown and DCVoltage.\n"
                "\n"
                "To introduce values, we need to select the panel and erase the quote. After \n"
                "that, we can introduce the value of the frequency, amplitude and type \n"
                "of signal. Once we have entered the data, we press the button 'Results' \n"
                "and we can see the results obtained. \n"
                "\n"
                "If we want to change the value, we just need to modified the values and \n"
                "press again the buttom. If we change values, the results will be different \n"
                "Finally, if we don't press the button, the image will dissapear in the \n"
                "following 60 seconds. \n")
        self.top = tk.Toplevel(parent)
        self.top.title("Help")
        display = tk.Text(self.top)
        display.pack()
        display.insert(tk.INSERT, text)
        display.config(state=tk.DISABLED)
        b = tk.Button(self.top, text="Close", command=self.cerrar)
        b.pack(pady=5)

    def cerrar(self):
        self.top.destroy()

class Main_Window:
    def __init__(self,  root):
        root.geometry("200x100")
        tk.Button(root, text="Information",  command = self.ayuda).pack()
    def ayuda(self):
        Ayuda_Dialog(root)
######Text box end code######
#---------------------------------------------------------------------------------------

"""
-> class Form.
Functionality: It shows the interface that we are goint to use to do the measurements. It appears some boxes and we will
use these boxes to enter the value of the frequency, amplitude and type of wave. The script will save the different values.
Then it shows the different modes of use.
"""
class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        #Create widgets.
        #self.frequency = QLineEdit("Insert frequency: ")
        #self.amplitude = QLineEdit("Insert amplitude: ")
        #self.typeSignal = QLineEdit("Insert type of signal: ")
        #self.voltageOffset = QLineEdit("Insert offset voltage: ")
        self.number_instants = QLineEdit("Insert number of instants to create the measurement: ")
        self.instant_to_Measure = QLineEdit("Insert number of instant to analyze: ")
        self.question_analyze = QLineEdit("Analyze all together: ")
        #self.gradePolynomial = QLineEdit("Insert number of polynomial degree: ")
        
        self.button1 = QPushButton("First mode")
        self.button2 = QPushButton("Second mode - Real Time")
        self.button3 = QPushButton("End simulation")

        #Create layout and add widgets
        layout = QVBoxLayout()
        #layout.addWidget(self.frequency)
        #layout.addWidget(self.amplitude)
        #layout.addWidget(self.typeSignal)
        #layout.addWidget(self.voltageOffset)
        layout.addWidget(self.number_instants)
        layout.addWidget(self.instant_to_Measure)
        layout.addWidget(self.question_analyze)
        #layout.addWidget(self.gradePolynomial)

        layout.addWidget(self.button1)
        layout.addWidget(self.button2)
        layout.addWidget(self.button3)

        #Set dialog layout.
        self.setLayout(layout)

        #Add button signal to greetings slot.
        self.button1.clicked.connect(self.first_mode)
        self.button2.clicked.connect(self.second_mode_LiveTime)
        self.button3.clicked.connect(self.stopProcess)

#"""
#Functionality: Interface to enter some values and variables to do the measurement. We need to
#enter: frequency, amplitude, type of wave, number of instant to measure, if we want to do all the
#processing at the same time. Also, we introduce some exceptions to control fails into the script.
#"""

    def EnterData(self):
        #Collection and allocation of information collected by the interface.
        try:
            """#Frequency
            valor = float(self.frequency.text())
            valor_frequency = valor
            #Amplitude
            valorA = float(self.amplitude.text())
            valorAmplitude = valorA
            #Type signal
            typeSignal_D = str(self.typeSignal.text())
            tipoSenal = typeSignal_D
            #Offset voltage
            offset_voltage = float(self.voltageOffset.text())
            value_offset_V = offset_voltage"""
            #Number of instants.
            numInstant = int(self.number_instants.text())
            value_numInstant = numInstant
            #Instants to measure.
            numInstantToMeasure = int(self.instant_to_Measure.text())
            instantToMeasure = numInstantToMeasure
            #Question to analyze.
            answer_analyze = str(self.question_analyze.text())
            answer_analyze_measurement = answer_analyze
            """#Grade polynomial
            num_Poly_approximation = float(self.gradePolynomial())
            grade_Poly_approximation = num_Poly_approximation"""
            
        except ValueError:
            print("In the panel 'Insert Frequency' and 'Insert Amplitude' you must introduce a float type value. Please, you must use the dot ( . ), instead coma ( , ).")
            os._exit(1)
        if value_numInstant < instantToMeasure:
            print("Instant to measure has to be bigger than the number of instants. Please, introduce the number correctly.")
            os._exit(1)
        #if (answer_analyze != "YES" or answer_analyze != "yes" or answer_analyze != "Yes" or answer_analyze != "NO" or answer_analyze != "no" or answer_analyze != "No" or answer_analyze != "Save Mode" or answer_analyze != "SAVE MODE" or answer_analyze != "Save mode"):
            #print("Incorrect answer")
            #os._exit(1)
        #return (valor_frequency, valorAmplitude, tipoSenal, wave_Duration)
        #return (valor_frequency, valorAmplitude, tipoSenal, value_vOffset, value_numInstant)
        return (value_numInstant, instantToMeasure, answer_analyze)

#"""
#Functionality: Button to active first mode of the script. This mode has two different parts or sections. We will
#selected the section trougth the interface. In the first mode, we will do the measurement and then we will save
#these signal into a csv fie. In the second mode, we will analyze these measurements and we will get a distance from
#the different signals. We can select if we want to do the processing at the same time or just analyze one instant.
#"""
    def first_mode(self):
        value_numInstant, value_InstantToMeasure, value_answer = self.EnterData()

        #print(__doc__)

        print("Attempting to open Picoscope 2000...")

        ps=ps2000.PS2000()

        print("Found the following picoscope:")
        print(ps.getAllUnitInfo())
    
        #waveform_desired_duration = 0.6144
        waveform_desired_duration = 0.3072
        obs_duration = waveform_desired_duration
        #obs_duration = 50 * waveform_desired_duration
        #print("Observation duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        #print("Sampling interval 2: " + str(sampling_interval))

        (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        #print(actualSamplingInterval)
        #print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9))
        #print("Taking  samples = %d" % nSamples)
        #print("Maximum samples = %d" % maxSamples)

        ###### Data for channelA ######
        channelA = 'A'
        coupling_channelA = 'DC'
        #VRange_channelA = 10.0
        VRange_channelA = 20.0
        #VOffset_channelA = 0.0 #Original laboratorio
        VOffset_channelA = 0
        enabled = True
        BWLimited = False
        #probeAttenuation = 1.0
        probeAttenuation = 10.0
        channelRange_A = ps.setChannel(channelA, coupling_channelA, VRange_channelA, VOffset_channelA, enabled, BWLimited, probeAttenuation)

        ###### Data for channelB ######
        channelB = 'B'
        coupling_channelB = 'AC'
        VRange_channelB = 200e-3
        VOffset_channelB = 0.0
        channelRange_B = ps.setChannel(channelB, coupling_channelB, VRange_channelB, VOffset_channelB, enabled, BWLimited, probeAttenuation= 1.0)

        ###### Signal generator ######
        #ps.setSigGenBuiltInSimple(offsetVoltage = 550e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20) #Input: 1V / 10V
        #ps.setSigGenBuiltInSimple(offsetVoltage = 500e-3, pkToPk = 600e-3, waveType="Triangle", frequency=20) #Input: 2V / 8V
        #ps.setSigGenBuiltInSimple(offsetVoltage = 465e-3, pkToPk = 540e-3, waveType="Triangle", frequency=20)
        ps.setSigGenBuiltInSimple(offsetVoltage = 500e-3, pkToPk = 540e-3, waveType="Triangle", frequency=20)

        dataTimeAxis = np.arange(nSamples) * actualSamplingInterval
        lTime = []
        lTime = dataTimeAxis.tolist()

        lista_auxA = []
        lista_auxB = []
        lista_dataA_aux = []
        lista_dataB_aux = []

        l_agrupation = []

        for i in range(value_numInstant):
            ps.runBlock()
            ps.waitReady() #After this. "Waiting for awg to settle".
            time.sleep(0.05)
            ps.runBlock()
            ps.waitReady() #After this. "Done waiting for trigger".
            dataA = ps.getDataV('A', nSamples, returnOverflow=False)
            dataB = ps.getDataV('B', nSamples, returnOverflow=False)

            ###### DataA ######
            lista_dataA_aux = dataA.tolist() #Lista dataA
            lista_auxA.append(lista_dataA_aux)

            ###### DataB ######
            lista_dataB_aux = dataB.tolist() #Lista dataB
            lista_auxB.append(lista_dataB_aux)
        ps.stop()
        ps.close()

        for g, f in zip(range(len(lista_auxA)), range(len(lista_auxB))):
            l_agrupation.append(lTime)
            l_agrupation.append(lista_auxA[g])
            l_agrupation.append(lista_auxB[f])

        half2 = int(len(l_agrupation)/3)

        length_to_split2 = [len(l_agrupation)//half2]*half2
        Output2 = [l_agrupation[x - y: x] for x, y in zip(
            accumulate(length_to_split2), length_to_split2)]

        ###### Save to csv file ######
        for t in range(len(Output2)):
            l_df = pd.DataFrame(Output2[t]).transpose()
            l_df.to_csv("C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement/file" + str(t) + ".csv", index = False, header = False)

        path = "C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement"

        ###### Analyze all measurement all together ######
        if (value_answer == "No" or  value_answer == "NO" or value_answer == "no"):
            print("Case 1 - Just one of the instants.")

            instant = value_InstantToMeasure
            file_measurement = 'C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement/file' + str(instant) + '.csv'

            name_rows_measurement = ['Time', 'Channel A', 'Channel B']
            df_measurement = pd.read_csv(file_measurement, sep = ',', error_bad_lines = False, names = name_rows_measurement)

            ###### Time ######
            lTime = list(df_measurement['Time'])             

            ###### Channel-A ######
            lChannelA = list(df_measurement['Channel A'])

            ###### Channel-B ######
            lChannelB = list(df_measurement['Channel B'])

            ### NEW MODIFICATION - Add module to analyze just one of the instants (file) ###
            (periods_Time, periods_channelA, periods_channelB, model_full_range, polyline_full_range, Time_mV, channelB_mV, channelA, peaks_Time_B, peaks_channelB) = process_selective_Instant(lTime, lChannelA, lChannelB, file_measurement, actualSamplingInterval)

            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Voltage (V)', color = color)
            ax1.plot(Time_mV, channelA, color = color, label = " Channel A")
            ax1.plot(periods_Time[0], periods_channelA[0], color = 'black', label = "Half-up period")
            ax1.plot(periods_Time[1], periods_channelA[1], color = 'black')
            ax1.plot(periods_Time[2], periods_channelA[2], color = 'black')
            ax1.plot(periods_Time[3], periods_channelA[3], color = 'black')
            ax1.tick_params(axis = 'y', labelcolor = color)
            ax1.legend(loc = "best")

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Voltage (mV)', color = color)
            ax2.plot(Time_mV, channelB_mV, color = color, label = "Channel B")
            ax2.plot(periods_Time[0], periods_channelB[0], color = 'green', label = "Frame to analyze")
            ax2.plot(periods_Time[1], periods_channelB[1], color = 'green')
            ax2.plot(periods_Time[2], periods_channelB[2], color = 'green')
            ax2.plot(periods_Time[3], periods_channelB[3], color = 'green')
            ax2.tick_params(axis = 'y', labelcolor = color)
            ax2.legend(loc = "best")

            plt.tight_layout()
            plt.grid()
            #plt.ion()
            plt.show()

            ### TELEGRAM BOT ###
            today= datetime.datetime.now()
            plt.savefig('picture.png')

            files = {'photo':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/picture.png','rb')}
            resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendPhoto?chat_id=9284473&caption=graph'.format(today), files=files)
            
        elif (value_answer == "Yes" or value_answer == "YES" or value_answer == "yes"):
            print("Caso 2 - All the instants.")
            files = glob.glob(path + "/*.csv")

            ### NEW MODIFICATION - Add module to analyze all of the instants together (file) ###
            (Time_aux, channelA, channelB_aux, periods_Tim, periods_chaB, periods_chaA) = process_Average_simult(files, actualSamplingInterval)
            #process_Average_simult(files, actualSamplingInterval)

            """
            fig, ax1 = plt.subplots()
            color = 'tab:blue'
            ax1.set_xlabel('Time (ms)')
            ax1.set_ylabel('Voltage (V)', color = color)
            ax1.plot(Time_aux[2], channelA[2], color = color, label = " Channel A")
            ax1.plot(periods_Tim[2][0], periods_chaA[2][0], color = 'black')
            ax1.plot(periods_Tim[2][1], periods_chaA[2][1], color = 'black')
            ax1.plot(periods_Tim[2][2], periods_chaA[2][2], color = 'black')
            ax1.plot(periods_Tim[2][3], periods_chaA[2][3], color = 'black')
            ax1.tick_params(axis = 'y', labelcolor = color)
            ax1.legend(loc = "best")

            ax2 = ax1.twinx()
            color = 'tab:orange'
            ax2.set_ylabel('Voltage (mV)', color = color)
            ax2.plot(Time_aux[2], channelB_aux[2], color = color, label = "Channel B")
            ax2.tick_params(axis = 'y', labelcolor = color)
            ax2.legend(loc = "best")

            plt.tight_layout()
            plt.grid()
            #plt.ion()
            plt.show()

            ### TELEGRAM BOT ###
            today= datetime.datetime.now()
            plt.savefig('picture_simul.png')

            files = {'photo':open('C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/TFM_Adri/pico-python-master/examples/picture_simul.png','rb')}
            resp = requests.post('https://api.telegram.org/bot5265235411:AAFA7kzkfGOtbyuOlplq_hvMFSc93O6Msl4/sendPhoto?chat_id=9284473&caption=graph'.format(today), files=files)
            #print(resp.status_code)
            """
            
        elif (value_answer == "Save Mode" or value_answer == "SAVE MODE" or value_answer == "Save mode"):
            print("Case 3 - Save files '.csv' in folder: measurement.")
            os._exit(1)
        
#"""
#Functionality: Button to active second mode of the script. In this mode we show the live mode. It will appear
#two plots. First one is the output signal (channel-B) and the second one is the half-section period (original output)
#and the half-section period (corrected signal with polynomial approimation).
#"""
    def second_mode_LiveTime(self):
        value_numInstant, value_InstantToMeasure, value_answer = self.EnterData()

        #print(__doc__)

        print("Attempting to open Picoscope 2000...")

        ps=ps2000.PS2000()

        print("Found the following picoscope:")
        print(ps.getAllUnitInfo())

        waveform_desired_duration = 0.3072
        obs_duration = waveform_desired_duration
        #obs_duration = 50 * waveform_desired_duration
        #print("Observation duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        #print("Sampling interval 2: " + str(sampling_interval))

        (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        #print(actualSamplingInterval)
        #print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9))
        #print("Taking  samples = %d" % nSamples)
        #print("Maximum samples = %d" % maxSamples)

        ###### Data for channelA ######
        channelA = 'A'
        coupling_channelA = 'DC'
        #VRange_channelA = 10.0
        VRange_channelA = 20.0
        #VOffset_channelA = 0.0 #Original laboratorio
        VOffset_channelA = 0
        enabled = True
        BWLimited = False
        #probeAttenuation = 1.0
        probeAttenuation = 10.0
        channelRange_A = ps.setChannel(channelA, coupling_channelA, VRange_channelA, VOffset_channelA, enabled, BWLimited, probeAttenuation)

        ###### Data for channelB ######
        channelB = 'B'
        coupling_channelB = 'AC'
        VRange_channelB = 200e-3
        VOffset_channelB = 0.0
        channelRange_B = ps.setChannel(channelB, coupling_channelB, VRange_channelB, VOffset_channelB, enabled, BWLimited, probeAttenuation = 1.0)

        ###### Signal generator ######
        #ps.setSigGenBuiltInSimple(offsetVoltage = 550e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20) #Input: 1V / 10V (Solo osciloscopio)
        #ps.setSigGenBuiltInSimple(offsetVoltage = 500e-3, pkToPk = 600e-3, waveType="Triangle", frequency=20) #Input: 2V / 8V (Solo osciloscopio)
        #ps.setSigGenBuiltInSimple(offsetVoltage = 465e-3, pkToPk = 540e-3, waveType="Triangle", frequency=20)
        ps.setSigGenBuiltInSimple(offsetVoltage = 500e-3, pkToPk = 540e-3, waveType="Triangle", frequency=20)

        dataTimeAxis = np.arange(nSamples) * actualSamplingInterval
        lTime = []
        lTime = dataTimeAxis.tolist()

        lista_auxA = []
        lista_auxB = []
        lista_dataA_aux = []
        lista_dataB_aux = []

        l_agrupation = []
        cont = 0
        line1 = []
        h1, = plt.plot([],[])

        l_median = []
        l_median_EC = []
        l_std = []

        try:
            while True:
                ps.runBlock()
                ps.waitReady()
                time.sleep(0.05)
                ps.runBlock()
                ps.waitReady()
                
                dataA = ps.getDataV('A', nSamples, returnOverflow = False)
                dataB = ps.getDataV('B', nSamples, returnOverflow = False)

                ###### DataA ######
                lista_dataA_aux = dataA.tolist() #Lista dataA
                #print(len(lista_dataA_aux))
                #print(lista_dataA_aux)
                lista_auxA.append(lista_dataA_aux)

                ###### DataB ######
                lista_dataB_aux = dataB.tolist() #Lista dataB
                #print(len(lista_dataB_aux))
                #print(lista_dataB_aux)
                lista_auxB.append(lista_dataB_aux)

                ###### Time ######
                dataTimeAxis = np.arange(nSamples) * actualSamplingInterval
                lTime = dataTimeAxis.tolist()
                #print(len(lTime))
                #print(lTime)

                ### ADD NEW MODIFICATION ###
                (median_Distance, median_Distance_EC, std_deviation, periods_Time, periods_channelB, corrected_Signals_periods, Time_mV, channelB_mV) = process_LiveTime(lTime, lista_dataA_aux, lista_dataB_aux, actualSamplingInterval)
                #(periods_Time, periods_channelB, corrected_Signals_periods, Time_mV, channelB_mV) = process_LiveTime(lTime, lista_dataA_aux, lista_dataB_aux, actualSamplingInterval)

                l_median.append(median_Distance)
                print("Distance: " + str(l_median))
                #print(l_median)

                l_median_EC.append(median_Distance_EC)
                print("Distance with error coefficient: " + str(l_median_EC))
                #print(l_median_EC)

                l_std.append(std_deviation)
                print("Standar deviation: " + str(l_std))
                #print(l_std)
                
                if cont > 9:
                    del l_median[0]
                    del l_median_EC[0]
                    del l_std[0]
                else:
                    pass
            
                ###### PLOT LIVE ######
                plt.figure(1)
                plt.subplot(221)
                plt.cla()
                plt.plot(Time_mV, lista_dataA_aux)
                plt.title("Input - Full range")
                plt.ylabel("Voltage (V)")
                plt.xlabel("Time (mseg)\n")

                plt.subplot(222)
                plt.cla()
                plt.plot(Time_mV, channelB_mV, color = 'green')
                plt.title("Output - Full range")
                plt.ylabel("Voltage (mV)")
                plt.xlabel("Time (mseg)")

                plt.subplot(212)
                plt.cla()
                plt.plot(periods_Time[1], periods_channelB[1], color = 'green',label = "Output")
                plt.plot(periods_Time[1], corrected_Signals_periods[1], color = 'orange', label = "Corrected")
                plt.ylabel("Voltage (mV)")
                plt.xlabel("Time (mseg)")
                plt.legend(loc = 'upper right')
                
                plt.tight_layout()
                plt.pause(0.07)
                plt.cla()
                time.sleep(0.01)

                cont = cont+1
                print("Simulation frame: " + str(cont) + "\n")
                
        except KeyboardInterrupt:
            pass

        ps.stop()
        ps.close()
   
        #################### ACABARLO ###################
        print("Ending of Real Time Mode.")

#"""
#Functionality: Button to close the interface and stop the process.
#"""
    def stopProcess(self):
        sys.exit() 

if __name__ == '__main__':
    #Information about enter data.
    root = tk.Tk()
    Main_Window(root)
    root.mainloop()
    
    #Second class 'QApplication'
    app = QApplication(sys.argv)

    #Create and show the form.
    form = Form()
    form.show()

    #Run the main Qt loop.
    sys.exit(app.exec_())


    
