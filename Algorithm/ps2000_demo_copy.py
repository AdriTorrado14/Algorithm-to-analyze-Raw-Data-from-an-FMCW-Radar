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
import numpy as np
import sys
import pandas as pd
from itertools import accumulate
import os
from pylive import live_plotter
import glob

from PySide6.QtWidgets import (QLineEdit, QPushButton, QApplication, QVBoxLayout, QDialog, QLabel, QMainWindow)
import tkinter as tk


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
        
#Text box end code.
#---------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------
class Form(QDialog):
    def __init__(self, parent=None):
        super(Form, self).__init__(parent)
        #Create widgets.
        #self.frequency = QLineEdit("Insert frequency: ")
        #self.amplitude = QLineEdit("Insert amplitude: ")
        #self.typeSignal = QLineEdit("Insert type of signal: ")
        self.number_instants = QLineEdit("Insert number of instants to create the measurement: ")
        self.instant_to_Measure = QLineEdit("Insert number of instant to analyze: ")
        self.question_analyze = QLineEdit("Analyze all together: ")
        #self.gradePolynomial = QLineEdit("Insert number of polynomial degree: ")
        
        self.button2 = QPushButton("First mode")
        self.button3 = QPushButton("Second mode")
        self.button4 = QPushButton("Third mode")
        self.button5 = QPushButton("End simulation")

        #Create layout and add widgets
        layout = QVBoxLayout()
        #layout.addWidget(self.frequency)
        #layout.addWidget(self.amplitude)
        #layout.addWidget(self.typeSignal)
        layout.addWidget(self.number_instants)
        layout.addWidget(self.instant_to_Measure)
        layout.addWidget(self.question_analyze)
        #layout.addWidget(self.gradePolynomial)

        layout.addWidget(self.button2)
        layout.addWidget(self.button3)
        layout.addWidget(self.button4)
        layout.addWidget(self.button5)

        #Set dialog layout.
        self.setLayout(layout)

        #Add button signal to greetings slot.
        self.button2.clicked.connect(self.mode1)
        self.button3.clicked.connect(self.mode2)
        self.button4.clicked.connect(self.mode3)
        self.button5.clicked.connect(self.stopProcess)

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
            tipoSenal = typeSignal_D"""
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
        if (answer_analyze != "YES" or answer_analyze != "yes" or answer_analyze != "Yes" or answer_analyze != "NO" or answer_analyze != "no" or answer_analyze != "No" or answer_analyze != "Save Mode" or answer_analyze != "SAVE MODE" or answer_analyze != "Save mode"):
            print("Incorrect answer")
            os._exit(1)
        #return (valor_frequency, valorAmplitude, tipoSenal, wave_Duration)
        #return (valor_frequency, valorAmplitude, tipoSenal, value_vOffset, value_numInstant)
        return (value_numInstant, instantToMeasure, answer_analyze)
        
############################################################################################
    def mode1(self):
        value_numInstant, value_InstantToMeasure, value_answer = self.EnterData()

        print(__doc__)

        print("Attempting to open Picoscope 2000...")

        ps=ps2000.PS2000()

        print("Found the following picoscope:")
        print(ps.getAllUnitInfo())

        waveform_desired_duration = 0.6144
        obs_duration = waveform_desired_duration
        #obs_duration = 50 * waveform_desired_duration
        print("Observation duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        print("Sampling interval 2: " + str(sampling_interval))

        (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        print(actualSamplingInterval)
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9))
        print("Taking  samples = %d" % nSamples)
        print("Maximum samples = %d" % maxSamples)

        #####
        #Data for channelA
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

        #####
        #Data for channelB
        channelB = 'B'
        coupling_channelB = 'AC'
        VRange_channelB = 100e-3
        VOffset_channelB = 0.0
        #channelRange_B = ps.setChannel(channelB, coupling_channelB, VRange_channelB, VOffset_channelB, enabled, BWLimited, probeAttenuation)

        #####
        #Signal generator
        #ps.setSigGenBuiltInSimple(offsetVoltage= 530e-3, pkToPk=0.85, waveType="Triangle", frequency=20) #ChannelA para laboratorio (Intentar aproximarlo mas: 1v-10V).
        #ps.setSigGenBuiltInSimple(offsetVoltage=740e-3, pkToPk=0.6, waveType="Triangle", frequency=20)
        ps.setSigGenBuiltInSimple(offsetVoltage = 150e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20)

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
            ###DataA###
            lista_dataA_aux = dataA.tolist() #Lista dataA
            lista_auxA.append(lista_dataA_aux)
            ###DataB###
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

        ###
        #Save to csv file.
        for t in range(len(Output2)):
            l_df = pd.DataFrame(Output2[t]).transpose()
            l_df.to_csv("C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement/file" + str(t) + ".csv", index = False, header = False)

        ###
        #Representation to see if the data is correct.
        plt.figure()
        plt.plot(lTime, lista_auxA[2], label="Clock")
        plt.grid()
        plt.show()

############################################################################################
    def mode2(self):
        value_numInstant, value_InstantToMeasure, value_answer = self.EnterData()

        print(__doc__)

        print("Attempting to open Picoscope 2000...")

        ps=ps2000.PS2000()

        print("Found the following picoscope:")
        print(ps.getAllUnitInfo())

        answer = value_answer
        print(answer)
        print(type(answer))

        waveform_desired_duration = 0.6144
        obs_duration = waveform_desired_duration
        #obs_duration = 50 * waveform_desired_duration
        print("Observation duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        print("Sampling interval 2: " + str(sampling_interval))

        (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        print(actualSamplingInterval)
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9))
        print("Taking  samples = %d" % nSamples)
        print("Maximum samples = %d" % maxSamples)

        #####
        #Data for channelA
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

        #####
        #Data for channelB
        channelB = 'B'
        coupling_channelB = 'AC'
        VRange_channelB = 100e-3
        VOffset_channelB = 0.0
        #channelRange_B = ps.setChannel(channelB, coupling_channelB, VRange_channelB, VOffset_channelB, enabled, BWLimited, probeAttenuation)

        #####
        #Signal generator
        #ps.setSigGenBuiltInSimple(offsetVoltage= 530e-3, pkToPk=0.85, waveType="Triangle", frequency=20) #ChannelA para laboratorio (Intentar aproximarlo mas: 1v-10V).
        #ps.setSigGenBuiltInSimple(offsetVoltage=740e-3, pkToPk=0.6, waveType="Triangle", frequency=20)
        ps.setSigGenBuiltInSimple(offsetVoltage = 150e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20)

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
            ###DataA###
            lista_dataA_aux = dataA.tolist() #Lista dataA
            lista_auxA.append(lista_dataA_aux)
            ###DataB###
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

        ###
        #Save to csv file.
        for t in range(len(Output2)):
            l_df = pd.DataFrame(Output2[t]).transpose()
            l_df.to_csv("C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement/file" + str(t) + ".csv", index = False, header = False)

            #fichero = 'file' + str(t) + '.csv'
            #nombre_columnas = ['Time', 'dataA', 'dataB']
            #df = pd.read_csv(fichero, sep = ',', error_bad_lines = False, names = nombre_columnas)

        path = "C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement"

        ###Analyze all measurement all together
        if (value_answer == "No" or  value_answer == "NO" or value_answer == "no"):
            print("Caso 1")
            instant = value_InstantToMeasure
            file_measurement = 'C:/Users/adrit/AppData/Local/Programs/Python/Python37/Scripts/measurements/measurement/file' + str(instant) + '.csv'

            name_rows_measurement = ['Time', 'Channel A', 'Channel B']
            df_measurement = pd.read_csv(file_measurement, sep = ',', error_bad_lines = False, names = name_rows_measurement)

            print(file_measurement)

            ###Time
            lTime = list(df_measurement['Time'])
            print(len(lTime))

            ###Channel A
            lChannelA = list(df_measurement['Channel A'])
            print(len(lChannelA))

            ###Channel B
            lChannelB = list(df_measurement['Channel B'])
            print(len(lChannelB))

        elif (value_answer == "Yes" or value_answer == "YES" or value_answer == "yes"):
            print("Caso 2")
            files = glob.glob(path + "/*.csv")
            print(files)
            name_rows_measurement = ['Time', 'Channel A', 'Channel B']
            list
            list_data_LTime = []
            list_data_lChannelA = []
            list_data_lChannelB = []
            for t in range(len(files)):
                print(files[t])
                df_measu = pd.read_csv(files[t], sep = ',', error_bad_lines = False, names = name_rows_measurement)
                list_data_LTime.append(list(df_measu['Time']))
                list_data_lChannelA.append(list(df_measu['Channel A']))
                list_data_lChannelB.append(list(df_measu['Channel B']))

        elif (value_answer == "Save Mode" or value_answer == "SAVE MODE" or value_answer == "Save mode"):
            print("Just save csv file mode in folder: measurement.")
            os._exit(1)

        """###
        #Representation to see if the data is correct.   
        plt.figure()
        plt.plot(lTime, lista_auxA[instant], label="Clock")
        plt.grid()
        plt.show()"""
        
############################################################################################
    def mode3(self):
        value_numInstant, value_InstantToMeasure = self.EnterData()

        print(__doc__)

        print("Attempting to open Picoscope 2000...")

        ps=ps2000.PS2000()

        print("Found the following picoscope:")
        print(ps.getAllUnitInfo())

        waveform_desired_duration = 0.6144
        obs_duration = waveform_desired_duration
        #obs_duration = 50 * waveform_desired_duration
        print("Observation duration: " + str(obs_duration))
        sampling_interval = obs_duration / 4096
        print("Sampling interval 2: " + str(sampling_interval))

        (actualSamplingInterval, nSamples, maxSamples) = ps.setSamplingInterval(sampling_interval, obs_duration)
        print(actualSamplingInterval)
        print("Sampling interval = %f ns" % (actualSamplingInterval * 1E9))
        print("Taking  samples = %d" % nSamples)
        print("Maximum samples = %d" % maxSamples)

        #####
        #Data for channelA
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

        #####
        #Data for channelB
        channelB = 'B'
        coupling_channelB = 'AC'
        VRange_channelB = 100e-3
        VOffset_channelB = 0.0
        #channelRange_B = ps.setChannel(channelB, coupling_channelB, VRange_channelB, VOffset_channelB, enabled, BWLimited, probeAttenuation)

        #####
        #Signal generator
        #ps.setSigGenBuiltInSimple(offsetVoltage= 530e-3, pkToPk=0.85, waveType="Triangle", frequency=20) #ChannelA para laboratorio (Intentar aproximarlo mas: 1v-10V).
        #ps.setSigGenBuiltInSimple(offsetVoltage=740e-3, pkToPk=0.6, waveType="Triangle", frequency=20)
        ps.setSigGenBuiltInSimple(offsetVoltage = 150e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20)

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

        try:
            while True:
                #print("Inicio")
                ps.runBlock()
                ps.waitReady()
                time.sleep(0.05)
                ps.runBlock()
                ps.waitReady()
                dataA = ps.getDataV('A', nSamples, returnOverflow = False)
                dataB = ps.getDataV('B', nSamples, returnOverflow = False)
                ###DataA###
                lista_dataA_aux = dataA.tolist() #Lista dataA
                lista_auxA.append(lista_dataA_aux)
                ###DataB###
                lista_dataB_aux = dataB.tolist() #Lista dataB
                lista_auxB.append(lista_dataB_aux)
                ###Time###
                dataTimeAxis = np.arange(nSamples) * actualSamplingInterval
                lTime = dataTimeAxis.tolist()

                ##PLOT LIVE
                
                cont = cont+1
                print(cont)
                    
        except KeyboardInterrupt:
            pass
        ps.stop()
        ps.close()

        ####################ACABARLO###################

        print(len(lista_auxA))
        print(len(lista_auxA[0]))

        print(len(lista_auxB))
        print(len(lista_auxB[0]))

        valorA = len(lista_auxA)
        print(valorA)

        valorB = len(lista_auxB)
        print(valorB)
        
        lista_auxA.pop()
        print(len(lista_auxA))

        lista_auxB.pop()
        print(len(lista_auxB))

        #print(lista_auxA)
        #print(lista_auxB)
        #print(lTime)

        """#Representation    
        plt.figure()
        plt.plot(lTime, lista_auxA[3], label="Clock")
        plt.grid()
        plt.show()"""

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


    
