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

"""#if __name__ == "__main__":
    print(__doc__)

    print("Attempting to open Picoscope 2000...")

    ps=ps2000.PS2000()

    print("Found the following picoscope:")
    print(ps.getAllUnitInfo())

    #waveform_desired_duration = 50e-3
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

    #(sampleRate, maxSamples) = ps.setSamplingFrequency(sampleFrequency =, noSamples =, oversample=0, segmentIndex =)
    #print(sampleRate)
    #print(maxSamples)

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
    #Set simple Trigger
    #ps.setSimpleTrigger('A', 1.0, 'Falling', timeout_ms=100, enabled=True)

    #####
    #Signal generator
    #ps.setSigGenBuiltInSimple(offsetVoltage= 530e-3, pkToPk=0.85, waveType="Triangle", frequency=20) #ChannelA para laboratorio (Intentar aproximarlo mas: 1v-10V).
    #ps.setSigGenBuiltInSimple(offsetVoltage=740e-3, pkToPk=0.6, waveType="Triangle", frequency=20)
    ps.setSigGenBuiltInSimple(offsetVoltage = 150e-3, pkToPk = 900e-3, waveType="Triangle", frequency=20)
    
    numberTimes = 4
    instant = 2
    
    lista_auxA = []
    lista_auxB = []
    lista_dataA_aux = []
    lista_dataB_aux = []

    l_agrupacion = []

    dataTimeAxis = np.arange(nSamples) * actualSamplingInterval
    lTime = []
    lTime = dataTimeAxis.tolist()

    l_agrupation = []
    
    print("Inicio bucle")
    for i in range(numberTimes):
        ps.runBlock()
        #ps.waitReady() #After this. "Waiting for awg to settle".
        #time.sleep(0.2)
        #ps.runBlock()
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
    print("Fin bucle")

    #print(lista_auxA)
    print(len(lista_auxA))
    print(len(lista_auxA[0]))

    #print(lista_auxB)
    print(len(lista_auxB))
    print(len(lista_auxB[0]))

cont = 0
l_agrupation_raw = []
try:
    while True:
        #print("Inicio")
        ps.runBlock()
        ps.waitReady()
        time.sleep(0.05)
        ps.runBlock()
        ps.waitReady()
        dataA = ps.getDataV('A', nSamples, returnOverflow = False)
        ###DataA###
        lista_dataA_aux = dataA.tolist() #Lista dataA
        lista_auxA.append(lista_dataA_aux)
        cont = cont+1
        print(cont)
except KeyboardInterrupt:
    pass

ps.stop()
ps.close()

print(len(lista_auxA))
print(len(lista_auxA[0]))

valor = len(lista_auxA)
print(valor)

lista_auxA.pop()
print(len(lista_auxA))

#print(len(lista_auxA[valor-1]))


def stopProcess(self):
    sys.exit()
        

fig, ax1 = plt.subplots()
color = 'tab:blue'
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)', color = color)
ax1.plot(dataTimeAxis, lista_auxA[instant], label="Clock")
ax1.tick_params(axis = 'y', labelcolor = color)
ax1.legend(loc = "best")

ax2 = ax1.twinx()
color = 'tab:orange'
ax2.set_ylabel('Voltage (mV)', color = color)
#ax2.plot(dataTimeAxis, lista_auxB[instant], color = 'orange', label="ChannelB")
ax2.tick_params(axis = 'y', labelcolor = color)
ax2.legend(loc = "best")

plt.tight_layout()
plt.grid()
plt.show()"""

"""plt.figure()
plt.plot(dataTimeAxis, lista_auxA[instant], label="Clock")
plt.grid()
plt.show()"""
