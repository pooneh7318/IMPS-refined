

import logging
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import queue
import argparse
from threading import Thread
import sys
import json
import pandas as pd

path_json = '/Users/pooneh/Documents/project/machine_improviser_data/gesture-rating-experiment/gestures'
import os
l = []
for json_filename in sorted(os.listdir(path_json)):
    i = 0
    if json_filename.endswith(".json"):
        with open(os.path.join(path_json, json_filename), 'r') as read_file:
            data = json.load(read_file)
            for i in data['analysis']['spectral']:
                l.append((max(data['analysis']['spectral'][i])))
                l.append((min(data['analysis']['spectral'][i])))

global In_zero, In_Chroma11, In_Chroma10, In_Chroma09, In_Chroma08, In_Chroma07, In_Chroma06, In_Chroma05, In_Chroma04, In_Chroma03, In_Chroma02, In_Chroma01, In_Chroma00, In_SpectralSlope, In_SpectralDecrease, In_SpectralVariation, In_SpectralRolloff, In_SpectralKurtosis, In_SpectralSkewness, In_SpectralSpread, In_SpectralCentroid, In_timedt
# In_zero=In_Chroma11=In_Chroma10=In_Chroma09=In_Chroma08=In_Chroma07=In_Chroma06=In_Chroma05=In_Chroma04=In_Chroma03=In_Chroma02=In_Chroma01=In_Chroma00=In_SpectralSlope=In_SpectralDecrease=In_SpectralVariation=In_SpectralRolloff=In_SpectralKurtosis=In_SpectralSkewness=In_SpectralSpread=In_SpectralCentroid=In_timedt= ([] for i in range(22))
global zero, Chroma11, Chroma10, Chroma09, Chroma08, Chroma07, Chroma06, Chroma05, Chroma04, Chroma03, Chroma02, Chroma01, Chroma00, SpectralSlope, SpectralDecrease, SpectralVariation, SpectralRolloff, SpectralKurtosis, SpectralSkewness, SpectralSpread, SpectralCentroid, timedt
# zero=Chroma11=Chroma10=Chroma09=Chroma08=Chroma07=Chroma06=Chroma05=Chroma04=Chroma03=Chroma02=Chroma01=Chroma00=SpectralSlope=SpectralDecrease=SpectralVariation=SpectralRolloff=SpectralKurtosis=SpectralSkewness=SpectralSpread=SpectralCentroid=timedt= ([] for i in range(22))
zero= []
Chroma11= []
Chroma10= []
Chroma09= []
Chroma08= []
Chroma07= []
Chroma06= []
Chroma05= []
Chroma04= []
Chroma03= []
Chroma02= []
Chroma01= []
Chroma00= []
SpectralSlope= []
SpectralDecrease= []
SpectralVariation= []
SpectralRolloff= []
SpectralKurtosis= []
SpectralSkewness= []
SpectralSpread= []
SpectralCentroid= []
timedt= []
In_zero= []
In_Chroma11= []
In_Chroma10= []
In_Chroma09= []
In_Chroma08= []
In_Chroma07= []
In_Chroma06= []
In_Chroma05= []
In_Chroma04= []
In_Chroma03= []
In_Chroma02= []
In_Chroma01= []
In_Chroma00= []
In_SpectralSlope= []
In_SpectralDecrease= []
In_SpectralVariation= []
In_SpectralRolloff= []
In_SpectralKurtosis= []
In_SpectralSkewness= []
In_SpectralSpread= []
In_SpectralCentroid= []
In_timedt= []
global dt_Out
global dt_In
dt_Out = []
dt_In = []

zero_T= []
Chroma11_T= []
Chroma10_T= []
Chroma09_T= []
Chroma08_T= []
Chroma07_T= []
Chroma06_T= []
Chroma05_T= []
Chroma04_T= []
Chroma03_T= []
Chroma02_T= []
Chroma01_T= []
Chroma00_T= []
SpectralSlope_T= []
SpectralDecrease_T= []
SpectralVariation_T= []
SpectralRolloff_T= []
SpectralKurtosis_T= []
SpectralSkewness_T= []
SpectralSpread_T= []
SpectralCentroid_T= []

log_location = "/Users/pooneh/imps/logsbefore/"
data_names = ['x'+str(i) for i in range(22-1)]
column_names = ['date', 'source'] + data_names
perf_df = pd.read_csv(log_location + "2023-03-08T22-28-45-22d-mdrnn.log",
                        header=None, parse_dates=True,
                        index_col=0, names=column_names)
#  Filter out RNN lines, just keep 'interface'
perf_df = perf_df[perf_df.source == 'interface']
# print("line 30",perf_df)
#  Process times.
perf_df['t'] = perf_df.index
perf_df.t = perf_df.t.diff()
perf_df.t = perf_df.t.dt.total_seconds()
perf_df = perf_df.dropna()
# print("line 36",perf_df)
# print("line 36",perf_df.t)


# int_input = osc_arguments
# print("///////////////////////OSCinput///////////////////////:  ", int_input)
# logger = logging.getLogger("impslogger")
# logger.info("{1},interface,{0}".format(','.join(map(str, int_input)),
#              datetime.datetime.now().isoformat()))
# dt_In.append(time.time() - last_user_interaction_time)
last_user_interaction_time = time.time()
# last_user_interaction_data = np.array([dt_In[-1], *int_input])
# In_zero.append(last_user_interaction_data[1])
# In_Chroma11.append(last_user_interaction_data[2])
# In_Chroma10.append(last_user_interaction_data[3])
# In_Chroma09.append(last_user_interaction_data[4])
# In_Chroma08.append(last_user_interaction_data[5])
# In_Chroma07.append(last_user_interaction_data[6])
# In_Chroma06.append(last_user_interaction_data[7])
# In_Chroma05.append(last_user_interaction_data[8])
# In_Chroma04.append(last_user_interaction_data[9])
# In_Chroma03.append(last_user_interaction_data[10])
# In_Chroma02.append(last_user_interaction_data[11])
# In_Chroma01.append(last_user_interaction_data[12])
# In_Chroma00.append(last_user_interaction_data[13])
# In_SpectralSlope.append(last_user_interaction_data[14])
# In_SpectralDecrease.append(last_user_interaction_data[15])
# In_SpectralVariation.append(last_user_interaction_data[16])
# In_SpectralRolloff.append(last_user_interaction_data[17])
# In_SpectralKurtosis.append(last_user_interaction_data[18])
# In_SpectralSkewness.append(last_user_interaction_data[19])
# In_SpectralSpread.append(last_user_interaction_data[20])
# In_SpectralCentroid.append(last_user_interaction_data[21])
# sumdt = 0
# for timeitem in dt_In:
#             sumdt += timeitem
# In_timedt.append(sumdt * 1000)
# print("teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeest")
# print(last_user_interaction_data)
# assert len(last_user_interaction_data) == args.dimension, "Input is incorrect dimension, set dimension to %r" % len(last_user_interaction_data)
# These values are accessed by the RNN in the interaction loop function.
global inputarray
inputarray = np.array(perf_df[['t']+data_names])
for i in range(150):
    # len(inputarray)
    last_user_interaction_data = inputarray[i]
    # print(last_user_interaction_data)
    In_zero.append(last_user_interaction_data[1]*(l[0] - l[1])+l[1])
    In_Chroma11.append(last_user_interaction_data[2]*(l[2] - l[3])+l[3])
    In_Chroma10.append(last_user_interaction_data[3]*(l[4] - l[5])+l[5])
    In_Chroma09.append(last_user_interaction_data[4]*(l[6] - l[7])+l[7])
    In_Chroma08.append(last_user_interaction_data[5]*(l[8] - l[9])+l[9])
    In_Chroma07.append(last_user_interaction_data[6]*(l[10] - l[11])+l[11])
    In_Chroma06.append(last_user_interaction_data[7]*(l[12] - l[13])+l[13])
    In_Chroma05.append(last_user_interaction_data[8]*(l[14] - l[15])+l[15])
    In_Chroma04.append(last_user_interaction_data[9]*(l[16] - l[17])+l[17])
    In_Chroma03.append(last_user_interaction_data[10]*(l[18] - l[19])+l[19])
    In_Chroma02.append(last_user_interaction_data[11]*(l[20] - l[21])+l[21])
    In_Chroma01.append(last_user_interaction_data[12]*(l[22] - l[23])+l[23])
    In_Chroma00.append(last_user_interaction_data[13]*(l[24] - l[25])+l[25])
    In_SpectralSlope.append(last_user_interaction_data[14]*(l[26] - l[27])+l[27])
    In_SpectralDecrease.append(last_user_interaction_data[15]*(l[28] - l[29])+l[29])
    In_SpectralVariation.append(last_user_interaction_data[16]*(l[30] - l[31])+l[31])
    In_SpectralRolloff.append(last_user_interaction_data[17]*(l[32] - l[33])+l[33])
    In_SpectralKurtosis.append(last_user_interaction_data[18]*(l[34] - l[35])+l[35])
    In_SpectralSkewness.append(last_user_interaction_data[19]*(l[36] - l[37])+l[37])
    In_SpectralSpread.append(last_user_interaction_data[20]*(l[38] - l[39])+l[39])
    In_SpectralCentroid.append(last_user_interaction_data[21]*(l[40] - l[41])+l[41])
    sumdt = 0
    dt_In.append(last_user_interaction_data[0])
    for timeitem in dt_In:
                sumdt += timeitem
    In_timedt.append(sumdt * 1000)

legend = 'Chroma11'
plt.plot(In_timedt, In_Chroma11, color='red', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma10'
plt.plot(In_timedt, In_Chroma10, color='green', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma09'
plt.plot(In_timedt, In_Chroma09, color='pink', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")


plt.legend()
plt.show()

legend = 'Chroma08'
plt.plot(In_timedt, In_Chroma08, color='green', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma07'
plt.plot(In_timedt, In_Chroma07, color='blue', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma06'
plt.plot(In_timedt, In_Chroma06, color='red', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")


plt.legend()
plt.show()



legend = 'Chroma05'
plt.plot(In_timedt, In_Chroma05, color='brown', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma04'
plt.plot(In_timedt, In_Chroma04, color='violet', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma03'
plt.plot(In_timedt, In_Chroma03, color='teal', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")

plt.legend()
plt.show()

legend = 'Chroma02'
plt.plot(In_timedt, In_Chroma02, color='violet', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma01'
plt.plot(In_timedt, In_Chroma01, color='turquoise', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")
legend = 'Chroma00'
plt.plot(In_timedt, In_Chroma00, color='maroon', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Chroma")


plt.legend()
plt.show()


# legend = 'zero'
plt.plot(In_timedt, In_zero, color='blue', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("Zero")

# plt.legend()
plt.show()

# legend = 'slope'
plt.plot(In_timedt, In_SpectralSlope, color='red', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralSlope")

# plt.legend()
plt.show()

# legend = 'decrease'
plt.plot(In_timedt, In_SpectralDecrease, color='brown', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralDecrease")

# plt.legend()
plt.show()

# legend = 'variation'
plt.plot(In_timedt, In_SpectralVariation, color='violet', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralVariatioon")

# plt.legend()
plt.show()

# legend = 'rolloff'
plt.plot(In_timedt, In_SpectralRolloff, color='maroon', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralRolloff")

# plt.legend()
plt.show()

# legend = 'kurtosis'
plt.plot(In_timedt, In_SpectralKurtosis, color='green', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralKurtosis")

# plt.legend()
plt.show()

# legend = 'skewness'
plt.plot(In_timedt, In_SpectralSkewness, color='black', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralSkewness")

# plt.legend()
plt.show()

# legend = 'spread'
plt.plot(In_timedt, In_SpectralSpread, color='cyan', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralSpread")

# plt.legend()
plt.show()

# legend = 'centroid'
plt.plot(In_timedt, In_SpectralCentroid, color='teal', label=legend)
plt.xlabel("Time(ms)")
plt.ylabel("SpectralCentroid")

# plt.legend()
plt.show()