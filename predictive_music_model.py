#!/usr/bin/python



import logging
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import queue
from pythonosc import dispatcher
from pythonosc import osc_server
from pythonosc import udp_client
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


# MAX = 59965.554455295205
# MIN = -2.619709253311157
# factor = MAX - MIN


with open("/Users/pooneh/Documents/project/result_allin_justReponse.json", "r") as json_file:
    json_decoded = json.load(json_file)

parser = argparse.ArgumentParser(description='Predictive Musical Interaction MDRNN Interface.')
parser.add_argument('-l', '--log', dest='logging', action="store_true", help='Save input and RNN data to a log file.')
parser.add_argument('-v', '--verbose', dest='verbose', action="store_true", help='Verbose mode, print prediction results.')
# Performance modes
parser.add_argument('-o', '--only', dest='useronly', action="store_true", help="User control only mode, no RNN.")
parser.add_argument('-c', '--call', dest='callresponse', action="store_true", help='Call and response mode.')
parser.add_argument('-p', '--polyphony', dest='polyphony', action="store_true", help='Harmony mode.')
parser.add_argument('-b', '--battle', dest='battle', action="store_true", help='Battle royale mode.')
parser.add_argument('--callresponsethresh', type=float, default=20.0, help="Seconds to wait before switching to response")
# OSC addresses
parser.add_argument("--clientip", default="localhost", help="The address of output device.")
parser.add_argument("--clientport", type=int, default=8001, help="The port the output device is listening on.")
parser.add_argument("--serverip", default="localhost", help="The address of this server.")
parser.add_argument("--serverport", type=int, default=7001, help="The port this server should listen on.")
# MDRNN arguments.
parser.add_argument('-d', '--dimension', type=int, dest='dimension', default=2, help='The dimension of the data to model, must be >= 2.')
parser.add_argument("--modelsize", default="s", help="The model size: xs, s, m, l, xl", type=str)
parser.add_argument("--sigmatemp", type=float, default=0.01, help="The sigma temperature for sampling.")
parser.add_argument("--pitemp", type=float, default=1, help="The pi temperature for sampling.")
args = parser.parse_args()

# import tensorflow, doing this later to make CLI more responsive.
print("Importing MDRNN.")
start_import = time.time()
import imps_mdrnn
import tensorflow.compat.v1 as tf
print("Done. That took", time.time() - start_import, "seconds.")

# Choose model parameters.
if args.modelsize == 'xxs':
    mdrnn_units = 16
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'xs':
    mdrnn_units = 32
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 's':
    mdrnn_units = 64
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'm':
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'l':
    mdrnn_units = 256
    mdrnn_mixes = 5
    mdrnn_layers = 2
elif args.modelsize == 'xl':
    mdrnn_units = 512
    mdrnn_mixes = 5
    mdrnn_layers = 3
else:
    mdrnn_units = 128
    mdrnn_mixes = 5
    mdrnn_layers = 2

# Interaction Loop Parameters
# All set to false before setting is chosen.
user_to_rnn = False
rnn_to_rnn = False
rnn_to_sound = False

# Interactive Mapping
if args.callresponse:
    print("Entering call and response mode.")
    # set initial conditions.
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = False
elif args.polyphony:
    print("Entering polyphony mode.")
    user_to_rnn = True
    rnn_to_rnn = False
    rnn_to_sound = True
elif args.battle:
    print("Entering battle royale mode.")
    user_to_rnn = False
    rnn_to_rnn = True
    rnn_to_sound = True
elif args.useronly:
    print("Entering user only mode.")
    user_to_rnn = False
    rnn_to_rnn = False
    rnn_to_sound = False


def build_network(sess):
    """Build the MDRNN."""
    imps_mdrnn.MODEL_DIR = "./models/"
    tf.keras.backend.set_session(sess)
    with compute_graph.as_default():
        net = imps_mdrnn.PredictiveMusicMDRNN(mode=imps_mdrnn.NET_MODE_RUN,
                                              dimension=args.dimension,
                                              n_hidden_units=mdrnn_units,
                                              n_mixtures=mdrnn_mixes,
                                              layers=mdrnn_layers)
        net.pi_temp = args.pitemp
        net.sigma_temp = args.sigmatemp
    print("MDRNN Loaded.")
    return net


def handle_interface_message(address: str, *osc_arguments) -> None:
    """Handler for OSC messages from the interface"""
    print("teeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeest1")
    global last_user_interaction_time
    global last_user_interaction_data
    
    if args.verbose:
        print("User:", time.time(), ','.join(map(str, osc_arguments)))

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
    for i in range(40):
        interface_input_queue.put_nowait(inputarray[i])
        last_user_interaction_data = inputarray[i]
        print(last_user_interaction_data)
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


def request_rnn_prediction(input_value):
    """ Accesses a single prediction from the RNN. """
    output_value = net.generate_touch(input_value)
    return output_value


def make_prediction(sess, compute_graph):
    # Interaction loop: reads input, makes predictions, outputs results.
    # Make predictions.

    # First deal with user --> MDRNN prediction
    if user_to_rnn and not interface_input_queue.empty():
        item = interface_input_queue.get(block=True, timeout=None)
        print(item)
        print("/////////////////////////item from make prediction/////////////////////////////////////")
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            rnn_output = request_rnn_prediction(item)
        #if args.verbose:
        #    print("User->RNN prediction:", rnn_output)
        if rnn_to_sound:
            rnn_output_buffer.put_nowait(rnn_output)
        interface_input_queue.task_done()

    # Now deal with MDRNN --> MDRNN prediction.
    if rnn_to_rnn and rnn_output_buffer.empty() and not rnn_prediction_queue.empty():
        item = rnn_prediction_queue.get(block=True, timeout=None)
        tf.keras.backend.set_session(sess)
        with compute_graph.as_default():
            rnn_output = request_rnn_prediction(item)
        #if args.verbose:
        #   print("RNN->RNN prediction out:", rnn_output)
        rnn_output_buffer.put_nowait(rnn_output)  # put it in the playback queue.
        #print("RNN->RNN prediction out:", rnn_output)
        rnn_prediction_queue.task_done()


def send_sound_command(command_args):
    """Send a sound command back to the interface/synth"""
    assert len(command_args)+1 == args.dimension, "Dimension not same as prediction size." # Todo more useful error.
    osc_client.send_message(OUTPUT_MESSAGE_ADDRESS, command_args)


def playback_rnn_loop():
    # Plays back RNN notes from its buffer queue.
    # sys.stdout.flush()
    
    while True:
        item = rnn_output_buffer.get(block=True, timeout=None)  # Blocks until next item is available.
        print("////////////////////item[0]///////////////////////////////////////")
        print(item[0])
        # print("processing an rnn command", time.time())
        dt_Out.append(item[0])
        x_pred = np.minimum(np.maximum(item[1:], 0), 1)
        print("ttttttttttyyyyyyyyyypppppppppppeeeeeeeeeeeee///////////////////////")
        print(type(x_pred))
        #dt = max(dt, 0.001)  # stop accidental minus and zero dt.
        #dt = dt * 10
        dt_Out[-1] = max(dt_Out[-1], 0.001)
        print("dttdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtddttdtdtdttdtdtdtd")
        print(dt_Out[-1])
        print("dttdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtdtddttdtdtdttdtdtdtd")
        time.sleep(dt_Out[-1])  # wait until time to play the sound
        # put last played in queue for prediction.
        rnn_prediction_queue.put_nowait(np.concatenate([np.array([dt_Out[-1]]), x_pred]))
        if rnn_to_sound:
            # print('helppppp&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&', flush=True)
            send_sound_command(x_pred)
            print("infooooooooooooooooooooooo////////////////////////////////////")
            print("RNN Played:", x_pred, "at", dt_Out[-1])

            zero.append(x_pred[0]*(l[0] - l[1])+l[1])
            Chroma11.append(x_pred[1]*(l[2] - l[3])+l[3])
            Chroma10.append(x_pred[2]*(l[4] - l[5])+l[5])
            Chroma09.append(x_pred[3]*(l[6] - l[7])+l[7])
            Chroma08.append(x_pred[4]*(l[8] - l[9])+l[9])
            Chroma07.append(x_pred[5]*(l[10] - l[11])+l[11])
            Chroma06.append(x_pred[6]*(l[12] - l[13])+l[13])
            Chroma05.append(x_pred[7]*(l[14] - l[15])+l[15])
            Chroma04.append(x_pred[8]*(l[16] - l[17])+l[17])
            Chroma03.append(x_pred[9]*(l[18] - l[19])+l[19])
            Chroma02.append(x_pred[10]*(l[20] - l[21])+l[21])
            Chroma01.append(x_pred[11]*(l[22] - l[23])+l[23])
            Chroma00.append(x_pred[12]*(l[24] - l[25])+l[25])
            SpectralSlope.append(x_pred[13]*(l[26] - l[27])+l[27])
            SpectralDecrease.append(x_pred[14]*(l[28] - l[29])+l[29])
            SpectralVariation.append(x_pred[15]*(l[30] - l[31])+l[31])
            SpectralRolloff.append(x_pred[16]*(l[32] - l[33])+l[33])
            SpectralKurtosis.append(x_pred[17]*(l[34] - l[35])+l[35])
            SpectralSkewness.append(x_pred[18]*(l[36] - l[37])+l[37])
            SpectralSpread.append(x_pred[19]*(l[38] - l[39])+l[39])
            SpectralCentroid.append(x_pred[20]*(l[40] - l[41])+l[41])
            sumdt = 0
            for timeitem in dt_Out:
                sumdt += timeitem
            timedt.append(sumdt * 1000)
            logger = logging.getLogger("impslogger")
            logger.info("{1},rnn,{0}".format(','.join(map(str, x_pred)),
                         datetime.datetime.now().isoformat()))
        rnn_output_buffer.task_done()


def monitor_user_action():
    # Handles changing responsibility in Call-Response mode.
    global call_response_mode
    global user_to_rnn
    global rnn_to_rnn
    global rnn_to_sound
    
    
    # Check when the last user interaction was
    dt = time.time() - last_user_interaction_time
    if dt > args.callresponsethresh:
        # switch to response modes.
        user_to_rnn = False
        rnn_to_rnn = True
        rnn_to_sound = True
        if call_response_mode is 'call':
            print("switching to response.")
            
            
            
            call_response_mode = 'response'
            
            while not rnn_prediction_queue.empty():
                # Make sure there's no inputs waiting to be predicted.
                rnn_prediction_queue.get()
                rnn_prediction_queue.task_done()
            rnn_prediction_queue.put_nowait(last_user_interaction_data)  # prime the RNN queue
            # In_zero=In_Chroma11=In_Chroma10=In_Chroma09=In_Chroma08=In_Chroma07=In_Chroma06=In_Chroma05=In_Chroma04=In_Chroma03=In_Chroma02=In_Chroma01=In_Chroma00=In_SpectralSlope=In_SpectralDecrease=In_SpectralVariation=In_SpectralRolloff=In_SpectralKurtosis=In_SpectralSkewness=In_SpectralSpread=In_SpectralCentroid=In_timedt= ([] for i in range(22))
            
    else:
        # switch to call mode.
        user_to_rnn = True
        rnn_to_rnn = False
        rnn_to_sound = False
        if call_response_mode=='response':
            global In_zero, In_Chroma11, In_Chroma10, In_Chroma09, In_Chroma08, In_Chroma07, In_Chroma06, In_Chroma05, In_Chroma04, In_Chroma03, In_Chroma02, In_Chroma01, In_Chroma00, In_SpectralSlope, In_SpectralDecrease, In_SpectralVariation, In_SpectralRolloff, In_SpectralKurtosis, In_SpectralSkewness, In_SpectralSpread, In_SpectralCentroid, In_timedt, dt_In
            # In_zero= []
            # In_Chroma11= []
            # In_Chroma10= []
            # In_Chroma09= []
            # In_Chroma08= []
            # In_Chroma07= []
            # In_Chroma06= []
            # In_Chroma05= []
            # In_Chroma04= []
            # In_Chroma03= []
            # In_Chroma02= []
            # In_Chroma01= []
            # In_Chroma00= []
            # In_SpectralSlope= []
            # In_SpectralDecrease= []
            # In_SpectralVariation= []
            # In_SpectralRolloff= []
            # In_SpectralKurtosis= []
            # In_SpectralSkewness= []
            # In_SpectralSpread= []
            # In_SpectralCentroid= []
            # In_timedt= []
            # time.sleep(0.5)
            # dt_In = []
            print("switching to call.")
            
            
            
            call_response_mode = 'call'
            # Empty the RNN queues.
            while not rnn_output_buffer.empty():
                # Make sure there's no actions waiting to be synthesised.
                rnn_output_buffer.get()
                rnn_output_buffer.task_done()
            # zero=Chroma11=Chroma10=Chroma09=Chroma08=Chroma07=Chroma06=Chroma05=Chroma04=Chroma03=Chroma02=Chroma01=Chroma00=SpectralSlope=SpectralDecrease=SpectralVariation=SpectralRolloff=SpectralKurtosis=SpectralSkewness=SpectralSpread=SpectralCentroid=timedt= ([] for i in range(22))
            global zero, Chroma11, Chroma10, Chroma09, Chroma08, Chroma07, Chroma06, Chroma05, Chroma04, Chroma03, Chroma02, Chroma01, Chroma00, SpectralSlope, SpectralDecrease, SpectralVariation, SpectralRolloff, SpectralKurtosis, SpectralSkewness, SpectralSpread, SpectralCentroid, timedt, dt_Out
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
            time.sleep(0.5)
            dt_Out = []
# Logging
LOG_FILE = datetime.datetime.now().isoformat().replace(":", "-")[:19] + "-" + str(args.dimension) + "d" +  "-mdrnn.log"  # Log file name.
LOG_FILE = "logs/" + LOG_FILE
LOG_FORMAT = '%(message)s'

if args.logging:
    formatter = logging.Formatter(LOG_FORMAT)
    handler = logging.FileHandler(LOG_FILE)        
    handler.setFormatter(formatter)
    logger = logging.getLogger("impslogger")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    print("Logging enabled:", LOG_FILE)
# Details for OSC output
INPUT_MESSAGE_ADDRESS = "/interface"
OUTPUT_MESSAGE_ADDRESS = "/prediction"

# Set up runtime variables.
# ## Load the Model
compute_graph = tf.Graph()
with compute_graph.as_default():
    sess = tf.Session()
net = build_network(sess)
interface_input_queue = queue.Queue()
rnn_prediction_queue = queue.Queue()
rnn_output_buffer = queue.Queue()
writing_queue = queue.Queue()
last_user_interaction_time = time.time()
last_user_interaction_data = imps_mdrnn.random_sample(out_dim=args.dimension)
rnn_prediction_queue.put_nowait(imps_mdrnn.random_sample(out_dim=args.dimension))
call_response_mode = 'call'

# Set up OSC client and server
osc_client = udp_client.SimpleUDPClient(args.clientip, args.clientport)
disp = dispatcher.Dispatcher()
print("helpppppppppppppppppppppppp")
disp.map(INPUT_MESSAGE_ADDRESS, handle_interface_message)
server = osc_server.ThreadingOSCUDPServer((args.serverip, args.serverport), disp)

thread_running = True  # todo is this line needed?

# Set up run loop.
print("Preparing MDRNN.")
tf.keras.backend.set_session(sess)
with compute_graph.as_default():
    net.load_model()  # try loading from default file location.
print("Preparing MDRNN thread.")
rnn_thread = Thread(target=playback_rnn_loop, name="rnn_player_thread", daemon=True)
print("Preparing Server thread.")
server_thread = Thread(target=server.serve_forever, name="server_thread", daemon=True)
# Thread(target=lambda x:writing_queue.put_nowait('hi         ))))))))))))))))'), name="ali", daemon=True).start()
# writing_queue.put_nowait('cheraaaaaaaaaaaaaaaaa')
try:
    
    rnn_thread.start()
    server_thread.start()
    print("Prediction server started.")
    print("Serving on {}".format(server.server_address))
    while True:
        # while not writing_queue.empty():
        #     print(writing_queue.get(block=True, timeout=None))
        make_prediction(sess, compute_graph)
        if args.callresponse:
            monitor_user_action()
except KeyboardInterrupt:
    
    print("\nCtrl-C received... exiting.")
    thread_running = False
    rnn_thread.join(timeout=0.1)
    server_thread.join(timeout=0.1)
    pass
finally:
    print(zero)

    dict = {}
    j = 0
    # cat1=[]
    # attributeValues = [zero, Chroma11, Chroma10, Chroma09, Chroma08, Chroma07, Chroma06, Chroma05, Chroma04, Chroma03, Chroma02, Chroma01, Chroma00, SpectralSlope, SpectralDecrease, SpectralVariation, SpectralRolloff, SpectralKurtosis, SpectralSkewness, SpectralSpread, SpectralCentroid, timedt]
    # for cat in attributeValues:
    #     for i in range(len(cat)):
    #         if(i==0):
    #             cat1.append(cat[i] * 1.5)
    #         else:
    #             if(cat[i]>=cat[i-1]):
    #                 cat1.append(cat[i] * 1.5)
    #             if(cat[i]<cat[i-1]):
    #                 cat1.append(cat[i] / 1.5)
    # zero = cat1[0:len(zero)] 
    # Chroma11 = cat1[len(zero):len(zero)+len(Chroma11)]  
    # Chroma10 = cat1[len(zero)+len(Chroma11):len(zero)+len(Chroma11)+len(Chroma10)]  
    # Chroma09 = cat1[len(zero)+len(Chroma11)+len(Chroma10):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)]  
    # Chroma08 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)]  
    # Chroma07 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)]
    # Chroma06 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)]
    # Chroma05 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)]
    # Chroma04 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)]
    # Chroma03 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)]
    # Chroma02 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)]
    # Chroma01 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)]
    # Chroma00 = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)]
    # SpectralSlope = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)]
    # SpectralDecrease = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)]
    # SpectralVariation = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)]
    # SpectralRolloff = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)]
    # SpectralKurtosis = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)]
    # SpectralSkewness = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)+len(SpectralSkewness)]
    # SpectralSpread = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)+len(SpectralSkewness):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)+len(SpectralSkewness)+len(SpectralSpread)]
    # SpectralCentroid = cat1[len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)+len(SpectralSkewness)+len(SpectralSpread):len(zero)+len(Chroma11)+len(Chroma10)+len(Chroma09)+len(Chroma08)+len(Chroma07)+len(Chroma06)+len(Chroma05)+len(Chroma04)+len(Chroma03)+len(Chroma02)+len(Chroma01)+len(Chroma00)+len(SpectralSlope)+len(SpectralDecrease)+len(SpectralVariation)+len(SpectralRolloff)+len(SpectralKurtosis)+len(SpectralSkewness)+len(SpectralSpread)+len(SpectralCentroid)]

            # print("///////////////////thisiszero/////////////////")
            # print(attributeValues[f])
    
    # print(zero)
                
    attributeValues = [zero, Chroma11, Chroma10, Chroma09, Chroma08, Chroma07, Chroma06, Chroma05, Chroma04, Chroma03, Chroma02, Chroma01, Chroma00, SpectralSlope, SpectralDecrease, SpectralVariation, SpectralRolloff, SpectralKurtosis, SpectralSkewness, SpectralSpread, SpectralCentroid, timedt]

    attributeList = ['0', 'Chroma11', 'Chroma10', 'Chroma09', 'Chroma08', 'Chroma07', 'Chroma06', 'Chroma05', 'Chroma04', 'Chroma03', 'Chroma02', 'Chroma01', 'Chroma00', 'SpectralSlope', 'SpectralDecrease', 'SpectralVariation', 'SpectralRolloff', 'SpectralKurtosis', 'SpectralSkewness', 'SpectralSpread', 'SpectralCentroid', 'time']
    for i in attributeList:
        dict[i] = attributeValues[j]
        j += 1
    json_decoded["analysis"]["spectral"] = dict
    with open("/Users/pooneh/Documents/project/resulttttt_allin_withInput.json", "w") as write_file:
        json.dump(json_decoded, write_file, indent=2)
    dict = {}
    j = 0
    attributeValuesin = [In_zero, In_Chroma11, In_Chroma10, In_Chroma09, In_Chroma08, In_Chroma07, In_Chroma06, In_Chroma05, In_Chroma04, In_Chroma03, In_Chroma02, In_Chroma01, In_Chroma00, In_SpectralSlope, In_SpectralDecrease, In_SpectralVariation, In_SpectralRolloff, In_SpectralKurtosis, In_SpectralSkewness, In_SpectralSpread, In_SpectralCentroid, In_timedt]
    attributeList = ['0', 'Chroma11', 'Chroma10', 'Chroma09', 'Chroma08', 'Chroma07', 'Chroma06', 'Chroma05', 'Chroma04', 'Chroma03', 'Chroma02', 'Chroma01', 'Chroma00', 'SpectralSlope', 'SpectralDecrease', 'SpectralVariation', 'SpectralRolloff', 'SpectralKurtosis', 'SpectralSkewness', 'SpectralSpread', 'SpectralCentroid', 'time']
    
    for i in attributeList:
        dict[i] = attributeValuesin[j]
        j += 1
    json_decoded["analysis"]["spectral"] = dict
    with open("/Users/pooneh/Documents/project/inputttt_allin.json", "w") as write_file:
        json.dump(json_decoded, write_file, indent=2)
    #In_zero=In_Chroma11=In_Chroma10=In_Chroma09=In_Chroma08=In_Chroma07=In_Chroma06=In_Chroma05=In_Chroma04=In_Chroma03=In_Chroma02=In_Chroma01=In_Chroma00=In_SpectralSlope=In_SpectralDecrease=In_SpectralVariation=In_SpectralRolloff=In_SpectralKurtosis=In_SpectralSkewness=In_SpectralSpread=In_SpectralCentroid=In_timedt= ([] for i in range(22))

    zero_T = In_zero + zero
    Chroma11_T= In_Chroma11 + Chroma11
    Chroma10_T= In_Chroma10 + Chroma10
    Chroma09_T= In_Chroma09 + Chroma09
    Chroma08_T= In_Chroma08 + Chroma08
    Chroma07_T= In_Chroma07 + Chroma07
    Chroma06_T= In_Chroma06 + Chroma06
    Chroma05_T= In_Chroma05 + Chroma05
    Chroma04_T= In_Chroma04 + Chroma04
    Chroma03_T= In_Chroma03 + Chroma03
    Chroma02_T= In_Chroma02 + Chroma02
    Chroma01_T= In_Chroma01 + Chroma01
    Chroma00_T= In_Chroma00 + Chroma00
    SpectralSlope_T= In_SpectralSlope + SpectralSlope
    SpectralDecrease_T= In_SpectralDecrease + SpectralDecrease
    SpectralVariation_T= In_SpectralVariation + SpectralVariation
    SpectralRolloff_T= In_SpectralRolloff + SpectralRolloff
    SpectralKurtosis_T= In_SpectralKurtosis + SpectralKurtosis
    SpectralSkewness_T= In_SpectralSkewness + SpectralSkewness
    SpectralSpread_T= In_SpectralSpread + SpectralSpread
    SpectralCentroid_T= In_SpectralCentroid + SpectralCentroid
    costumized_timedt = []
    for t in timedt:
        costumized_timedt.append(t + In_timedt[-1])
   
    timedt_T = In_timedt + costumized_timedt

    # legend = 'zero'
    # plt.plot(timedt_T, zero_T, color='blue', label=legend)
    # plt.xlabel("Time(ms)")
    # plt.ylabel("Spectral")
    legend = 'Chroma11'
    plt.plot(timedt_T, Chroma11_T, color='red', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma10'
    plt.plot(timedt_T, Chroma10_T, color='green', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma09'
    plt.plot(timedt_T, Chroma09_T, color='pink', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    

    plt.legend()
    plt.show()

    legend = 'Chroma08'
    plt.plot(timedt_T, Chroma08_T, color='green', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma07'
    plt.plot(timedt_T, Chroma07_T, color='blue', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma06'
    plt.plot(timedt_T, Chroma06_T, color='red', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")


    plt.legend()
    plt.show()



    legend = 'Chroma05'
    plt.plot(timedt_T, Chroma05_T, color='brown', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma04'
    plt.plot(timedt_T, Chroma04_T, color='violet', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma03'
    plt.plot(timedt_T, Chroma03_T, color='teal', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")

    plt.legend()
    plt.show()

    legend = 'Chroma02'
    plt.plot(timedt_T, Chroma02_T, color='violet', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma01'
    plt.plot(timedt_T, Chroma01_T, color='turquoise', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")
    legend = 'Chroma00'
    plt.plot(timedt_T, Chroma00_T, color='maroon', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Chroma")


    plt.legend()
    plt.show()


    # legend = 'zero'
    plt.plot(timedt_T, zero_T, color='blue', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("Zero")

    # plt.legend()
    plt.show()

    # legend = 'slope'
    plt.plot(timedt_T, SpectralSlope_T, color='red', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralSlope")

    # plt.legend()
    plt.show()

    # legend = 'decrease'
    plt.plot(timedt_T, SpectralDecrease_T, color='brown', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralDecrease")

    # plt.legend()
    plt.show()

    # legend = 'variation'
    plt.plot(timedt_T, SpectralVariation_T, color='violet', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralVariatioon")

    # plt.legend()
    plt.show()

    # legend = 'rolloff'
    plt.plot(timedt_T, SpectralRolloff_T, color='maroon', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralRolloff")

    # plt.legend()
    plt.show()

    # legend = 'kurtosis'
    plt.plot(timedt_T, SpectralKurtosis_T, color='green', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralKurtosis")

    # plt.legend()
    plt.show()

    # legend = 'skewness'
    plt.plot(timedt_T, SpectralSkewness_T, color='black', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralSkewness")

    # plt.legend()
    plt.show()

    # legend = 'spread'
    plt.plot(timedt_T, SpectralSpread_T, color='cyan', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralSpread")

    # plt.legend()
    plt.show()

    # legend = 'centroid'
    plt.plot(timedt_T, SpectralCentroid_T, color='teal', label=legend)
    plt.xlabel("Time(ms)")
    plt.ylabel("SpectralCentroid")

    # plt.legend()
    plt.show()



    print("\nDone, shutting down.")
