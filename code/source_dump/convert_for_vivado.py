import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Lambda, Flatten
from keras.layers import Input, Concatenate
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import time
from scipy import sparse

import matplotlib.pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

params = {
    "epochs": 300,
    "mini_ep": 15,
    "batch_size": 64,
    "step": 15,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 500,
    "validation_split": 0.1,
    "input_size":3
}

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers


def rnn_walk_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=256, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)

    left2 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left1)

    # left3 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left2)

    sliced = Lambda(lambda x: x[:,-1*layers[3]:,:], output_shape=(layers[3], layers[2]))(left2)

    x = Dense(units=layers[2],activation='relu')(sliced)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=1,activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
    x = Flatten()(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

def quantize_weights(inp_flt):
    val = round(inp_flt/(2**-7), 0)*2**-7
    val = round(val*(2**7), 0)

    hex_val = hex(int(val))

    if(hex_val[-2]=='x'):
        return hex_val[:-1] + "0" + hex_val[-1]
    else:
        return hex_val

def create_lstm(inp_mat, layer_name):

    final_output = ""
    def_output = ""

    hl = inp_mat[0]
    rl = inp_mat[1]
    bl = inp_mat[2]

    units = np.shape(hl)[1]//4

    weights = {}
    biases = {}

    weights['W_i'] = hl[:, :units]
    weights['W_f'] = hl[:, units: units * 2]
    weights['W_c'] = hl[:, units * 2: units * 3]
    weights['W_o'] = hl[:, units * 3:]

    weights['U_i'] = rl[:, :units]
    weights['U_f'] = rl[:, units: units * 2]
    weights['U_c'] = rl[:, units * 2: units * 3]
    weights['U_o'] = rl[:, units * 3:]

    biases['b_i'] = bl[:units]
    biases['b_f'] = bl[units: units * 2]
    biases['b_c'] = bl[units * 2: units * 3]
    biases['b_o'] = bl[units * 3:]

    for ele in weights:
        curr_we = weights[ele]
        curr_we_shape = curr_we.shape
        if('W' in ele):
            cut_pos = (curr_we.shape[0]//2)*2
            curr_we = curr_we[:cut_pos, :]
            curr_we = np.reshape(curr_we, (2, -1, curr_we_shape[1]))
        else:
            cut_pos = (curr_we.shape[0]//4)*4
            curr_we = curr_we[:cut_pos, :]
            curr_we = np.reshape(curr_we, (4, -1, curr_we_shape[1]))
        curr_we_shape = curr_we.shape
        def_output += "typedef ap_fixed<8 , 1, AP_RND_ZERO, AP_WRAP> " + layer_name + "_" + ele + ";\n"
        final_output += "const ap_uint<8> " + layer_name + "_" + ele + " [" + str(curr_we_shape[0]) + "][" + str(curr_we_shape[1]) + "][1][" + str(curr_we_shape[2]) + "] = { "
        for i in range(curr_we_shape[0]):
            final_output += "{ "
            for j in range(curr_we_shape[1]):
                final_output += "{ { "
                for k in range(curr_we_shape[2]):
                    final_output += quantize_weights(curr_we[i][j][k])
                    if(k!=curr_we_shape[2]-1):
                        final_output += ",\n"
                    else:
                        final_output += "\n"
                        final_output += "}\n}\n"
                if(j!=curr_we_shape[1]-1):
                    final_output += ",\n"
                else:
                    final_output += "\n"
                    final_output += "}\n"
            if(i!=curr_we_shape[0]-1):
                final_output += ",\n"
            else:
                final_output += "}\n;\n\n"

    for ele in biases:
        curr_bias = biases[ele]
        curr_bias_shape = curr_bias.shape
        def_output += "typedef ap_fixed<8 , 1, AP_RND_ZERO, AP_WRAP> " + layer_name + "_" + ele + ";\n"
        final_output += "const ap_uint<8> " + layer_name + "_" + ele + " [1][" + str(curr_bias_shape[0]) + "] = { { "
        for i in range(curr_bias_shape[0]):
            final_output += quantize_weights(curr_bias[i])
            if(i!=curr_bias_shape[0]-1):
                final_output += ",\n"
            else:
                final_output += "\n"
                final_output += "}\n}\n;\n\n"

    # exit()
    return def_output, final_output

def create_fc(inp_mat, layer_name):
    final_output = ""
    def_output = ""

    weights = {}
    biases = {}

    weights['Wl'] = inp_mat[0]
    biases['Bl'] = inp_mat[1]

    for ele in weights:
        curr_we = weights[ele]
        curr_we_shape = curr_we.shape
        print(curr_we_shape)
        def_output += "typedef ap_fixed<8 , 1, AP_RND_ZERO, AP_WRAP> " + layer_name + "_" + ele + ";\n"
        final_output += "const ap_uint<8> " + layer_name + "_" + ele + " [" + str(curr_we_shape[0]) + "][" + str(curr_we_shape[1]) + "] = { "
        for i in range(curr_we_shape[0]):
            final_output += "{ "
            for j in range(curr_we_shape[1]):
                final_output += quantize_weights(curr_we[i][j])
                if(j!=curr_we_shape[1]-1):
                    final_output += ",\n"
                else:
                    final_output += "\n"
                    final_output += "}\n"
            if(i!=curr_we_shape[0]-1):
                final_output += ",\n"
            else:
                final_output += "}\n;\n\n"

    for ele in biases:
        curr_bias = biases[ele]
        curr_bias_shape = curr_bias.shape
        print(curr_bias_shape)
        def_output += "typedef ap_fixed<8 , 1, AP_RND_ZERO, AP_WRAP> " + layer_name + "_" + ele + ";\n"
        final_output += "const ap_uint<8> " + layer_name + "_" + ele + " [1][" + str(curr_bias_shape[0]) + "] = { { "
        for i in range(curr_bias_shape[0]):
            final_output += quantize_weights(curr_bias[i])
            if(i!=curr_bias_shape[0]-1):
                final_output += ",\n"
            else:
                final_output += "\n"
                final_output += "}\n}\n;\n\n"

    # exit()
    return def_output, final_output


# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 7, X_train.shape[3]]
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]

lstm_layer = [26, 15, 500, 7, 3]

model = rnn_walk_lstm(lstm_layer, params)

saved_model = "BI_LSTM_L2_walk_ahead"

from keras.models import load_model, save_model

print(model.summary())

from keras.models import Model

model.load_weights("models_window/%s" % (saved_model) + "_val")

all_l = model.layers

lstm_ite = 1
fc_ite = 1
new_file_head = ""
new_file = ""

for ele in model.layers:
    if(len(ele.get_weights())==3):
        str_res_head, str_res = create_lstm(ele.get_weights(), "lstm" + str(lstm_ite))
        new_file_head += str_res_head
        new_file += str_res
        lstm_ite += 1
    elif(len(ele.get_weights())==2):
        str_res_head, str_res = create_fc(ele.get_weights(), "fc" + str(fc_ite))
        new_file_head += str_res_head
        new_file += str_res
        fc_ite += 1

file = open("testing.txt", "w")
file.write(new_file_head)
file.write(new_file)
file.close()
