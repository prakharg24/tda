import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Lambda, Flatten
from keras.layers import Input, Concatenate, Reshape
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from keras import backend as K
import time
from scipy import sparse
import random
from functools import partial
from scipy.stats import boxcox
from obspy.signal.detrend import polynomial
import joblib
from scipy import signal
from keras.models import load_model, save_model
import sys

random.seed(24)
np.random.seed(24)

import matplotlib.pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf

test_df = pd.read_csv('test_data_200_1500_random.csv')
# test_df = pd.read_csv('test_data_200_1500_class.csv')
train_df = pd.read_csv('train_data_200_1500_random.csv')
# train_df = pd.read_csv('train_data_200_1500_class.csv')
# test_df = pd.read_csv('test_update_feature_df_200_800_2.csv')
# train_df =pd.read_csv("training_data/train_6000_update_feature_df_200_800_2.csv")

# print(train_df)

class DataLoader():
    def __init__(self, X,y, y_out, batch_size, step,input_size,num_outputs, output_st, isFixed, mode):
        self.batch_size = batch_size
        self.step = step

        shift_X = []
        for ele in X:
            sg1 = ele[:len(ele)//3]
            sg2 = ele[len(ele)//3:2*len(ele)//3]
            sg3 = ele[2*len(ele)//3:]
            sg1p = [sg1[i] - sg1[i-1] for i in range(1, len(sg1))]
            sg2p = [sg2[i] - sg2[i-1] for i in range(1, len(sg2))]
            sg3p = [sg3[i] - sg3[i-1] for i in range(1, len(sg3))]
            new_ele = []
            new_ele.extend(sg1p)
            new_ele.extend(sg2p)
            new_ele.extend(sg3p)
            shift_X.append(new_ele)

        X = np.array(shift_X)
        # print(y)
        #
        # for i in range(10):
        #     signal1 = X[i][:len(X[i])//3]
        #     y_gr = [val for val in signal1]
        #     x = list(range(200, 2*len(signal1)+200, 2))
        #     plt.plot(x, y_gr, color='red')
        #     plt.title("Delay start : " + str(y_out[i][0]) + " -- Delay Value : " + str(y[i][0]))
        #     plt.savefig("plots/robustnew" + str(i))
        #     plt.clf()
        #
        # exit()

        X_shape = list(X.shape)
        # print(y_out)
        X_shape[-1] = int(X_shape[-1]/input_size)

        seq_length = int(X_shape[-1]/step)
        lengh = step*seq_length

        # print(step, seq_length)

        X = X.reshape((X_shape[0],input_size,-1))[:,:,:lengh]

        new_X = []
        new_y_cls = []
        new_y_reg = []
        new_y_pos = []

        for eX, ey, eyo in zip(X, y, y_out):
            if(eyo[0]<800):
                continue


            if(mode=="train"):
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    end_ind = min(len(eX[0])//step, (eyo[0]-200)//(2*step) + someite)
                    if(isFixed):
                        st_ind = end_ind - num_outputs - output_st
                    else:
                        st_ind = 1
                    new_X.append(eX[:,st_ind*step:end_ind*step])
                    # break
            else:
                # end_ind = (eyo[0]-200)//30 + num_outputs
                end_ind = len(eX[0])//step
                if(isFixed):
                    st_ind = end_ind - num_outputs - output_st
                else:
                    st_ind = 1
                new_X.append(eX[:,st_ind*step:end_ind*step])

                # end_ind = (eyo[0]-200)//30 + num_outputs//2
                # if(isFixed):
                #     st_ind = end_ind - num_outputs - output_st
                # else:
                #     st_ind = 1
                # new_X.append(eX[:,st_ind*step:end_ind*step])
                # print("H3", end_ind - st_ind)


            delay_st = output_st

            y_temp_reg = []
            y_temp_cls = []
            we_temp = []
            y_temp_reg.append([ey[0]])
            if(ey[0]==0.):
                y_temp_cls.append([0.])
            else:
                y_temp_cls.append([1.])

            if(mode=="train"):
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    new_y_cls.append(np.array(y_temp_cls))
                    # break
            else:
                new_y_cls.append(np.array(y_temp_cls))
            if(mode=="train"):
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    new_y_reg.append(np.array(y_temp_reg))
                    # break
            else:
                new_y_reg.append(np.array(y_temp_reg))

            new_y_pos.append(eyo[0])
            # print("Done")
            # if(len(new_X[-1][0])==40*step):
            #     print(end_ind)
            #     print(delay_st)
            #     print(np.shape(y_temp_cls))

        X = []
        shape_dict = {}
        for ele in new_X:
            # print(np.shape(ele))
            # ele_four = np.abs(np.fft.fft(ele, axis=1))
            # ele = np.concatenate((ele, ele_four), axis=0)
            # ele_temp = ele.reshape((2*input_size, -1, step))
            ele_temp = ele.reshape((input_size, -1, step))
            ele_temp = ele_temp.transpose((1, 2, 0))
            X.append(ele_temp)

        self.X = X
        print(list(np.shape(self.X)))

        self.y_cls = new_y_cls
        self.y_reg = new_y_reg
        self.y_pos = new_y_pos


    def dataset(self):
        return (self.X, self.y_cls, self.y_reg, self.y_pos)


params = {
    "epochs": 300,
    "mini_ep": 15,
    "batch_size": 64,
    "step": 15,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 512,
    "validation_split": 0.1,
    "input_size":3,
    "num_outputs":10,
    "output_st":10
}

def preprocess(df_inp):
    y_out = df_inp[['delay']]
    y_out_st = df_inp[['delay_st']]

    drop_terms = []

    drop_terms.extend(['delay', 'Unnamed: 0', 'delay_st'])
    X_out = df_inp.drop(drop_terms,axis=1)

    return np.array(X_out), np.array(y_out), np.array(y_out_st)

def poly_trend(X_inp):
    X_inp_poly = []
    for ele in X_inp:
        X_temp = []
        X_temp.extend(polynomial(ele[:len(ele)//3], 6, plot=False))
        X_temp.extend(polynomial(ele[len(ele)//3:2*len(ele)//3], 6, plot=False))
        X_temp.extend(polynomial(ele[2*len(ele)//3:], 6, plot=False))
        X_inp_poly.append(X_temp)

    return np.array(X_inp_poly)

def low_pass_trend(X_inp):
    X_inp_low = []
    b, a = signal.butter(5, 0.01, 'low')
    for ele in X_inp:
        X_temp = []
        X_temp.extend(ele[:len(ele)//3] - signal.filtfilt(b, a, ele[:len(ele)//3]))
        X_temp.extend(ele[len(ele)//3:2*len(ele)//3] - signal.filtfilt(b, a, ele[len(ele)//3:2*len(ele)//3]))
        X_temp.extend(ele[2*len(ele)//3:] - signal.filtfilt(b, a, ele[2*len(ele)//3:]))
        X_inp_low.append(X_temp)

    return np.array(X_inp_low)

X_test, y_test, y_test_out = preprocess(test_df)
X_train, y_train, y_train_out = preprocess(train_df)

# X_test = low_pass_trend(X_test)
# X_train = low_pass_trend(X_train)

print("Data Loaded")

# scaler_X = MinMaxScaler()
# scaler_X = MaxAbsScaler()
# scaler_X = StandardScaler()
scaler_X = RobustScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))
#
scaler_filename = "scalers/regression_scaler_robust.save"
joblib.dump(scaler_X, scaler_filename)

# scaler_filename = "scalers/regression_scaler_robust.save"
# scaler_X = joblib.load(scaler_filename)

# exit()

data_loader = DataLoader(scaler_X.transform(X_test),scaler_y.transform(y_test), np.array(y_test_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], False, "test")
X_test, y_test_classification, y_test_regression, y_test_pos = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train),scaler_y.transform(y_train), np.array(y_train_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], True, "train")
X_train, y_train_classification, y_train_regression, y_train_pos = data_loader.dataset()
# exit()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers


def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks

def build_classification_branch(inputs, hiddenCells):

    x = TimeDistributed(Dense(units=hiddenCells, activation='relu'))(inputs)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = TimeDistributed(Dense(units=1,activation='sigmoid'), name="classification_output")(x)
    # x = Flatten(name="classification_output")(x)

    return x

def build_regression_branch(inputs, hiddenCells):

    x = TimeDistributed(Dense(units=hiddenCells,activation='relu'))(inputs)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = TimeDistributed(Dense(units=1,activation='relu',kernel_regularizer=regularizers.l2(0.001)), name="regression_output")(x)
    # x = Flatten(name="regression_output")(x)

    return x

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 280:
        lr *= 0.5e-3
    elif epoch > 240:
        lr *= 1e-3
    elif epoch > 180:
        lr *= 1e-2
    elif epoch >120:
        lr *= 1e-1
    print('Learning rate: ', lr)
    # return lr
    return 1e-3

def rnn_walk_cls(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(None, layers[1], layers[4]))
    # weights_tensor = Input(shape=(layers[3],))

    # low0 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    # low0 = Dropout(params['dropout_keep_prob'])(low0)

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=256, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    # low2 = Reshape((-1, layers[1]*layers[4]))(inputs)
    # low2 = Reshape((-1, layers[4]))(inputs)

    # left0 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    # left0 = Dropout(params['dropout_keep_prob'])(left0)

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    left1 = Dropout(params['dropout_keep_prob'])(left1)

    left2 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left1)
    left2 = Dropout(params['dropout_keep_prob'])(left2)

    # left3 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left2)

    sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, layers[2]))(left2)

    classification_branch = build_classification_branch(sliced, layers[2])
    # regression_branch = build_regression_branch(sliced, layers[2])

    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    # model = Model(inputs=inputs, outputs=[classification_branch, regression_branch])
    model = Model(inputs=inputs, outputs=classification_branch)
    # model = Model(inputs=inputs, outputs=regression_branch)

    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    losses = {"classification_output": "binary_crossentropy", "regression_output": "mean_squared_error"}
    lossWeights = {"classification_output": 1.0, "regression_output": 0}

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    model.compile(loss='binary_crossentropy', optimizer=adam)
    # model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def rnn_walk_reg(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(None, layers[1], layers[4]))
    # weights_tensor = Input(shape=(layers[3],))

    # low0 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    # low0 = Dropout(params['dropout_keep_prob'])(low0)

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=256, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    # low2 = Reshape((-1, layers[1]*layers[4]))(inputs)
    # low2 = Reshape((-1, layers[4]))(inputs)

    # left0 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    # left0 = Dropout(params['dropout_keep_prob'])(left0)

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    left1 = Dropout(params['dropout_keep_prob'])(left1)

    left2 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left1)
    left2 = Dropout(params['dropout_keep_prob'])(left2)

    # left3 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left2)

    sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, layers[2]))(left2)

    # classification_branch = build_classification_branch(sliced, layers[2])
    regression_branch = build_regression_branch(sliced, layers[2])

    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    # model = Model(inputs=inputs, outputs=[classification_branch, regression_branch])
    # model = Model(inputs=inputs, outputs=classification_branch)
    model = Model(inputs=inputs, outputs=regression_branch)

    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    losses = {"classification_output": "binary_crossentropy", "regression_output": "mean_squared_error"}
    lossWeights = {"classification_output": 1.0, "regression_output": 0}

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    # model.compile(loss='binary_crossentropy', optimizer=adam)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

def rnn_walk_comb(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(None, layers[1], layers[4]))
    # weights_tensor = Input(shape=(layers[3],))

    # low0 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    # low0 = Dropout(params['dropout_keep_prob'])(low0)

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=256, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    # low2 = Reshape((-1, layers[1]*layers[4]))(inputs)
    # low2 = Reshape((-1, layers[4]))(inputs)

    # left0 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    # left0 = Dropout(params['dropout_keep_prob'])(left0)

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    left1 = Dropout(params['dropout_keep_prob'])(left1)

    left2 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left1)
    left2 = Dropout(params['dropout_keep_prob'])(left2)

    # left3 = LSTM(units=layers[2], return_sequences=True, activation='relu')(left2)

    sliced = Lambda(lambda x: x[:,20:,:], output_shape=(None, layers[2]))(left2)

    classification_branch = build_classification_branch(sliced, layers[2])
    regression_branch = build_regression_branch(sliced, layers[2])

    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=[classification_branch, regression_branch])
    # model = Model(inputs=inputs, outputs=classification_branch)
    # model = Model(inputs=inputs, outputs=regression_branch)

    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    losses = {"classification_output": "binary_crossentropy", "regression_output": "mean_squared_error"}
    lossWeights = {"classification_output": 1.0, "regression_output": 0}

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    # model.compile(loss='binary_crossentropy', optimizer=adam)
    # model.compile(loss='mean_squared_error', optimizer=adam)

    return model

last_loss = 10

# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
lstm_layer = [X_train[0].shape[0], X_train[0].shape[1], params['hidden_unit'], params['num_outputs'], X_train[0].shape[2]]
print(lstm_layer)
# model = rnn_walk_reg(lstm_layer, params)
# model = rnn_walk_cls(lstm_layer, params)
# old_model = rnn_walk_reg(lstm_layer, params)

cls_model = rnn_walk_cls(lstm_layer, params)
reg_model = rnn_walk_reg(lstm_layer, params)

model = rnn_walk_comb(lstm_layer, params)

# old_name = "LSTM_robust_aug1_sliding_reg_2_2"
saved_model = "LSTM_robust_aug1_sliding_reg_vanilla"
# saved_model = "LSTM_robust_aug1_sliding_reg_s_15"

reg_name = "LSTM_robust_aug1_sliding_reg_2_2"
cls_name = "LSTM_robust_aug1_sliding_cls_2_2"

# old_model.load_weights('arch_search/' + old_name + '_val')
reg_model.load_weights('arch_search/' + reg_name + '_val')
cls_model.load_weights('arch_search/' + cls_name + '_val')

# print(reg_model.summary())
# print(cls_model.summary())
# # print(old_model.summary())
# print(model.summary())

# exit()

# old_layers = ['time_distributed_5', 'time_distributed_6', 'time_distributed_7', 'lstm_9', 'lstm_10']
# new_layers = ['time_distributed_1', 'time_distributed_2', 'time_distributed_3', 'lstm_4', 'lstm_5']
# for layer, olayer in zip(new_layers, old_layers):
#     model.get_layer(name=layer).set_weights(old_model.get_layer(name=olayer).get_weights())
#     model.get_layer(name=layer).trainable = False
#
# adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
# model.compile(loss='binary_crossentropy', optimizer=adam)


old_layers = ['time_distributed_1', 'time_distributed_2', 'lstm_3', 'lstm_4', 'time_distributed_3', 'time_distributed_6', 'classification_output', 'regression_output']
new_layers = ['time_distributed_7', 'time_distributed_8', 'lstm_11', 'lstm_12', 'time_distributed_9', 'time_distributed_10', 'classification_output', 'regression_output']
regcls = [1, 1, 1, 1, 1, 0, 1, 0]
for flip, layer, olayer in zip(regcls, new_layers, old_layers):
    if(flip==0):
        model.get_layer(name=layer).set_weights(reg_model.get_layer(name=olayer).get_weights())
        # model.get_layer(name=layer).trainable = False
    else:
        model.get_layer(name=layer).set_weights(cls_model.get_layer(name=olayer).get_weights())

losses = {"classification_output": "binary_crossentropy", "regression_output": "mean_squared_error"}
lossWeights = {"classification_output": 1.0, "regression_output": 0}
adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)


print(model.summary())

print("Start training")
# loss =  0.03297723470484536
loss = 10
# val_loss = 0.0530523572180158
val_loss = 10

print(np.shape(X_train))
print(np.shape(y_train_regression))

zc = 0
nzc = 0
dict_X = {}
dict_y_reg = {}
dict_y_cls = {}
for ele, yer, yec in zip(X_train, y_train_regression, y_train_classification):
    if(ele.shape[0] not in dict_X):
        dict_X[ele.shape[0]] = []
        dict_y_reg[ele.shape[0]] = []
        dict_y_cls[ele.shape[0]] = []

    # print(ele.shape)
    # print(ye.shape)
    for yee in yec:
        if(yee[0]==0.):
            zc += 1
        else:
            nzc += 1

    dict_X[ele.shape[0]].append(ele)
    dict_y_reg[ele.shape[0]].append(yer)
    dict_y_cls[ele.shape[0]].append(yec)

print(zc, nzc)
# exit()
from keras.models import Model
# # #
# model.load_weights("arch_search/%s" % (saved_model))

# exit()

print(len(dict_X))

params['epochs'] = 100
#
# for ele in dict_X:
# #     # print(np.shape(dict_X[ele]))
# #     # print(np.shape(dict_y[ele]))
# #     # history = model.fit(np.array(dict_X[ele]), [np.array(dict_y_cls[ele]), np.array(dict_y_reg[ele])],
#     history = model.fit(np.array(dict_X[ele]), np.array(dict_y_reg[ele]),
#     # history = model.fit(np.array(dict_X[ele]), np.array(dict_y_cls[ele]),
#               batch_size=params['batch_size'],
#               # epochs=params['epochs'],
#               epochs = params['epochs'],
#               validation_split=params['validation_split'],
#               verbose=1,
#               callbacks = [ModelCheckpoint(filepath="arch_search/"+saved_model,monitor='loss',verbose=1, save_best_only=True,save_weights_only=True),\
#                           ModelCheckpoint(filepath="arch_search/"+saved_model+"_val",monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)]
#               )
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.legend(['train', 'val'], loc='upper left')
#     plt.ylim((0, 0.015))
#     plt.savefig('fulldata_robust.jpg')
#     np.save('fulldata_robust_loss', history.history['loss'])
#     np.save('fulldata_robust_val_loss', history.history['val_loss'])

# exit()
print("Predicting")
st = time.time()

# predictions = model.predict(X_test)
#
# print(predictions)

# print(predictions)
#
# for ele in predictions:
#     for j in range(0, 6):
#         if(ele[j]!=ele[j+1]):
#             print(ele)
#             break
#
# exit()

# predict = []
# predict_cls = []
# y_true_cls = []
# y_true_reg = []

# dict_X = {}
# dict_y_cls = {}
# dict_y_reg = {}
# for ele, yce, yre in zip(X_test, y_test_classification, y_test_regression):
#     if(ele.shape[0] not in dict_X):
#         dict_X[ele.shape[0]] = []
#         dict_y_cls[ele.shape[0]] = []
#         dict_y_reg[ele.shape[0]] = []
#
#     # print(ele.shape)
#     # print(ye.shape)
#
#     dict_X[ele.shape[0]].append(ele)
#     dict_y_cls[ele.shape[0]].append(yce)
#     dict_y_reg[ele.shape[0]].append(yre)

# print(dict_X)
# print(dict_y)
#
# exit()

print("Different Length predicting X ", len(dict_X))

fnl_pred = model.predict(np.array(X_test))

# for i in range(len(fnl_pred[0])):
#     del_st = (y_test_pos[i]-800)//30 + 1
#     cls_arr = [round(ele, 0) for ele in fnl_pred[0][i][del_st:, 0]]
#     reg_arr = [round(ele*50, 0) for ele in fnl_pred[1][i][del_st:, 0]]
#
#     if(y_test_regression[i][0][0]==0):
#         continue
#     reg_val = round(y_test_regression[i][0][0]*50, 0)
#     print(cls_arr)
#     print(reg_arr)
#     use_reg = False
#     print(reg_val)
#     x_arr = []
#     y_arr = []
#     for j in range(0, min(11, len(cls_arr))):
#         x_arr.append(j/11)
#         if(cls_arr[j]==1):
#             use_reg = True
#         if(use_reg):
#             y_arr.append(abs(reg_arr[j] - reg_val)/reg_val)
#         else:
#             y_arr.append(abs(0. - reg_val)/reg_val)

    # plt.xlim(0, 1)
    # plt.ylim(0, 1)
    # plt.plot(x_arr, y_arr, color='blue')
    # plt.show()
    # if(i==20):
    #     break

# exit()

# predict = []
# predict_cls = []
# dist_arr = []
#
# corr_count = 0
# inc_count = 0
# for i in range(len(fnl_pred[0])):
#     del_st = (y_test_pos[i]-800)//30 + 1
#     if(y_test_regression[i][0][0]==0):
#         cls_arr = [round(ele, 0) for ele in fnl_pred[0][i][:, 0]]
#         reg_arr = fnl_pred[1][i][:, 0]
#         # cls_arr = []
#         # reg_arr = []
#     else:
#         cls_arr = [round(ele, 0) for ele in fnl_pred[0][i][:del_st, 0]]
#         reg_arr = fnl_pred[1][i][:del_st, 0]
#
#     # for j in range(len(cls_arr)-1):
#     #     ec = cls_arr[j]
#     #     ec1 = cls_arr[j+1]
#     #     # ec1 = 1
#     #     er = round(reg_arr[j+1]*50, 0)
#     #     # if(er>50):
#     #     #     print(er)
#     #     if(ec==1 and ec1==1):
#     #         inc_count += 1
#     #         predict.append(er)
#     #         # break
#     #     else:
#     #         corr_count += 1
#     flag = -1
#     for j in range(len(cls_arr)-1):
#         ec = cls_arr[j]
#         ec1 = cls_arr[j+1]
#         # er = round(reg_arr[j+1]*50, 0)
#         # et = round(y_test_regression[i][0][0]*50, 0)
#         if(ec==1 and ec1==1):
#             # dist_arr.append(30*(j+1) + 30 - (y_test_pos[i]-800)%30)
#             # print(er, et)
#             # predict.append(abs(er - et))
#             flag = j
#             break
#
#     if(len(cls_arr)==0):
#         continue
#     if(flag!=-1):
#         corr_count += 1
#         for j in range(flag, len(cls_arr)-1):
#             er = round(reg_arr[j]*50, 0)
#             er1 = round(reg_arr[j+1]*50, 0)
#             et = round(y_test_regression[i][0][0]*50, 0)
#             if(abs(er - er1) <=1 or j==len(cls_arr)-2):
#                 dist_arr.append(30*(j+1) + 30 - (y_test_pos[i]-800)%30)
#                 # predict.append(abs(er1 - et))
#                 break
#     else:
#         inc_count += 1
#         predict.append(round(y_test_regression[i][0][0]*50, 0))
#
# # print(predict)
# print(np.mean(predict))
# print(np.mean(dist_arr))
# print(inc_count, corr_count)
# # plt.hist(predict, bins=50)
# # plt.gca().set(title='Frequency Histogram', ylabel='Frequency')
# # plt.show()
#
# exit()

predict = []
predict_cls = []
dist_arr = []
cls_dist = []
for i in range(len(fnl_pred[0])):
    del_st = (y_test_pos[i]-800)//30 + 1
    cls_arr = [round(ele, 0) for ele in fnl_pred[0][i][del_st:, 0]]
    reg_arr = fnl_pred[1][i][del_st:, 0]


    some_res = -1
    skip = 0
    for j in range(0, min(11, len(cls_arr))):
        if(cls_arr[j]==1):
            if(skip==0):
                some_res = j
                break
            skip -= 1
        else:
            skip = 0

    # if(some_res!=-1):
    #     cls_dist.append(some_res)

    if(some_res!=-1):
        # print(reg_arr[:10]*50, y_test_regression[i]*50, i)
        # continue
    # if(False):
        predict_cls.append([cls_arr[some_res]])
        chng_thr = 5
        trigger = 35
        # wait = int(sys.argv[1])
        prev_pred = reg_arr[min(some_res, len(reg_arr)-1)]
        for reg_ite in range(1, 11):
            # dist_arr.append(wait)
            # predict.append([reg_arr[min(some_res+wait, len(reg_arr)-1)]])
            # break
            # if(reg_ite==6):
            #     dist_arr.append(reg_ite-1)
            #     predict.append([prev_pred])
            #     break
            # if(prev_pred*50>trigger):
            #     dist_arr.append(reg_ite-1)
            #     predict.append([prev_pred])
            #     break
            # prev_pred = reg_arr[min(some_res+reg_ite, len(reg_arr)-1)]

            curr_pred = reg_arr[min(some_res+reg_ite, len(reg_arr)-1)]
            if(prev_pred==0):
                dist_arr.append(reg_ite -1)
                predict.append([prev_pred])
                break
            elif(abs(curr_pred - prev_pred)/prev_pred < chng_thr/100):
                dist_arr.append(reg_ite)
                predict.append([curr_pred])
                break
            elif(reg_ite==10):
                dist_arr.append(reg_ite)
                predict.append([curr_pred])
                break
            prev_pred = curr_pred
    else:
        predict.append([0.])
        # predict.append([reg_arr[min(10, len(reg_arr)-1)]])
        predict_cls.append([0.])
        # dist_arr.append(-1)
# exit()
print("Average Distance", np.mean(dist_arr)*30)
print("Standard Deviation Distance", np.std(dist_arr)*30)
print("Maximum Distance", np.amax(dist_arr)*30)
print("Minimum Distance", np.amin(dist_arr)*30)

# exit()

# for ele in dict_X:
# predict_cls = np.array([eee[-1] for eee in fnl_pred[0]])
# predict = np.array([eee[-1] for eee in fnl_pred[1]])
y_true_reg = np.array([eee[0] for eee in y_test_regression])
y_true_cls = np.array([eee[0] for eee in y_test_classification])

print(np.shape(predict))
print(np.shape(y_true_reg))
# exit()
#
# print("Time taken", time.time() - st)

y_true_reg = scaler_y.inverse_transform(np.array(y_true_reg))
predict = scaler_y.inverse_transform(np.array(predict))

# print(np.shape(y_true_cls))
# print(np.shape(predict))

# plt.ion()
# for e1, e2 in zip(predict, y_true_cls):
#     if(np.random.randint(0,10)>2):
#         continue
#     print(np.shape(e1), np.shape(e2))
#
#     lbl_arr = []
#     for i in range(len(e2)):
#         lbl_arr.append(290 + (params["output_st"] +1)*30 + i*30)
#     # plt.plot(lbl_arr, [x[0] for x in e1])
#     plt.plot(lbl_arr, [x[0] for x in e1], color='red')
#     plt.plot(lbl_arr, [x[0] for x in e2], color='blue')
#
#     plt.show()
#     plt.pause(0.5)
#     plt.close()
#     plt.clf()
# # plt.savefig('classification')


# conf_mat = np.zeros((2, 2))
# for e1, e2 in zip(predict, y_true_cls):
#     arr1 = [x[0] for x in e1]
#     arr2 = [x[0] for x in e2]
#     for a1, a2 in zip(arr1, arr2):
#         conf_mat[int(a1>0.5)][int(a2>0)] += 1
#
# print(conf_mat)
# exit()

# for ele in predict:
#     flag = False
#     for i, pred in enumerate(ele):
#         # if(round(pred[0], 0)==1):
#         if((not flag) and pred[0]>0.5):
#             if(i!=7):
#                 print("Start :", 400 + i*30)
#             flag = True
#         if(flag and pred[0]<0.5):
#             if(i!=17):
#                 print("End :", 400 + i*30)
#             flag = False


# print(np.shape(y_true))
# print(np.shape(predict))

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


print("Predicting")
st = time.time()

# predict = model.predict(np.array(X_test))
# predict =  scaler_y.inverse_transform([ele[0] for ele in predict])

predict_cls = np.array([[round(x[0], 0)] for x in predict_cls])

# predict_reg = np.array([[round(x[0], 0)] for x in predict])
predict_reg = []
for i in range(len(predict)):
    if(int(predict_cls[i])==1):
        predict_reg.append([round(predict[i][0], 0)])
    elif(int(predict_cls[i])==0):
        predict_reg.append([0.])
    else:
        print("Weird")

predict_reg = np.array(predict_reg)

conf_mat = [[0, 0], [0, 0]]

print("Complete")
for i in range(len(predict_cls)):
    conf_mat[int(predict_cls[i])][int(y_true_cls[i])] += 1

print(np.array(conf_mat))

print((conf_mat[0][0]+conf_mat[1][1])/len(predict_cls))
print((conf_mat[1][0])/len(predict_cls))
print((conf_mat[0][1])/len(predict_cls))

jmp = 1
for lim in range(0, 10, jmp):
    conf_mat = [[0, 0], [0, 0]]
    print("Limit ", lim, " to ", lim+jmp)
    for i in range(len(predict_cls)):
        if(y_true_reg[i]>lim and y_true_reg[i]<=lim+jmp):
            conf_mat[int(predict_cls[i])][int(y_true_cls[i])] += 1
    print(np.array(conf_mat))


# predict = model.predict(X_test)
# predict = np.argmax(predict, axis=1)
print("Time taken", time.time() - st)
#
# y_true = np.argmax(y_test, axis=1)
#
# print(y_true)

# print(np.array(y_test_out))

# y_true = []
# for a, b in zip(np.array(y_test), np.array(y_test_out)):
#     # print(b[0])
#     if(b[0]>=800):
#         y_true.append(a)
# # y_true  =  scaler_y.inverse_transform(y_test)
# y_true = np.array(y_true)
# print(np.shape(y_true))

def NRMSD(y_true, y_pred):
    rmsd = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    y_min = min(y_true)
    y_max = max(y_true)

    return rmsd/(y_max - y_min)

def MAPE(y_true, y_pred):
    print(np.shape(y_true))
    print(np.shape(y_pred))
    y_true_select = (y_true!=0)

    y_true = y_true[y_true_select]
    y_pred = y_pred[y_true_select]

    errors = y_true - y_pred
    return sum(abs(errors/y_true))*100.0/len(y_true)

# for e1, e2 in zip(y_true_reg, predict):
#     plt.plot(e1[0], e2[0], 'ro', color='blue')
#
# plt.plot([0, 50], [0, 50], color='black')
# plt.savefig('Figure_walk_grouping')
mae_arr = []
mape_arr = []
# for i in range(5):
    # st = 10*i
    # en = 10*(i+1)
    # print(st, en)

# filtered = np.array([[ele1, ele2] for ele1, ele2 in zip(y_true_reg, predict_reg) if (ele1[0]<=en and ele1[0]>st)])
filtered = np.array([[ele1, ele2] for ele1, ele2 in zip(y_true_reg, predict_reg)])

y_true_reg_t = filtered[:, 0, :]
predict_reg_t = filtered[:, 1, :]

nrmsd = NRMSD(y_true_reg_t, predict_reg_t)
mape  = MAPE(y_true_reg_t, predict_reg_t)
mae   = mean_absolute_error(y_true_reg_t, predict_reg_t)
rmse   = np.sqrt(mean_squared_error(y_true_reg_t, predict_reg_t))
print ("NRMSD",nrmsd)
print ("MAPE",mape)
print ("neg_mean_absolute_error",mae)
print ("Root mean squared error",rmse)
mae_arr.append(mae)
mape_arr.append(mape)
#
# df = pd.DataFrame({"predict":predict.flatten(),"y_true": y_true.flatten()})
# df.to_csv('result-%s.csv' % (saved_model),index=True, header=True)

for ele in mae_arr:
    print(ele)

for ele in mape_arr:
    print(ele)

from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
