from dataloader import PPCSDataLoader
from hlstm_model import HLSTM


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

from scipy import signal
from keras.models import load_model, save_model
import sys

random.seed(24)
np.random.seed(24)

import matplotlib.pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg


import argparse
import joblib
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--reg_csv', type=str, help='CSV file containing the regression dataset', default='dataset/train_data_200_1500_regression.csv')
parser.add_argument('--cls_csv', type=str, help='CSV file containing the classification dataset', default='dataset/train_data_200_1500_classification.csv')

parser.add_argument('--lower_step', type=int, help='Number of steps for the lower LSTM before being reset', default=15)
parser.add_argument('--sensor_channels', type=int, help='Number of different sensors in the dataset', default=3)
parser.add_argument('--window_length', type=int, help='window_length * lower_step = Length of the sliding window', default=20)
parser.add_argument('--start_overhead', type=int, help='Number of initial outputs rejected', default=10)
parser.add_argument('--sliding_step', type=int, help='Slide value while creating training dataset', default=1)

parser.add_argument('--upper_depth', type=int, help='Depth of the Upper LSTM', default=2)
parser.add_argument('--lower_depth', type=int, help='Depth of the Lower LSTM', default=2)
parser.add_argument('--dense_hidden_units', type=int, help='Number of hidden units in the dense fully connected layer', default=512)
parser.add_argument('--lower_lstm_units', type=int, help='Number of hidden units in the upper LSTM', default=512)
parser.add_argument('--upper_lstm_units', type=int, help='Number of hidden units in the lower LSTM', default=256)
parser.add_argument('--dropout', type=float, help='Dropout percentage', default=0.1)

args = parser.parse_args(sys.argv[1:])

regression_data = PPCSDataLoader(args.reg_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, args.sliding_step,
                                 'train')

scalerX, scalerY = regression_data.get_scalers()
joblib.dump(scalerX, 'scalerX')
joblib.dump(scalerY, 'scalerY')

classification_data = PPCSDataLoader(args.cls_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, args.sliding_step,
                                 'train', scalerX=scalerX, scalerY=scalerY)

print("Data Loaded")

regx, regy_classification, regy_regression, regy_pos = regression_data.get_data()
clsx, clsy_classification, clsy_regression, clsy_pos = classification_data.get_data()

print("Setting up training environment")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

from keras.callbacks import ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers


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
