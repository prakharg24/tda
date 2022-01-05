import numpy as np
from keras.layers.core import Dense, Activation, Dropout, Lambda, Flatten
from keras.layers import Input, Concatenate
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras import backend as K
import time
from scipy import sparse
import random
from functools import partial
from scipy.stats import boxcox
import joblib
from obspy.signal.detrend import polynomial

random.seed(24)

import matplotlib.pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf

test_df = pd.read_csv('test_data_200_1500_class.csv')
train_df = pd.read_csv('train_data_200_1500_class.csv')

# test_df = pd.read_csv('../data_trace/test_feature_df_100_900_2.csv')
# train_df = pd.read_csv('../data_trace/train_feature_df_100_900_2.csv')

# print(train_df)

class DataLoader():
    def __init__(self, X,y, y_out, batch_size, step,input_size,num_outputs, output_st, isFixed):
        self.batch_size = batch_size
        self.step = step

        # shift_X = []
        # for ele in X:
        #     sg1 = ele[:len(ele)//3]
        #     sg2 = ele[len(ele)//3:2*len(ele)//3]
        #     sg3 = ele[2*len(ele)//3:]
        #     sg1p = [sg1[i] - sg1[i-1] for i in range(1, len(sg1))]
        #     sg2p = [sg2[i] - sg2[i-1] for i in range(1, len(sg2))]
        #     sg3p = [sg3[i] - sg3[i-1] for i in range(1, len(sg3))]
        #     new_ele = []
        #     new_ele.extend(sg1p)
        #     new_ele.extend(sg2p)
        #     new_ele.extend(sg3p)
        #     shift_X.append(new_ele)
        #
        # X = np.array(shift_X)
        # # print(y)
        #
        # for i in range(10):
        #     signal1 = X[i][:len(X[i])//3]
        #     y_gr = [val for val in signal1]
        #     x = list(range(200, 2*len(signal1)+200, 2))
        #     plt.plot(x, y_gr)
        #     plt.title("Delay start : " + str(y_out[i][0]) + " -- Delay Value : " + str(y[i][0]))
        #     plt.savefig("plots/trend" + str(i))
        #     plt.clf()
        #
        # exit()

        X_shape = list(X.shape)
        print(X_shape)
        # print(y_out)
        X_shape[-1] = int(X_shape[-1]/input_size)

        seq_length = int(X_shape[-1]/step)
        lengh = step*seq_length

        # print(step, seq_length)

        X = X.reshape((X_shape[0],input_size,-1))[:,:,:lengh]

        new_X = []
        new_y_cls = []
        new_y_reg = []

        for eX, ey, eyo in zip(X, y, y_out):
            if(eyo[0]<800):
                continue

            end_ind = (eyo[0]-200)//30 + num_outputs
            if(isFixed):
                st_ind = end_ind - num_outputs - output_st
            else:
                st_ind = 1
            new_X.append(eX[:,st_ind*15:end_ind*15])


            delay_st = output_st

            y_temp_reg = []
            y_temp_cls = []
            we_temp = []
            y_temp_reg.append([ey[0]])
            if(ey[0]==0.):
                y_temp_cls.append([0.])
            else:
                y_temp_cls.append([1.])

            new_y_cls.append(np.array(y_temp_cls))
            new_y_reg.append(np.array(y_temp_reg))
            # print("Done")
            # if(len(new_X[-1][0])==40*15):
                # print(end_ind)
                # print(delay_st)
                # print(np.shape(y_temp_cls))

        X = []

        for ele in new_X:
            # print(np.shape(ele))
            # ele_four = np.abs(np.fft.fft(ele, axis=1))
            # ele = np.concatenate((ele, ele_four), axis=0)
            # ele_temp = ele.reshape((2*input_size, -1, step))
            ele_temp = ele.reshape((input_size, -1, step))
            ele_temp = ele_temp.transpose((1, 2, 0))
            X.append(ele_temp)

        self.X = X

        self.y_cls = new_y_cls
        self.y_reg = new_y_reg


    def dataset(self):
        return (self.X, self.y_cls, self.y_reg)


params = {
    "epochs": 300,
    "mini_ep": 15,
    "batch_size": 64,
    "step": 15,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 512,
    "validation_split": 0.1,
    "input_size":6,
    "num_outputs":10,
    "output_st":10
}

def preprocess(df_inp):
    y_out = df_inp[['delay']]
    y_out_st = df_inp[['delay_st']]

    drop_terms = []
    # prev_ele = None
    #
    # columns = list(df_inp)
    # for ele in columns:
    #     if(prev_ele is None):
    #         prev_ele = ele
    #         continue
    #     if(prev_ele[:-4]!=ele[:-4]):
    #         drop_terms.append(prev_ele)
    #         prev_ele = ele
    #         continue
    #     df_inp[prev_ele] = df_inp[prev_ele] - df_inp[ele]
    #     prev_ele = ele

    # drop_terms.extend(['period','powerSetPoint','sigma','delay'])
    drop_terms.extend(['delay', 'Unnamed: 0', 'delay_st'])
    X_out = df_inp.drop(drop_terms,axis=1)

    # X_out = np.array(X_out)

    return np.array(X_out), np.array(y_out), np.array(y_out_st)

X_test, y_test, y_test_out = preprocess(test_df)
X_train, y_train, y_train_out = preprocess(train_df)

signal = X_train[2][:len(X_train[0])//3]
polynomial(signal, order=6, plot=True)
exit()

trend_X = []
for ele in X_train:
    sg1 = ele[:len(ele)//3]
    sg2 = ele[len(ele)//3:2*len(ele)//3]
    sg3 = ele[2*len(ele)//3:]
    polynomial(sg1, order=6)
    polynomial(sg2, order=6)
    polynomial(sg3, order=6)
    new_ele = []
    new_ele.extend(sg1)
    new_ele.extend(sg2)
    new_ele.extend(sg3)
    new_ele.extend(ele-new_ele)
    trend_X.append(new_ele)

X_train = np.array(trend_X)

trend_X = []
for ele in X_test:
    sg1 = ele[:len(ele)//3]
    sg2 = ele[len(ele)//3:2*len(ele)//3]
    sg3 = ele[2*len(ele)//3:]
    polynomial(sg1, order=6)
    polynomial(sg2, order=6)
    polynomial(sg3, order=6)
    new_ele = []
    new_ele.extend(sg1)
    new_ele.extend(sg2)
    new_ele.extend(sg3)
    new_ele.extend(ele-new_ele)
    trend_X.append(new_ele)

X_test = np.array(trend_X)

# old_signal = np.copy(X_train[0][:len(X_train[0])//3])
# plt.plot(old_signal, color='red')
# polynomial(old_signal, order=3)
# plt.plot(X_train[0][:len(X_train[0])//3] - old_signal, color='blue')
# plt.show()
# exit()
# X_running = []
# for ele, ey, eyo in zip(X_test, y_test, y_test_out):
#     if(eyo[0]>800 and ey[0]==0):
#         X_running.append(ele)
# for ele, ey, eyo in zip(X_train, y_train, y_train_out):
#     if(eyo[0]>800 and ey[0]==0):
#         X_running.append(ele)
#
# print(len(X_running))
#
# X_len = len(X_running)
# X_sum = np.sum(X_running, axis=0)
# X_sum = X_sum/X_len
#
# signal1 = X_sum[:len(X_sum)//3]
# y_gr = [val for val in signal1]
# x = list(range(200, 2*len(signal1)+200, 2))
# plt.plot(x, y_gr)
# plt.show()
# # exit()
# # print(X_test[0])
# X_test = X_test - X_sum
# X_train = X_train - X_sum


# print(X_test[0])
# exit()

print("Data Loaded")

# scaler_X = MinMaxScaler()
# scaler_X = StandardScaler()
scaler_X = RobustScaler()
scaler_y = MinMaxScaler()


# scaler_filename = "robust_scaler.save"
# scaler_X = joblib.load(scaler_filename)

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

data_loader = DataLoader(scaler_X.transform(X_test),np.array(y_test), np.array(y_test_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], True)
X_test, y_test_classification, y_test_regression = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train),np.array(y_train), np.array(y_train_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], True)
X_train, y_train_classification, y_train_regression = data_loader.dataset()


from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint, LambdaCallback
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

    x = Dense(units=hiddenCells, activation='relu')(inputs)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=1,activation='sigmoid',kernel_regularizer=regularizers.l2(0.001), name="classification_output")(x)
    # x = Flatten(name="classification_output")(x)

    return x

def build_regression_branch(inputs, hiddenCells):

    x = Dense(units=hiddenCells,activation='relu')(inputs)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=1,activation='relu',kernel_regularizer=regularizers.l2(0.001), name="regression_output")(x)
    # x = Flatten(name="regression_output")(x)

    return x

def rnn_walk_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(None, layers[1], layers[4]))
    # weights_tensor = Input(shape=(layers[3],))

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=256, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

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
    # model = Model(inputs=[inputs, weights_tensor], outputs=[classification_branch, regression_branch])
    model = Model(inputs=inputs, outputs=classification_branch)
    # model = Model(inputs=inputs, outputs=regression_branch)

    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    # losses = {"classification_output": "binary_crossentropy", "regression_output": cl4}
    # lossWeights = {"classification_output": 1.0, "regression_output": 1.5}

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    model.compile(loss='binary_crossentropy', optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam')

    return model

last_loss = 10

from keras.models import load_model, save_model
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
lstm_layer = [X_train[0].shape[0], X_train[0].shape[1], params['hidden_unit'], params['num_outputs'], X_train[0].shape[2]]
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
model = rnn_walk_lstm(lstm_layer, params)

# old_model = load_model("models_window/" + "LSTM_L2_walk_ahead_fixed_trace_robust_shift.h5")
# # model = load_model("models_window/" + "LSTM_L2_walk_ahead_fixed_trace_robust_shift.h5")
#
# # for ele in model.get_weights():
# #     print(ele)
# #     break
# #
# # for ele in model.get_weights()[::-1][1:]:
# #     print(ele)
# #     break
#
# for layer, old_layer in zip(model.layers, old_model.layers):
#     print(layer.name, old_layer.name)
#     if('lstm' in layer.name or 'time_distributed' in layer.name):
#         model.get_layer(name=layer.name).set_weights(old_model.get_layer(name=old_layer.name).get_weights())
#         model.get_layer(name=layer.name).trainable = False
#
# model.compile(loss='binary_crossentropy', optimizer='adam')
# exit()
saved_model = "LSTM_L2_walk_ahead_fixed_trace_trend_6"

# for ele in model.get_weights():
#     print(ele)
#     break
#
# for ele in model.get_weights()[::-1][1:]:
#     print(ele[:10])
#     break
#
# model.load_weights("models_window/BI_LSTM_L2_window_val")

# model = load_model("models_exp/BI_LSTM_L2_heirar_comp-6000")
# # model.load_weights("models_pruned/BI_LSTM_L2_dropPruning_ext-6000")
# # pruningArr = [46., 76., 86., 66., 36.]
#
# sess = tf.keras.backend.get_session()
# tf.contrib.quantize.create_training_graph(sess.graph)
# sess.run(tf.global_variables_initializer())
#
# adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
# model.compile(loss="mean_squared_error", optimizer=adam)

print(model.summary())
# exit()
# get_zeroes(model, 10)

# all_w = model.get_weights()
# # np.savez_compressed('size/pruned', all_w)
# for i, ele in enumerate(all_w):
#     # print(ele)
#     # print(np.amax(np.abs(ele)))
#     # print(np.amin(np.abs(ele[np.nonzero(ele)])))
#     sA = sparse.csr_matrix(ele)
#     sparse.save_npz('size/scipy_dynamic_quant/' + str(i), sA)
#
# exit()
'''
from keras.models import load_model
try:
    df_his = pd.read_csv("history_%s.csv" %(saved_model),index_col=0)
    model = load_model("models_pruned/%s" % (saved_model))
except:
    print("re train")
    df_his = None
'''
df_his=None

print("Start training")
# loss =  0.03297723470484536
loss = 10
# val_loss = 0.0530523572180158
val_loss = 10

zc = 0
nzc = 0
dict_X = {}
dict_y = {}
for ele, ye in zip(X_train, y_train_classification):
    if(ele.shape[0] not in dict_X):
        dict_X[ele.shape[0]] = []
        dict_y[ele.shape[0]] = []

    # print(ele.shape)
    # print(ye.shape)
    for yee in ye:
        if(yee[0]==0.):
            zc += 1
        else:
            nzc += 1

    dict_X[ele.shape[0]].append(ele)
    dict_y[ele.shape[0]].append(ye)

print(zc, nzc)
# exit()
from keras.models import Model
# # #
# model.load_weights("models_window/%s" % (saved_model))

# exit()

print(len(dict_X))

# params['epochs'] = 100

# for i in range(params['epochs']):
    # print("EPOCH ", i)
hist_loss = 0
hist_val_loss = 0
ttl_len = 0

# model.load_weights("models_window/%s" % (saved_model))
#
# for ele in dict_X:
# #     # print(np.shape(dict_X[ele]))
# #     # print(np.shape(dict_y[ele]))
#     history = model.fit(np.array(dict_X[ele]), np.array(dict_y[ele]),
#               batch_size=params['batch_size'],
#               # epochs=params['epochs'],
#               epochs = params['epochs'],
#               validation_split=params['validation_split'],
#               verbose=1,
#               callbacks = [ModelCheckpoint(filepath="models_window/"+saved_model,monitor='loss',verbose=1, save_best_only=True,save_weights_only=True),\
#                           ModelCheckpoint(filepath="models_window/"+saved_model+"_val",monitor='val_loss',verbose=1, save_best_only=True,save_weights_only=True)]
#               )
    # hist_loss += len(dict_X[ele])*(history.history['loss'][0]**2)
    # hist_val_loss += len(dict_X[ele])*(history.history['val_loss'][0]**2)
    # ttl_len += len(dict_X[ele])
#
# print("Total Training Size", ttl_len)
# for ele in model.get_weights():
#     print(ele)
#     break
#
# for ele in model.get_weights()[::-1][1:]:
#     print(ele)
#     break

# hist_loss = (hist_loss/ttl_len)**0.5
# hist_val_loss = (hist_val_loss/ttl_len)**0.5
#
# if(hist_loss<loss):
#     print("Training Loss Improved from ", loss, " to ", hist_loss)
#     model.save_weights("models_window/%s" % (saved_model))
#     loss = hist_loss
# else:
#     print("Training Loss ", hist_loss)
#
# if(hist_val_loss<val_loss):
#     print("Validation Loss Improved from ", val_loss, " to ", hist_val_loss)
#     model.save_weights("models_window/%s" % (saved_model) + "_val")
#     val_loss = hist_val_loss
# else:
#     print("Validation Loss ", hist_val_loss)
#
# print("Final Losses ", loss, val_loss)

#
# # model.save_weights("models_pruned/%s" % (saved_model) + ".h5")
# # print(model.summary())
#
# # In[6]:
# if df_his is None:
#     df = pd.DataFrame(history.history)
#     df.to_csv("history_%s.csv" %(saved_model),header=True)
# else:
#     df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
#     df.to_csv("history_%s.csv" %(saved_model),header=True)


# model = load_model("models_pruned/%s" % (saved_model))
#
# a = np.array(model.get_weights())
# for ele in a:
#     med = np.percentile(ele, 90)
#     ele[ele < med] = 0
# model.set_weights(a)

model.load_weights("models_window/%s" % (saved_model) + "_val")


# for ele in model.get_weights():
#     print(ele)
#     break
#
# for ele in model.get_weights()[::-1][1:]:
#     print(ele[:10])
#     break

# save_model(model, "models_window/%s" % (saved_model) + ".h5")
# model.save("models_pruned/BI_LSTM_L2_dropPruning_new_fine-6000_model_val")

# model = tf.keras.models.load_model("models_pruned/BI_LSTM_L2_dropPruning_new-6000_model_val")

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

predict = []
y_true_cls = []
y_true_reg = []

dict_X = {}
dict_y_cls = {}
dict_y_reg = {}
for ele, yce, yre in zip(X_test, y_test_classification, y_test_regression):
    if(ele.shape[0] not in dict_X):
        dict_X[ele.shape[0]] = []
        dict_y_cls[ele.shape[0]] = []
        dict_y_reg[ele.shape[0]] = []

    # print(ele.shape)
    # print(ye.shape)

    dict_X[ele.shape[0]].append(ele)
    dict_y_cls[ele.shape[0]].append(yce)
    dict_y_reg[ele.shape[0]].append(yre)

# print(dict_X)
# print(dict_y)
#
# exit()


print("Different Length predicting X ", len(dict_X))

for ele in dict_X:
    predict.extend([eee[0][0] for eee in model.predict(np.array(dict_X[ele]))])
    y_true_cls.extend([int(eee[0][0]) for eee in dict_y_cls[ele]])
    y_true_reg.extend([int(eee[0][0]) for eee in dict_y_reg[ele]])

print(np.shape(predict))
print(np.shape(y_true_cls))

# print(predict[:5])
# exit()
#
# print("Time taken", time.time() - st)

# y_true_reg = scaler_y.inverse_transform(np.array(y_true_reg))
# predict = scaler_y.inverse_transform(np.array(predict))

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

predict = np.array([int(round(x, 0)) for x in predict])

print(np.shape(predict))
print(np.shape(y_true_cls))
print(np.shape(y_true_reg))

conf_mat = [[0, 0], [0, 0]]

print("Complete")
for i in range(len(predict)):
    conf_mat[predict[i]][y_true_cls[i]] += 1

print(np.array(conf_mat))

jmp = 10
for lim in range(0, 50, jmp):
    conf_mat = [[0, 0], [0, 0]]
    print("Limit ", lim, " to ", lim+jmp)
    for i in range(len(predict)):
        if(y_true_reg[i]>lim and y_true_reg[i]<=lim+jmp):
            conf_mat[predict[i]][y_true_cls[i]] += 1
    print(np.array(conf_mat))

exit()
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

nrmsd = NRMSD(y_true_reg, predict)
mape  = MAPE(y_true_reg, predict)
mae   = mean_absolute_error(y_true_reg, predict)
rmse   = np.sqrt(mean_squared_error(y_true_reg, predict))
print ("NRMSD",nrmsd)
print ("MAPE",mape)
print ("neg_mean_absolute_error",mae)
print ("Root mean squared error",rmse)
#
# df = pd.DataFrame({"predict":predict.flatten(),"y_true": y_true.flatten()})
# df.to_csv('result-%s.csv' % (saved_model),index=True, header=True)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
