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
import random
from functools import partial

random.seed(24)

import matplotlib.pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf


def custom_loss_4(y_true, y_pred, weights):
    return K.mean(K.square(y_true - y_pred) * weights)

def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


test_df = pd.read_csv('test_data_200_1500.csv')
train_df = pd.read_csv('train_data_200_1500.csv')

# print(train_df)

class DataLoader():
    def __init__(self, X,y, batch_size, step,input_size):
        self.batch_size = batch_size
        self.step = step

        # print(y)

        X_shape = list(X.shape)
        print(X_shape)
        X_shape[-1] = int(X_shape[-1]/input_size)

        seq_length = int(X_shape[-1]/step)
        lengh = step*seq_length

        # print(step, seq_length)

        X = X.reshape((X_shape[0],input_size,-1))[:,:,:lengh]

        new_X = []
        new_y = []
        new_di = []
        new_we = []
        for eX, ey in zip(X, y):
            for i in range(5):
                st_ind = random.randint(0, seq_length - 30)
                new_X.append(eX[:,st_ind*15:st_ind*15+450])

                delay_st = (400 - st_ind*15)//15 + 1
                delay_st = max(0, delay_st - 20)

                new_di.append(st_ind)

                # print(delay_st)

                y_temp = []
                we_temp = []
                onh = np.zeros((50))
                for di in range(delay_st):
                    # onhy = np.copy(onh)
                    # onhy[0] = 1
                    # y_temp.append(onhy)
                    y_temp.append(0.)
                    we_temp.append(5)
                for di in range(delay_st, 11):
                    # onhy = np.copy(onh)
                    # onhy[int(ey[0])] = 1
                    # y_temp.append(onhy)
                    y_temp.append(ey[0])
                    ite_ind = di - delay_st
                    we_temp.append(st_ind + ite_ind + 1)

                # print(y_temp)
                # print(we_temp)

                we_temp = we_temp/np.sum(we_temp)

                # print(we_temp)

                new_we.append(we_temp)
                new_y.append(y_temp)
                # print("Done")

        print(np.shape(new_X))
        print(np.shape(new_y))

        X = np.array(new_X)
        y = np.array(new_y)

        # print(y)
        # exit()

        X = X.reshape((X.shape[0],input_size,30,-1))
        self.X = X.transpose((0, 2, 3, 1))

        self.y = y
        self.we = np.array(new_we)

        self.di = np.array(new_di)

        print(np.shape(self.X))
        print(np.shape(self.y))
        print(np.shape(self.di))
        print(np.shape(self.we))

        # exit()


        # print(self.y)
        print("Wait")
        # print(self.di)

        # exit()
        # print(self.y)

        # self.y = y


    def dataset(self):
        return (self.X, self.y, self.di, self.we)



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

def preprocess(df_inp):
    y_out = df_inp[['delay']]

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

    drop_terms.extend(['delay', 'Unnamed: 0'])
    X_out = df_inp.drop(drop_terms,axis=1)

    return X_out, y_out

X_test, y_test = preprocess(test_df)
X_train, y_train = preprocess(train_df)

print("Data Loaded")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

data_loader = DataLoader(scaler_X.transform(X_test),scaler_y.transform(y_test), params["batch_size"], params["step"],params["input_size"])
X_test, y_test, di_test, weights_test = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train),scaler_y.transform(y_train), params["batch_size"], params["step"],params["input_size"])
X_train, y_train, di_train, weights_train = data_loader.dataset()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers

def rnn_app_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))

    low1 = TimeDistributed(LSTM(units=256, return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=128, return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(low2)
    right1 = LSTM(units=layers[2], return_sequences=True, activation='relu', go_backwards=True)(low2)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2], return_sequences=False, activation='relu')(concat1)
    right2 = LSTM(units=layers[2], return_sequences=False, activation='relu', go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    x = Dense(units=layers[2],activation='relu')(concat2)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

def rnn_walk_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))
    # weights_tensor = Input(shape=(layers[3],))

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
    # model = Model(inputs=[inputs, weights_tensor], outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model

last_loss = 10

# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 11, X_train.shape[3]]
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
model = rnn_walk_lstm(lstm_layer, params)

saved_model = "LSTM_L2_walk_ahead"

from keras.models import load_model, save_model
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
loss = 10
val_loss = 10

from keras.models import Model
# #
# history = model.fit(X_train, y_train,
#           batch_size=params['batch_size'],
#           epochs=params['epochs'],
#           validation_split=params['validation_split'],
#           callbacks = [ModelCheckpoint(filepath="models_window/"+saved_model,monitor='loss',verbose=1, save_best_only=True, save_weights_only=True),\
#                       ModelCheckpoint(filepath="models_window/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True, save_weights_only=True)]
#           )
# #
# # # model.save_weights("models_pruned/%s" % (saved_model) + ".h5")
# # # print(model.summary())
# #
# # # In[6]:
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

# model.save("models_pruned/BI_LSTM_L2_dropPruning_new_fine-6000_model_val")

# model = tf.keras.models.load_model("models_pruned/BI_LSTM_L2_dropPruning_new-6000_model_val")

print("Predicting")
st = time.time()
# print(model.get_weights())

# new_we = []
# for ele in model.get_weights():
#     ele = np.round(ele/2**-5)
#     ele = ele*2**-5
#
#     new_we.append(ele)
#
# model.set_weights(new_we)

all_w = model.get_weights()

for ele in all_w:
    fl_arr = ele.flatten()
    print(len(fl_arr))
    plt.hist(fl_arr, normed=True, bins=200)
    plt.show()
    plt.clf()
    # print(fl_arr)

exit()

print("After")
# print(model.get_weights())

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

predict =  scaler_y.inverse_transform(model.predict(X_test))

print("Time taken", time.time() - st)

y_true  =  scaler_y.inverse_transform(y_test)

# predict = np.array([[round(x[0], 0)] for x in predict])

# predict_new = []
# y_true_new = []
#
# for e1, e2 in zip(y_true, predict):
#     e2_new = np.array([np.argmax(ele) for ele in e2])
#     e1_new = np.array([np.argmax(ele) for ele in e1])
#
#     y_true_new.append(e1_new)
#     predict_new.append(e2_new)
#
# predict = np.copy(predict_new)
# y_true = np.copy(y_true_new)

# np.save("final_predict", np.array(predict))

# predict = np.load("final_predict.npy", allow_pickle=True)

print(np.shape(y_true))
print(np.shape(predict))

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error


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

# exit()
# In[13]:

print("Hey")
# for i in range(len(y_true)):
#     print(y_true[i], predict[i])
# print(y_true)
# print(predict)

# exit()

tmp_true = []
tmp_predict = []
for i in range(len(y_true)):
    delay_st = (400 - di_test[i]*15)//15 + 1
    delay_st = max(0, delay_st - 20)

    for j in range(delay_st):
        tmp_true.append([round(y_true[i,j], 0)])
        tmp_predict.append([round(predict[i,j], 0)])

count = 0
for e1, e2 in zip(tmp_true, tmp_predict):
    if(e1[0]!=e2[0]):
        pass
        # print(e1, e2)
    else:
        count += 1

print(count, len(tmp_true), count/len(tmp_true))

mae = mean_absolute_error(tmp_true, tmp_predict)
rmse = np.sqrt(mean_squared_error(tmp_true, tmp_predict))

print(mae, rmse)

ttl_true = []
ttl_predict = []

sctt_mae = []
sctt_rmse = []
sctt_j = []
for j in range(0, 17):
    # tmp_true = []
    # tmp_predict = []
    # for i in range(j, len(predict), 7):
    #     tmp_true.append(y_true[i])
    #     tmp_predict.append(predict[i])

    tmp_true = []
    tmp_predict = []
    for i in range(len(y_true)):
        new_ind = 27 + j - di_test[i]
        if(new_ind<=30 and new_ind>=20):
            tmp_true.append([round(y_true[i,new_ind-20], 0)])
            tmp_predict.append([round(predict[i,new_ind-20], 0)])


    if(j>0):
        ax = plt.subplot(4, 4, j)
        for e1, e2 in zip(tmp_true, tmp_predict):
            ax.plot(e1[0], e2[0], 'ro', color='red')

        ax.plot([0, 50], [0, 50], color='black')
        ax.title.set_text(str(1010 + j*30))

    ttl_true.extend(tmp_true)
    ttl_predict.extend(tmp_predict)

    mae = mean_absolute_error(tmp_true, tmp_predict)
    rmse = np.sqrt(mean_squared_error(tmp_true, tmp_predict))

    print(len(tmp_true))
    print(mae, rmse)

    sctt_j.append(str(1010 + j*30))
    sctt_mae.append(mae)
    sctt_rmse.append(rmse)


fig = plt.gcf()
fig.set_size_inches(12,12)
plt.savefig('plots2/complete_progress')
plt.clf()

ttl_true = np.array(ttl_true)
ttl_predict = np.array(ttl_predict)


plt.plot(sctt_j, sctt_mae, '-o', color='red')
for i, name in enumerate(sctt_mae):
    plt.annotate(round(name, 2), (sctt_j[i], sctt_mae[i]))
plt.plot(sctt_j, sctt_rmse, '-o', color='blue')
for i, name in enumerate(sctt_rmse):
    plt.annotate(round(name, 2), (sctt_j[i], sctt_rmse[i]))

plt.plot((0, 17), (2.240928882438316, 2.240928882438316), 'k-', color='red')
plt.plot((0, 17), (3.461796474822204, 3.461796474822204), 'k-', color='blue')
fig = plt.gcf()
fig.set_size_inches(11,8)
plt.savefig('Figure_1_running')
plt.clf()


# for e1, e2 in zip(y_true, predict):
#     print(e1[0], e2[0], e2[-1])

nrmsd = NRMSD(ttl_true, ttl_predict)
mape  = MAPE(ttl_true, ttl_predict)
mae   = mean_absolute_error(ttl_true, ttl_predict)
rmse   = np.sqrt(mean_squared_error(ttl_true, ttl_predict))
print ("NRMSD",nrmsd)
print ("MAPE",mape)
print ("neg_mean_absolute_error",mae)
print ("Root mean squared error",rmse)
#
#
# df = pd.DataFrame({"predict":predict.flatten(),"y_true": y_true.flatten()})
# df.to_csv('result-%s.csv' % (saved_model),index=True, header=True)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
