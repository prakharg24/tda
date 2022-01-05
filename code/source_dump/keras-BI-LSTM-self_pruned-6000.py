import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Concatenate, LeakyReLU
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
from matplotlib import pyplot as plt

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf
#
# K.set_floatx('float16')
# K.set_epsilon(1e-4)

# pruningArr = [40.,40.,40.,60.,60.,60.,70.,70.,70.,70.,70.,70.,80.,80.,80.,80.,80.,80.,60.,60.,30.,30.]
pruningArr = [36.1, 37.183, 36.1, 62.77469112334501, 55.774499999999996, 54.15, 69.033028225, 65.07025, 63.175, 73.23713964390252, 80.02819989166267, 63.175, 91.4607998761859, 91.4607998761859, 72.2, 91.4607998761859, 91.4607998761859, 72.2, 66.59766981275672, 54.15, 27.075, 27.075]

class PruningConstraint(Constraint):
    def __init__(self, norm_ind):
        self.norm_ind = norm_ind

    def __call__(self, w):
        global pruningArr
        # norms = K.sqrt(K.square(w))
        # norm_th = K.mean(norms)
        print("Something happening")

        norm_th = tf.contrib.distributions.percentile(K.abs(w), pruningArr[self.norm_ind])

        w_pos = K.relu(w - norm_th) + norm_th
        w_neg = K.relu(tf.scalar_mul(-1, w + norm_th)) + norm_th


        w_total = tf.add(w_pos, tf.scalar_mul(-1, w_neg))
        # w_rem = tf.subtract(w, w_total)
        #
        # orig_shape = K.shape(w_rem)
        # selector = tf.random.uniform(shape=orig_shape, minval=0, maxval=2, dtype='int32')
        #
        # # selector = K.cast(selector, dtype='float16')
        # selector = K.cast(selector, dtype='float32')
        # w_sample = tf.multiply(w_rem, selector)
        #
        # w_total = tf.add(w_total, w_sample)

        # w_total = tf.cast(w_total, tf.float16)
        # w_total = tf.round(1e4*w_total)/1e4
        # term = 2**(-11)
        # w_total = tf.round(w_total/term)*term
        # w_total = tf.round(w_total)
        # print("Something else")
        # print(w_total.dtype)
        # w_total = tf.to_bfloat16(w_total)

        # nz = tf.count_nonzero(w_total)

        # self.rate = self.rate*0.8

        return w_total


def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


def my_gradients(model, X, y):
    """Return the gradient of every trainable weight in model

    Parameters
    -----------
    model : a keras model instance

    First, find all tensors which are trainable in the model. Surprisingly,
    `model.trainable_weights` will return tensors for which
    trainable=False has been set on their layer (last time I checked), hence the extra check.
    Next, get the gradients of the loss with respect to the weights.

    """
    loss_layer = model.total_loss
    listOfVariableTensors = model.trainable_weights
    gradients = K.gradients(loss_layer, listOfVariableTensors)

    sess = K.get_session()
    all_op = sess.graph.get_operations()

    inp, dns, dnssw = 'input_1', 'dense_2_target', 'dense_2_sample_weights'
    for ele in all_op:
        if('input_' in ele.name):
            print(ele.name)
            inp = ele.name
        if('dense_' in  ele.name and '_target' in ele.name):
            print(ele.name)
            dns = ele.name
        if('dense_' in ele.name and '_sample_weights' in ele.name):
            print(ele.name)
            dnssw = ele.name
    evaluated_gradients = sess.run(gradients,feed_dict={inp + ":0":X, dns + ":0":y, dnssw + ":0":np.ones(np.shape(y)[0])})
    # evaluated_gradients = sess.run(gradients,feed_dict=[X, y, np.ones(np.shape(y)[0])])

    # print(evaluated_gradients)
    return evaluated_gradients


test_df = pd.read_csv('test_update_feature_df_200_800_2.csv')
train_df = pd.read_csv('training_data/train_6000_update_feature_df_200_800_2.csv')


class DataLoader():
    def __init__(self, X,y, batch_size, seq_length,input_size):
        self.batch_size = batch_size
        self.seq_length = seq_length

        X_shape = list(X.shape)
        X_shape[-1] = int(X_shape[-1]/input_size)

        step= int(X_shape[-1]/seq_length)
        lengh = step*seq_length

        X = X.reshape((X_shape[0],input_size,-1))[:,:,:lengh]
        # self.X =  X.reshape((X_shape[0],seq_length,-1))
        X =  X.reshape((X_shape[0],input_size,seq_length,-1))
        self.X = X.transpose((0, 2, 3, 1))


        self.y = y

    def dataset(self):
        return (self.X, self.y)



params = {
    "epochs": 300,
    "mini_ep": 15,
    "batch_size": 64,
    "seq_length": 20,
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

    drop_terms.extend(['period','powerSetPoint','sigma','delay'])
    X_out = df_inp.drop(drop_terms,axis=1)

    return X_out, y_out

X_test, y_test = preprocess(test_df)
X_train, y_train = preprocess(train_df)

print("Data Loaded")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

data_loader = DataLoader(scaler_X.transform(X_test),scaler_y.transform(y_test), params["batch_size"], params["seq_length"],params["input_size"])
X_test, y_test = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train),scaler_y.transform(y_train), params["batch_size"], params["seq_length"],params["input_size"])
X_train, y_train = data_loader.dataset()

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1]))

    left1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(0), bias_constraint=PruningConstraint(1), recurrent_constraint=PruningConstraint(2), return_sequences=True, activation='relu')(inputs)
    right1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(3), bias_constraint=PruningConstraint(4), recurrent_constraint=PruningConstraint(5), return_sequences=True, activation='relu', go_backwards=True)(inputs)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(6), bias_constraint=PruningConstraint(7), recurrent_constraint=PruningConstraint(8),  return_sequences=True, activation='relu')(concat1)
    right2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(9), bias_constraint=PruningConstraint(10), recurrent_constraint=PruningConstraint(11),  return_sequences=True, activation='relu', go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    left3 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(12), bias_constraint=PruningConstraint(13), recurrent_constraint=PruningConstraint(14),  return_sequences=False, activation='relu')(concat2)
    right3 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(15), bias_constraint=PruningConstraint(16), recurrent_constraint=PruningConstraint(17),  return_sequences=False, activation='relu', go_backwards=True)(concat2)

    concat3 = Concatenate()([left3, right3])

    x = Dense(units=layers[2],activation='relu', kernel_constraint=PruningConstraint(18), bias_constraint=PruningConstraint(19))(concat3)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001), kernel_constraint=PruningConstraint(20), bias_constraint=PruningConstraint(21))(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model


def rnn_leaky_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))

    low1 = TimeDistributed(LSTM(units=256, kernel_constraint=PruningConstraint(0), bias_constraint=PruningConstraint(1), recurrent_constraint=PruningConstraint(2), return_sequences=True, activation=LeakyReLU(alpha=0.3)))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=128, kernel_constraint=PruningConstraint(3), bias_constraint=PruningConstraint(4), recurrent_constraint=PruningConstraint(5), return_sequences=False, activation=LeakyReLU(alpha=0.3)))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    left1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(6), bias_constraint=PruningConstraint(7), recurrent_constraint=PruningConstraint(8),  return_sequences=True, activation=LeakyReLU(alpha=0.3))(low2)
    right1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(9), bias_constraint=PruningConstraint(10), recurrent_constraint=PruningConstraint(11),  return_sequences=True, activation=LeakyReLU(alpha=0.3), go_backwards=True)(low2)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(12), bias_constraint=PruningConstraint(13), recurrent_constraint=PruningConstraint(14),  return_sequences=False, activation=LeakyReLU(alpha=0.3))(concat1)
    right2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(15), bias_constraint=PruningConstraint(16), recurrent_constraint=PruningConstraint(17),  return_sequences=False, activation=LeakyReLU(alpha=0.3), go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    x = Dense(units=layers[2],activation=LeakyReLU(alpha=0.3), kernel_constraint=PruningConstraint(18), bias_constraint=PruningConstraint(19))(concat2)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation=LeakyReLU(alpha=0.3),kernel_regularizer=regularizers.l2(0.001), kernel_constraint=PruningConstraint(20), bias_constraint=PruningConstraint(21))(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

def rnn_app_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))

    low1 = TimeDistributed(LSTM(units=256, kernel_constraint=PruningConstraint(0), bias_constraint=PruningConstraint(1), recurrent_constraint=PruningConstraint(2), return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=128, kernel_constraint=PruningConstraint(3), bias_constraint=PruningConstraint(4), recurrent_constraint=PruningConstraint(5), return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    left1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(6), bias_constraint=PruningConstraint(7), recurrent_constraint=PruningConstraint(8),  return_sequences=True, activation='relu')(low2)
    right1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(9), bias_constraint=PruningConstraint(10), recurrent_constraint=PruningConstraint(11),  return_sequences=True, activation='relu', go_backwards=True)(low2)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(12), bias_constraint=PruningConstraint(13), recurrent_constraint=PruningConstraint(14),  return_sequences=False, activation='relu')(concat1)
    right2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(15), bias_constraint=PruningConstraint(16), recurrent_constraint=PruningConstraint(17),  return_sequences=False, activation='relu', go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    x = Dense(units=layers[2],activation='relu', kernel_constraint=PruningConstraint(18), bias_constraint=PruningConstraint(19))(concat2)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001), kernel_constraint=PruningConstraint(20), bias_constraint=PruningConstraint(21))(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model


last_loss = 10

def get_zeroes(inp_model, loss):
    global last_loss
    print(inp_model.get_weights()[6])
    # return 0

    print("Hello")

    print(loss)
    ttl_sum = 0
    ttl_total = 0
    for ele in model.layers:
        curr_sum = 0
        curr_total = 0
        for weight in ele.get_weights():
            tmp_we = weight[np.nonzero(weight)]
            if(len(tmp_we)==0):
                continue
            # print(np.amax(np.abs(tmp_we)))
            # print(np.amin(np.abs(tmp_we)))
            curr_arr = np.array(weight)
            curr_total += curr_arr.size
            curr_sum += curr_arr.size - np.count_nonzero(curr_arr)

        if(curr_total!=0):
            ttl_sum += curr_sum
            ttl_total += curr_total
            print(curr_sum, curr_total)

    print("Total :", ttl_total, "\nPruned :", ttl_sum, "\nLeft :", (ttl_total - ttl_sum))

    return 0

lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
model = rnn_app_lstm(lstm_layer, params)
# model = rnn_leaky_lstm(lstm_layer, params)

saved_model = "BI_LSTM_L2_dropPruning_grad_exp-6000"

from keras.models import load_model, save_model
#
# model = rnn_app_lstm(lstm_layer, params)
model.load_weights("models_pruned/BI_LSTM_L2_dropPruning_grad-6000_val")

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

get_zeroes(model, 10)
#
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

for ele in model.layers:
    ele._kernel_constraint = PruningConstraint(12)

adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
model.compile(loss="mean_squared_error", optimizer=adam)

df_his=None

print("Start training")
loss = 10
val_loss = 10

from keras.models import Model

# for i in range(params['epochs']//(2*params['mini_ep'])):
#     print("VALUE OF I", i)
#     print(pruningArr)
    # print(model.get_weights())
history = model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          # epochs=params['mini_ep'],
          epochs=1,
          validation_split=params['validation_split'],
          callbacks = [ModelCheckpoint(filepath="models_pruned/"+saved_model,monitor='loss',verbose=1, save_best_only=True, save_weights_only=True),\
                      ModelCheckpoint(filepath="models_pruned/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True, save_weights_only=True),\
                      LambdaCallback(on_epoch_end=lambda batch, logs: get_zeroes(model, logs['val_loss']))]
          )

exit()
    # grads = my_gradients(model, X_train[-1*params['batch_size']:], y_train[-1*params['batch_size']:])

    # sumt = 0
    # numt = 0
    # for ele in grads:
    #     sumt += np.sum(np.abs(ele))
    #     numt += np.count_nonzero(ele)
    #
    # av = sumt/numt
    #
    # if(np.amin(history.history['val_loss'])<val_loss):
    #     val_loss = np.amin(history.history['val_loss'])
    #     new_arr = []
    #     for pe, ge in zip(pruningArr, grads):
    #         curr_avg = np.average(np.abs(ge[np.nonzero(ge)]))
    #         if(curr_avg>av):
    #             new_arr.append(pe)
    #         else:
    #             new_arr.append(pe*1.03)
    #     pruningArr = [ele for ele in new_arr]
    # else:
    #     pruningArr = [ele*0.95 for ele in pruningArr]
    #
    # # exit()
    #
    # adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(loss="mean_squared_error", optimizer=adam)

#
# print(pruningArr)
# history = model.fit(X_train, y_train,
#           batch_size=params['batch_size'],
#           # epochs=params['epochs']//2,
#           epochs=params['epochs']//8,
#           validation_split=params['validation_split'],
#           callbacks = [ModelCheckpoint(filepath="models_pruned/"+saved_model,monitor='loss',verbose=1, save_best_only=True, save_weights_only=True),\
#                       ModelCheckpoint(filepath="models_pruned/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True, save_weights_only=True),\
#                       LambdaCallback(on_epoch_end=lambda batch, logs: get_zeroes(model, logs['val_loss']))]
#           )
#
# model.save_weights("models_pruned/%s" % (saved_model) + ".h5")
# print(model.summary())
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

#
# def get_redundancy_fc(inp_weights, old_red):
#     w = inp_weights[0]
#     b = inp_weights[1]
#
#     print(w.shape)
#     print(b.shape)
#
#     for i in range(len(w[0])):
#         if (i in old_red):
#             w[:, i] = w[:, i]*0
#             b[i] = b[i]*0
#
#     ans = []
#     for i in range(len(w)):
#         if(np.count_nonzero(w[i])==0):
#             ans.append(i)
#
#     return ans, inp_weights
#
# def get_redundancy_lstm(inp_weights, old_red):
#     w = inp_weights[0]
#     rw = inp_weights[1]
#     b = inp_weights[2]
#
#     print(w.shape)
#     print(rw.shape)
#     print(b.shape)
#
#     units = len(w[0])//4
#
#     for i in range(units):
#         if (i in old_red):
#             w[:, 3*units + i] = w[:, 3*units + i]*0
#             rw[:, 3*units + i] = rw[:, 3*units + i]*0
#             b[3*units + i] = b[3*units + i]*0
#
#     ans = []
#     for i in range(len(w)):
#         if(np.count_nonzero(w[i])==0):
#             ans.append(i)
#
#     return ans, inp_weights
#
# layers = model.layers
# #
# print(layers)
# #
# # exit()
#
# print("Started")
#
# redun = []
# redun, weg = get_redundancy_fc(layers[-1].get_weights(), redun)
# layers[-1].set_weights(weg)
# redun, weg = get_redundancy_fc(layers[-3].get_weights(), redun)
# layers[-3].set_weights(weg)

# redun1 = [ele for ele in redun if (ele<500)]
# redun2 = [(ele-500) for ele in redun if (ele>=500)]
# redun1, weg = get_redundancy_lstm(layers[-5].get_weights(), redun1)
# layers[-5].set_weights(weg)
# redun2, weg = get_redundancy_lstm(layers[-6].get_weights(), redun2)
# layers[-6].set_weights(weg)
# redun = [ele for ele in redun2 if (ele in redun1)]
# redun1 = [ele for ele in redun if (ele<500)]
# redun2 = [(ele-500) for ele in redun if (ele>=500)]
# redun1, weg = get_redundancy_lstm(layers[-8].get_weights(), redun1)
# layers[-8].set_weights(weg)
# redun2, weg = get_redundancy_lstm(layers[-9].get_weights(), redun2)
# layers[-9].set_weights(weg)
# redun = [ele for ele in redun2 if (ele in redun1)]

# print(redun)

# weights = model.get_weights()
# for ele in weights:
#     if(len(ele.shape)==2):
#         count_r = 0
#         count_c = 0
#         for i in range(len(ele)):
#             if(np.count_nonzero(ele[i, :])==0):
#                 count_c += 1
#         for i in range(len(ele[0])):
#             if(np.count_nonzero(ele[:, i])==0):
#                 count_r += 1
#         print(ele.shape, count_c, count_r)
#         # print(ele)
#
# # exit()

# exit()
#
# for repi in range(5):

# grads = my_gradients(model, X_train[-1*params['batch_size']:], y_train[-1*params['batch_size']:])
grads = my_gradients(model, X_train[-1:], y_train[-1:])

# for i in range(10):
#     curr_grads = my_gradients(model, X_train[i*params['batch_size']:(i+1)*params['batch_size']], y_train[i*params['batch_size']:(i+1)*params['batch_size']])
#     grads = [e1 + e2 for e1, e2 in zip(grads, curr_grads)]

weights = model.get_weights()

# file = open(str(repi), "w")

new_weights = []
for ite, (e1, e2) in enumerate(zip(weights, grads)):
    old_shp = e1.shape
    e1f = e1.flatten()
    e2f = e2.flatten()

    count = 0
    for i in range(len(e1f)):
        if(e1f[i]!=0 and e2f[i]==0):
            if(ite!=7):
                count += 1
                e1f[i]=0
            else:
                # file.write(str(e1f[i]) + "\n")
                # print(e1f[i], e2f[i])
                e1f[i]=0
                count += 1

    print(old_shp)
    e1f = np.reshape(e1f, old_shp)
    print(e1f.shape)
    new_weights.append(e1f)

    print(count)

    # model = rnn_app_lstm(lstm_layer, params)
    #
    # model.set_weights(new_weights)
    # history = model.fit(X_train, y_train,
    #           batch_size=params['batch_size'],
    #           epochs=1,
    #           # epochs=params['epochs']//8,
    #           validation_split=params['validation_split'],
    #           callbacks = [ModelCheckpoint(filepath="models_pruned/"+saved_model,monitor='loss',verbose=1, save_best_only=True, save_weights_only=True),\
    #                       ModelCheckpoint(filepath="models_pruned/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True, save_weights_only=True),\
    #                       LambdaCallback(on_epoch_end=lambda batch, logs: get_zeroes(model, logs['val_loss']))]
    #           )
    #
    # model = rnn_leaky_lstm(lstm_layer, params)
    # model.set_weights(new_weights)


model = rnn_app_lstm(lstm_layer, params)

model.set_weights(new_weights)
# exit()
# all_w = model.get_weights()
#
# for ele in all_w:
#     fl_arr = ele.flatten()
#     print(len(fl_arr))
#     # plt.hist(fl_arr, normed=True, bins=200)
#     # plt.show()
#     # plt.clf()
#     print(np.amin(np.abs(fl_arr[np.nonzero(fl_arr)])))
#     # print(fl_arr)

# all_l = model.layers
#
# for ele in model.layers:
#     print(len(ele.get_weights()))
#
# exit()

get_zeroes(model, 10)

# model.save_weights("models_pruned/BI_LSTM_L2_dropPruning_grad-6000")

print("Predicting")
st = time.time()
predict =  scaler_y.inverse_transform(model.predict(X_test))

print("Time taken", time.time() - st)

y_true  =  scaler_y.inverse_transform(y_test)


predict = np.array([[round(x[0], 0)] for x in predict])


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


# In[13]:


nrmsd = NRMSD(y_true, predict)
mape  = MAPE(y_true, predict)
mae   = mean_absolute_error(y_true, predict)
rmse   = np.sqrt(mean_squared_error(y_true, predict))
print ("NRMSD",nrmsd)
print ("MAPE",mape)
print ("neg_mean_absolute_error",mae)
print ("Root mean squared error",rmse)

df = pd.DataFrame({"predict":predict.flatten(),"y_true": y_true.flatten()})
df.to_csv('result-%s.csv' % (saved_model),index=True, header=True)


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
