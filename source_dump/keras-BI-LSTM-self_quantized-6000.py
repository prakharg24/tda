import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Concatenate
from keras.models import Sequential, Model
from keras.optimizers import Adam, Optimizer
from keras.legacy import interfaces
from keras.layers.recurrent import LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
import time
from sklearn.cluster import KMeans

from keras.constraints import Constraint
from keras.constraints import NonNeg

import tensorflow as tf
import pickle

pruningArr = [40., 60., 70., 80., 60., 30.]

class PruningConstraint(Constraint):
    def __init__(self, norm_ind):
        self.norm_ind = norm_ind

    def __call__(self, w):
        global pruningArr
        # norms = K.sqrt(K.square(w))
        # norm_th = K.mean(norms)
        print("Something happening")

        norm_th = tf.contrib.distributions.percentile(K.abs(w), pruningArr[self.norm_ind])

        w_pos = K.relu(w - norm_th)
        w_neg = K.relu(tf.scalar_mul(-1, w + norm_th))

        w_total = tf.add(w_pos, tf.scalar_mul(-1, w_neg))
        w_rem = tf.subtract(w, w_total)

        orig_shape = K.shape(w_rem)
        selector = tf.random.uniform(shape=orig_shape, minval=0, maxval=2, dtype='int32')

        selector = K.cast(selector, dtype='float32')
        w_sample = tf.multiply(w_rem, selector)

        w_total = tf.add(w_total, w_sample)

        # nz = tf.count_nonzero(w_total)

        # self.rate = self.rate*0.8

        return w_total

class CustomOptimizer(Optimizer):

    def __init__(self, new_cl,lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=None, decay=0., amsgrad=False, **kwargs):
        super(CustomOptimizer, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.new_cl = new_cl
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)

        print("Getting Update")
        new_grads = []

        new_cl = self.new_cl

        for i, (we, msk) in enumerate(zip(grads, new_cl)):
            if(i not in [6]):
                new_grads.append(we)
                continue
            # we = tf.Variable(we)
            we_var = tf.Variable(tf.zeros(tf.shape(we)))
            # new_grads.append(we)
            # continue
            print("Layer :", we, "with", len(msk), "clusters")
            for j, cls in enumerate(msk):
                if(j%1000==0):
                    print(j)
                # cls = np.transpose(cls)
                if(len(cls[0])==1):
                    print("Hey")
                    break
                else:
                    # cls = K.variable(cls, dtype='int64')
                    temp_we = tf.gather_nd(we, K.variable(np.array(cls), dtype='int64'))
                    temp_sum = tf.reduce_sum(temp_we)
                    temp_sum = tf.expand_dims(temp_sum, axis=0)
                    temp_sum = tf.tile(temp_sum, tf.shape(temp_we))
                    # for ele in cls:
                        # dim_ele = tf.expand_dims(K.variable(ele, dtype='int64'), axis=0)
                    we_var = tf.scatter_nd_update(we_var, K.variable(np.array(cls), dtype='int64'), temp_sum)

            new_grads.append(we_var)


        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, new_grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(CustomOptimizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def fork (model, n=2):
    forks = []
    for i in range(n):
        f = Sequential()
        f.add (model)
        forks.append(f)
    return forks


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
    "epochs": 10,
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
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1]))

    left1 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(inputs)
    right1 = LSTM(units=layers[2],  return_sequences=True, activation='relu', go_backwards=True)(inputs)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2],  return_sequences=True, activation='relu')(concat1)
    right2 = LSTM(units=layers[2],  return_sequences=True, activation='relu', go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    left3 = LSTM(units=layers[2],  return_sequences=False, activation='relu')(concat2)
    right3 = LSTM(units=layers[2],  return_sequences=False, activation='relu', go_backwards=True)(concat2)

    concat3 = Concatenate()([left3, right3])

    x = Dense(units=layers[2],activation='relu')(concat3)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001))(x)
    #optimizer = Adam(clipvalue=0.5)

    model = Model(inputs=inputs, outputs=x)
    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

def rnn_app_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    inputs = Input(shape=(layers[0], layers[1], layers[4]))

    low1 = TimeDistributed(LSTM(units=256, kernel_constraint=PruningConstraint(0), bias_constraint=PruningConstraint(0), recurrent_constraint=PruningConstraint(0), return_sequences=True, activation='relu'))(inputs)
    low1 = Dropout(params['dropout_keep_prob'])(low1)

    low2 = TimeDistributed(LSTM(units=128, kernel_constraint=PruningConstraint(1), bias_constraint=PruningConstraint(1), recurrent_constraint=PruningConstraint(1), return_sequences=False, activation='relu'))(low1)
    low2 = Dropout(params['dropout_keep_prob'])(low2)

    left1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(2), bias_constraint=PruningConstraint(2), recurrent_constraint=PruningConstraint(2),  return_sequences=True, activation='relu')(low2)
    right1 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(2), bias_constraint=PruningConstraint(2), recurrent_constraint=PruningConstraint(2),  return_sequences=True, activation='relu', go_backwards=True)(low2)

    concat1 = Concatenate()([left1, right1])

    left2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(3), bias_constraint=PruningConstraint(3), recurrent_constraint=PruningConstraint(3),  return_sequences=False, activation='relu')(concat1)
    right2 = LSTM(units=layers[2], kernel_constraint=PruningConstraint(3), bias_constraint=PruningConstraint(3), recurrent_constraint=PruningConstraint(3),  return_sequences=False, activation='relu', go_backwards=True)(concat1)

    concat2 = Concatenate()([left2, right2])

    x = Dense(units=layers[2],activation='relu', kernel_constraint=PruningConstraint(4), bias_constraint=PruningConstraint(4))(concat2)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001), kernel_constraint=PruningConstraint(5), bias_constraint=PruningConstraint(5))(x)
    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    model = Model(inputs=inputs, outputs=x)
    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

last_loss = 10

def get_zeroes(inp_model, loss):
    global last_loss
    # print(inp_model.get_weights()[:3])
    # return 0

    print("Hello")
    print(loss)
    ttl_sum = 0
    ttl_total = 0
    for ele in model.layers:
        curr_sum = 0
        curr_total = 0
        for weight in ele.get_weights():
            curr_arr = np.array(weight)
            curr_total += curr_arr.size
            curr_sum += curr_arr.size - np.count_nonzero(curr_arr)

        if(curr_total!=0):
            ttl_sum += curr_sum
            ttl_total += curr_total
            print(curr_sum, curr_total)

    print("Total :", ttl_total, "\nPruned :", ttl_sum, "\nLeft :", (ttl_total - ttl_sum))

    return 0

def get_unique(inp_model, loss):
    global last_loss
    global isImproving
    # print(inp_model.get_weights())
    # return 0

    print("Hello")
    ttl_sum = 0
    ttl_total = 0
    for ele in model.layers:
        curr_sum = 0
        curr_total = 0
        for weight in ele.get_weights():
            curr_arr = np.array(weight)
            curr_total += np.count_nonzero(curr_arr)
            curr_sum += np.unique(curr_arr).size

        if(curr_total!=0):
            ttl_sum += curr_sum
            ttl_total += curr_total
            print(curr_sum, curr_total)

    print("Total :", ttl_total, "\nAfter Quantization :", ttl_sum)

    return 0

def get_both(inp_model, loss):
    get_zeroes(inp_model, loss)
    get_unique(inp_model, loss)


lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1, X_train.shape[3]]
model = rnn_app_lstm(lstm_layer, params)

saved_model = "BI_LSTM_L2_dropPruning_quantize-6000"

from keras.models import load_model, save_model

model.load_weights("models_pruned/BI_LSTM_L2_dropPruning_new_fine-6000_val")
# model.load_weights("models_pruned/BI_LSTM_L2_New_arr-6000.h5")

# print(model.summary())
# get_both(model, 10)

np.save("early_weights", model.get_weights())
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

loss = 10
val_loss = 10

from keras.models import Model
print("Setting weights")

file = open("final_clusters.pkl",'rb')
new_cl = pickle.load(file)
file.close()

# new_we = np.load("final_weights.npy", allow_pickle=True)

# old_we = model.get_weights()
# print(np.shape(old_we))
#
# new_we = []
#
# for i, (we, msk) in enumerate(zip(old_we, new_cl)):
#     # if(i!=18):
#     #     new_we.append(we)
#     #     continue
#     print("Layer :", np.shape(we), "with", len(msk), "clusters")
#     for cls in msk:
#         if(len(np.shape(we))==1):
#             eles = we[(np.array(cls))]
#         else:
#             cls = np.transpose(cls)
#             eles = we[(cls[0], cls[1])]
#
#         centroid = np.sum(eles)/np.size(eles)
#
#         if(len(np.shape(we))==1):
#             we[(np.array(cls))] = centroid
#         else:
#             we[(cls[0], cls[1])] = centroid
#     # exit()
#     # continue
#     print(np.unique(we).size)
#     print(len(msk))
#
#     new_we.append(we)
#
#     np.save("final_weights", np.array(new_we))
#
# print(np.shape(new_we))
# for ele in new_we:
#     print(np.shape(ele))

new_we = np.load("final_weights.npy", allow_pickle=True)

model.set_weights(new_we)

newopt = CustomOptimizer(new_cl, clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
model.compile(loss="mean_squared_error", optimizer=newopt)

print("Start training")

print(model.summary())
get_both(model, 10)

# exit()

history = model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          validation_split=params['validation_split'],
          callbacks = [ModelCheckpoint(filepath="models_pruned/"+saved_model,monitor='loss',verbose=1, save_best_only=True,save_weights_only=True),\
                      ModelCheckpoint(filepath="models_pruned/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True,save_weights_only=True),\
                      LambdaCallback(on_epoch_end=lambda batch, logs: get_both(model, logs['val_loss']))]
          )
#
# print(model.summary())

# In[6]:
if df_his is None:
    df = pd.DataFrame(history.history)
    df.to_csv("history_%s.csv" %(saved_model),header=True)
else:
    df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
    df.to_csv("history_%s.csv" %(saved_model),header=True)


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
