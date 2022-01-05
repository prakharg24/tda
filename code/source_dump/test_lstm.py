import numpy as np
from keras.layers.core import Dense, Activation, Dropout
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

from keras.constraints import Constraint
from keras.constraints import NonNeg

class CustomConstraint(Constraint):
    def __init__(self, perc_limit):
        self.perc_limit = perc_limit

    def __call__(self, w):
        norms = K.sqrt(K.square(w))
        norm_th = K.mean(norms)
        w_pos = K.relu(w, threshold=norm_th)
        w_neg = K.relu(tf.scalar_mul(-1, w), threshold=norm_th)

        w_pos = K.expand_dims(w_pos)
        w_neg = K.expand_dims(tf.scalar_mul(-1, w_neg))

        w_total = K.concatenate([w_pos, w_neg], axis=-1)
        w_total = K.sum(w_total, axis=-1)

        return w_total

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
    "epochs": 0,
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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))



from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional
from keras import regularizers


def rnn_app_lstm(layers, params):
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


def rnn_lstm(layers, params):
    """Build RNN (LSTM) model on top of Keras and Tensorflow"""

    model = Sequential()

    # model.add(Dense(units=layers[2],activation='relu'))
    # model.add(Dropout(params['dropout_keep_prob']))
    # model.add(Dense(units=layers[1],activation='relu'))
    # model.add(Dropout(params['dropout_keep_prob']))
    model.add(Bidirectional(LSTM(units=layers[2], return_sequences=True,activation='relu'),input_shape=(layers[0], layers[1])))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Bidirectional(LSTM(units=layers[2],return_sequences=True,activation='relu')))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Bidirectional(LSTM(units=layers[2], return_sequences=False,activation='relu')))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(units=layers[2],activation='relu'))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    #optimizer = Adam(clipvalue=0.5)
    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model

lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
# model = rnn_lstm(lstm_layer, params)

saved_model = "BI_LSTM_L2_heirar_pyramid-6000"

from keras.models import load_model, save_model
# model = load_model("models_pruned/%s" % (saved_model))
# model.save_weights("models_pruned/%s" % (saved_model) + ".h5")

# model.load_weights("models_old/%s" % (saved_model) + ".h5")

# print(model.summary())


# model.load_weights(mdl_fldr + "/%s" % (saved_model))
model = load_model("models_exp/%s" % (saved_model))
print(model.summary())

# history = model.fit(X_train, y_train,
#           batch_size=params['batch_size'],
#           epochs=2,
#           validation_split=params['validation_split'])

# if(mdl_fldr=='models_pruned'):
#     a = np.array(model.get_weights())
#     for ele in a:
#         med = np.percentile(ele, 50)
#         ele[ele < med] = 0
#     model.set_weights(a)

print("Predicting")

st = time.time()
predict =  scaler_y.inverse_transform(model.predict(X_test))

print("Time taken", time.time() - st)

y_true  =  scaler_y.inverse_transform(y_test)


predict = np.array([[round(x[0], 0)] for x in predict])

# print(predict)
print(np.shape(predict))
print(np.shape(y_true))

# exit()

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
