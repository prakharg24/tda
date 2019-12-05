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
from obspy.signal.detrend import polynomial
import joblib
from scipy import signal


test_df = pd.read_csv('test_update_feature_df_200_800_2.csv')
train_df =pd.read_csv("training_data/train_6000_update_feature_df_200_800_2.csv")


class DataLoader():
    def __init__(self, X,y, batch_size, seq_length,input_size):
        self.batch_size = batch_size
        self.seq_length = seq_length

        X_shape = list(X.shape)
        X_shape[-1] = int(X_shape[-1]/input_size)

        step= int(X_shape[-1]/seq_length)
        lengh = step*seq_length

        X = X.reshape((X_shape[0],input_size,-1))[:,:,:lengh]
        self.X =  X.reshape((X_shape[0],seq_length,-1,input_size))


        self.y = y.reshape(y.shape[0], 1, 1)

    def dataset(self):
        return (self.X, self.y)



params = {
    "epochs": 300,
    "batch_size": 64,
    "seq_length": 20,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 500,
    "validation_split": 0.1,
    "input_size":3,
    "num_outputs":10
}


# In[5]:


X_test = test_df.drop(['period','powerSetPoint','sigma','delay'],axis=1)
y_test = test_df[['delay']]

X_train = train_df.drop(['period','powerSetPoint','sigma','delay'],axis=1)
y_train = train_df[['delay']]


scaler_X = RobustScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

scaler_filename = "old_scaler.save"
joblib.dump(scaler_X, scaler_filename)

data_loader = DataLoader(scaler_X.transform(X_test),scaler_y.transform(y_test), params["batch_size"], params["seq_length"],params["input_size"])
X_test, y_test = data_loader.dataset()

data_loader = DataLoader(scaler_X.transform(X_train),scaler_y.transform(y_train), params["batch_size"], params["seq_length"],params["input_size"])
X_train, y_train = data_loader.dataset()

print(X_train.shape)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))




from keras.callbacks import ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers

# In[13]:

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
    x = Dense(units=1,activation='sigmoid', name="classification_output")(x)
    # x = Flatten(name="classification_output")(x)

    return x

def build_regression_branch(inputs, hiddenCells):

    x = Dense(units=hiddenCells,activation='relu')(inputs)
    x = Dropout(params['dropout_keep_prob'])(x)
    x = Dense(units=1,activation='relu',kernel_regularizer=regularizers.l2(0.001), name="regression_output")(x)
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

    # classification_branch = build_classification_branch(sliced, layers[2])
    regression_branch = build_regression_branch(sliced, layers[2])

    #optimizer = Adam(clipvalue=0.5)
    print("Hey!!!")
    # model = Model(inputs=[inputs, weights_tensor], outputs=[classification_branch, regression_branch])
    # model = Model(inputs=inputs, outputs=classification_branch)
    model = Model(inputs=inputs, outputs=regression_branch)

    # cl4 = partial(custom_loss_4, weights=weights_tensor)
    # losses = {"classification_output": "binary_crossentropy", "regression_output": cl4}
    # lossWeights = {"classification_output": 1.0, "regression_output": 1.5}

    # sess = tf.keras.backend.get_session()
    # tf.contrib.quantize.create_training_graph(sess.graph)
    # sess.run(tf.global_variables_initializer())

    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    # model.compile(loss=losses, loss_weights=lossWeights, optimizer=adam)
    # model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['acc'])
    model.compile(loss='mean_squared_error', optimizer=adam)

    return model


# In[14]:


lstm_layer = [X_train[0].shape[0], X_train[0].shape[1], params['hidden_unit'], params['num_outputs'], X_train[0].shape[2]]
model = rnn_walk_lstm(lstm_layer, params)

print(model.summary())
saved_model = "BI_LSTM_L2-3000"

from keras.models import load_model
# try:
# #    df_his = pd.read_csv("history_%s.csv" %(saved_model),index_col=0)
#     model = load_model("models/%s" % (saved_model))
# except:
#     print("re train")
#     df_his = None

df_his=None

# Train RNN (LSTM) model with train set
# history = model.fit(X_train, y_train,
#           batch_size=params['batch_size'],
#           epochs=params['epochs'],
#           validation_split=params['validation_split'],
#           callbacks = [ModelCheckpoint(filepath="models/"+saved_model,monitor='loss',verbose=1, save_best_only=True),\
#                       ModelCheckpoint(filepath="models/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True)]
#           )
#
# # In[6]:
# if df_his is None:
#     df = pd.DataFrame(history.history)
#     df.to_csv("history_%s.csv" %(saved_model),header=True)
# else:
#     df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
#     df.to_csv("history_%s.csv" %(saved_model),header=True)

from keras.models import load_model
model = load_model("models/%s" % (saved_model) + "_val")


predict = model.predict(X_test)
predict = np.array([[ele[0][0]] for ele in predict])
y_test = np.array([[ele[0][0]] for ele in y_test])

predict =  scaler_y.inverse_transform(predict)
y_true  =  scaler_y.inverse_transform(y_test)


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


#from keras.utils.vis_utils import plot_model
#plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
