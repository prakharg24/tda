import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.recurrent import GRU,LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler


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
        self.X =  X.reshape((X_shape[0],seq_length,-1, 1))


        self.y = y

    def dataset(self):
        return (self.X, self.y)



params = {
    "epochs": 300,
    "batch_size": 64,
    "seq_length": 40,
    "dropout_keep_prob": 0.1,
    "hidden_unit": 500,
    "validation_split": 0.1,
    "input_size":3
}


X_test = test_df.drop(['period','powerSetPoint','sigma','delay'],axis=1)
y_test = test_df[['delay']]

X_train = train_df.drop(['period','powerSetPoint','sigma','delay'],axis=1)
y_train = train_df[['delay']]

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
from keras.layers import Bidirectional, Conv2D, MaxPooling2D, Flatten
from keras import regularizers


def cnn_lstm(layers, params):
    """Build CNN model on top of Keras and Tensorflow"""

    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=(layers[0], layers[1], 1)))
    model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    # model.add(Conv2D(filters=256, kernel_size=5, activation='relu'))
    # model.add(Dropout(params['dropout_keep_prob']))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    # model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(units=layers[2],activation='relu'))
    model.add(Dropout(params['dropout_keep_prob']))
    model.add(Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001)))
    #optimizer = Adam(clipvalue=0.5)
    adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
    model.compile(loss="mean_squared_error", optimizer=adam)

    return model


lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
model = cnn_lstm(lstm_layer, params)

print(model.summary())

saved_model = "CNN_L2-6000"
'''
from keras.models import load_model
try:
    df_his = pd.read_csv("history_%s.csv" %(saved_model),index_col=0)
    model = load_model("models_exp/%s" % (saved_model))
except:
    print("re train")
    df_his = None
'''
df_his=None

print("Start training")

# Train RNN (LSTM) model with train set
history = model.fit(X_train, y_train,
          batch_size=params['batch_size'],
          epochs=params['epochs'],
          validation_split=params['validation_split'],
          callbacks = [ModelCheckpoint(filepath="models_exp/"+saved_model,monitor='loss',verbose=1, save_best_only=True),\
                      ModelCheckpoint(filepath="models_exp/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True)]
          )

# print(model.summary())

# In[6]:
if df_his is None:
    df = pd.DataFrame(history.history)
    df.to_csv("history_%s.csv" %(saved_model),header=True)
else:
    df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
    df.to_csv("history_%s.csv" %(saved_model),header=True)

from keras.models import load_model
model = load_model("models_exp/%s" % (saved_model))

print("Predicting")
predict =  scaler_y.inverse_transform(model.predict(X_test))

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


from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
