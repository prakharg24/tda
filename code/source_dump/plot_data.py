# from keras.layers.core import Dense, Activation, Dropout
# from keras.models import Sequential
# from keras.layers.recurrent import GRU,LSTM
import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# mymodel = np.array(pd.read_csv('result-BI_LSTM_L2_retry-6000.csv'))
# earlier = np.array(pd.read_csv('result-BI_LSTM_L2-6000.csv'))


# for ele in mymodel:
#     plt.plot(float(ele[2]), float(ele[1]), 'ro', color='red')
#
# plt.plot([0, 50], [0, 50], color='black')
# plt.show()
# plt.clf()

# for ele in earlier:
#     plt.plot(float(ele[2]), float(ele[1]), 'ro', color='blue')
#
# plt.plot([0, 50], [0, 50], color='black')
# plt.show()
#
# exit()

test_df = pd.read_csv('test_data_200_1500_random.csv')
train_df = pd.read_csv('train_data_200_1500_random.csv')


class DataLoader():
    def __init__(self, X,y, y_out, batch_size, step,input_size,num_outputs, output_st, isTrain):
        self.batch_size = batch_size
        self.step = step

        # print(y)

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
            if(isTrain):
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
    "input_size":3,
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

    return X_out, y_out, y_out_st

X_test, y_test, y_test_out = preprocess(test_df)
X_train, y_train, y_train_out = preprocess(train_df)

print("Data Loaded")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

scaler_X.fit(np.concatenate([X_test,X_train],axis=0))
scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

# data_loader = DataLoader(scaler_X.transform(X_test),scaler_y.transform(y_test), np.array(y_test_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], False)
# X_test, y_test_classification, y_test_regression = data_loader.dataset()
#
# data_loader = DataLoader(scaler_X.transform(X_train),scaler_y.transform(y_train), np.array(y_train_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], False)
# X_train, y_train_classification, y_train_regression = data_loader.dataset()

# X_test = scaler_X.transform(X_test)
X_test = np.array(X_test)
y_test = np.array(y_test)
y_test_out = np.array(y_test_out)


X_running = []
for ele, ey, eyo in zip(X_test, y_test, y_test_out):
    if(eyo[0]>800 and ey[0]==0):
        X_running.append(ele)

X_len = len(X_running)
X_sum = np.sum(X_running, axis=0)
X_sum = X_sum/X_len

print(np.shape(X_test))
print(np.shape(X_sum))
X_test = X_test - X_sum

print(np.shape(X_test))

# exit()

print(len(X_test))
for ele, ey, eyo in zip(X_test, y_test, y_test_out):
    if(eyo[0]<800):
        continue
    if(ey[0]!=0 and ey[0]!=25 and ey[0]!=50):
        continue
    signal1 = ele[:len(ele)//3]
    y = [val for val in signal1]
    x = list(range(200, 2*len(signal1)+200, 2))
    plt.plot(x, y)
    plt.title("Delay start : " + str(eyo[0]) + " -- Delay Value : " + str(ey[0]))
    plt.show()
    # for i in range(len(ele)):
    #     print(ele[i])

exit()


arr = []
for i in range(100, 900, 2):
    arr.append(i)

# plt.ion()
for i in range(len(y_test)):
    if(y_test[i]==40):
        plt.plot(arr, X_test[i][:len(arr)])
        plt.savefig('../d40old/' + str(i))
        plt.clf()


exit()


plt.ion()
for ex_ind in range(100):
    ex = X_test[ex_ind]
    re = earlier[ex_ind]
    if(abs(re[2]-re[1])<=1):
        continue
    X = []
    y = []
    for i, ele in enumerate(ex):
        X.append(ele[0])
        y.append(200 + i*2)

    plt.plot(y, X)
    plt.title("Didnt Work " + str(ex_ind) + " .. True : " + str(re[2]) + " Predicted : " + str(re[1]))
    plt.show()
    plt.pause(2)
    plt.close()

for ex_ind in range(100):
    ex = X_test[ex_ind]
    re = earlier[ex_ind]
    if(abs(re[2]-re[1])>0.3):
        continue
    X = []
    y = []
    for i, ele in enumerate(ex):
        X.append(ele[0])
        y.append(200 + i*2)

    plt.plot(y, X)
    plt.title("Worked " + str(ex_ind) + " .. True : " + str(re[2]) + " Predicted : " + str(re[1]))
    plt.show()
    plt.pause(2)
    plt.close()
#
#
# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
#
#
#
# from keras.callbacks import ModelCheckpoint
# from keras.optimizers import RMSprop,Adam
# from keras.layers import Bidirectional
# from keras import regularizers
#
#
# def rnn_lstm(layers, params):
#     """Build RNN (LSTM) model on top of Keras and Tensorflow"""
#
#     model = Sequential()
#
#     model.add(Dense(units=layers[2],activation='relu'))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Dense(units=layers[1],activation='relu'))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Bidirectional(LSTM(units=layers[2], return_sequences=True,activation='relu'),input_shape=(layers[0], layers[1])))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Bidirectional(LSTM(units=layers[2],return_sequences=True,activation='relu')))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Bidirectional(LSTM(units=layers[2], return_sequences=False,activation='relu')))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Dense(units=layers[2],activation='relu'))
#     model.add(Dropout(params['dropout_keep_prob']))
#     model.add(Dense(units=layers[3],activation='relu',kernel_regularizer=regularizers.l2(0.001)))
#     #optimizer = Adam(clipvalue=0.5)
#     adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
#     model.compile(loss="mean_squared_error", optimizer=adam)
#
#     return model
#
#
# lstm_layer = [X_train.shape[1], X_train.shape[2], params['hidden_unit'], 1]
# model = rnn_lstm(lstm_layer, params)
#
#
# saved_model = "BI_LSTM_L2-6000"
# '''
# from keras.models import load_model
# try:
#     df_his = pd.read_csv("history_%s.csv" %(saved_model),index_col=0)
#     model = load_model("models_exp/%s" % (saved_model))
# except:
#     print("re train")
#     df_his = None
# '''
# df_his=None
#
# print("Start training")
#
# # Train RNN (LSTM) model with train set
# history = model.fit(X_train, y_train,
#           batch_size=params['batch_size'],
#           epochs=params['epochs'],
#           validation_split=params['validation_split'],
#           callbacks = [ModelCheckpoint(filepath="models_exp/"+saved_model,monitor='loss',verbose=1, save_best_only=True),\
#                       ModelCheckpoint(filepath="models_exp/"+saved_model+"_val",monitor='val_loss',verbose=1, mode='min',save_best_only=True)]
#           )
#
# # print(model.summary())
#
# # In[6]:
# if df_his is None:
#     df = pd.DataFrame(history.history)
#     df.to_csv("history_%s.csv" %(saved_model),header=True)
# else:
#     df = pd.concat([df_his, pd.DataFrame(history.history)]).reset_index()
#     df.to_csv("history_%s.csv" %(saved_model),header=True)
#
# from keras.models import load_model
# model = load_model("models_exp/%s" % (saved_model))
#
# print("Predicting")
# predict =  scaler_y.inverse_transform(model.predict(X_test))
#
# y_true  =  scaler_y.inverse_transform(y_test)
#
#
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error
#
#
# def NRMSD(y_true, y_pred):
#     rmsd = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
#     y_min = min(y_true)
#     y_max = max(y_true)
#
#     return rmsd/(y_max - y_min)
#
# def MAPE(y_true, y_pred):
#     y_true_select = (y_true!=0)
#
#     y_true = y_true[y_true_select]
#     y_pred = y_pred[y_true_select]
#
#     errors = y_true - y_pred
#     return sum(abs(errors/y_true))*100.0/len(y_true)
#
#
# # In[13]:
#
#
# nrmsd = NRMSD(y_true, predict)
# mape  = MAPE(y_true, predict)
# mae   = mean_absolute_error(y_true, predict)
# rmse   = np.sqrt(mean_squared_error(y_true, predict))
# print ("NRMSD",nrmsd)
# print ("MAPE",mape)
# print ("neg_mean_absolute_error",mae)
# print ("Root mean squared error",rmse)
#
# df = pd.DataFrame({"predict":predict.flatten(),"y_true": y_true.flatten()})
# df.to_csv('result-%s.csv' % (saved_model),index=True, header=True)
#
#
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
