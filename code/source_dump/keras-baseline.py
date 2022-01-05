import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import time
from scipy import sparse
import random
from functools import partial
from scipy.stats import boxcox
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from scipy import signal

random.seed(24)

import matplotlib.pyplot as plt


# test_df = pd.read_csv('test_data_200_1500_random.csv')
test_df = pd.read_csv('test_data_200_1500_class.csv')
# train_df = pd.read_csv('train_data_200_1500_random.csv')
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
        #     plt.savefig("plots/shifted" + str(i))
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
                y_temp_cls.append([0])
            else:
                y_temp_cls.append([1])

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

    # X_out = np.array(X_out)

    return np.array(X_out), np.array(y_out), np.array(y_out_st)

X_test, y_test, y_test_out = preprocess(test_df)
X_train, y_train, y_train_out = preprocess(train_df)

curr_signal = X_test[0][:len(X_test[0])//3]

plt.plot(curr_signal[0:], color='red')
sos = signal.butter(100, 50, 'low', fs=1000, output='sos')
filtered = signal.sosfilt(sos, curr_signal)

print(len(curr_signal))
print(len(filtered))

plt.plot(filtered[0:], color='green')
plt.show()
exit()

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
# scaler_y.fit(np.concatenate([y_test,y_train],axis=0))

data_loader = DataLoader(np.array(X_test),np.array(y_test), np.array(y_test_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], True)
X_test, y_test_classification, y_test_regression = data_loader.dataset()

data_loader = DataLoader(np.array(X_train),np.array(y_train), np.array(y_train_out), params["batch_size"], params["step"],params["input_size"],params["num_outputs"],params["output_st"], True)
X_train, y_train_classification, y_train_regression = data_loader.dataset()



# X_test = np.reshape(np.array(X_test)[:,:,:,0], [len(X_test), -1])
X_test = np.reshape(np.array(X_test), [len(X_test), -1])
# y_test_regression = np.reshape(y_test_regression, [-1])
y_test_classification = np.reshape(y_test_classification, [-1])
# X_train = np.reshape(np.array(X_train)[:,:,:,0], [len(X_train), -1])
X_train = np.reshape(np.array(X_train), [len(X_train), -1])
# y_train_regression = np.reshape(y_train_regression, [-1])
y_train_classification = np.reshape(y_train_classification, [-1])

print(np.shape(X_test))
# print(np.shape(y_test_regression))
print(np.shape(y_test_classification))
print(np.shape(X_train))
# print(np.shape(y_train_regression))
print(np.shape(y_train_classification))

print(y_train_classification[:10])
# exit()
# neigh = KNeighborsClassifier(n_neighbors=1)
# clf = RandomForestClassifier(n_estimators=100, random_state=0)
for c in [10, 30, 50, 70, 100, 150, 200, 250, 300, 500]:
    print(c)
    clf = SVC(gamma='scale', C=c)
    # neigh.fit(X_train, y_train_regression)
    # neigh.fit(X_train, y_train_classification)
    # clf.fit(X_train, y_train_regression)
    clf.fit(X_train, y_train_classification)

    # exit()


    print("Predicting")
    st = time.time()

    # predict = neigh.predict(X_test)
    predict = clf.predict(X_test)

    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

    # predict = np.array([[x] for x in predict])
    predict = np.array([x for x in predict])
    y_true_reg  = np.array([[x] for x in y_test_regression])
    y_true_cls  = np.array([x for x in y_test_classification])

    print("Time taken", time.time() - st)

    conf_mat = [[0, 0], [0, 0]]

    print("Complete")
    for i in range(len(predict)):
        conf_mat[predict[i]][y_true_cls[i]] += 1

    print(np.array(conf_mat))

    jmp = 10
    for lim in range(0, 50, jmp):
        conf_mat = [[0, 0], [0, 0]]
        # print("Limit ", lim, " to ", lim+jmp)
        for i in range(len(predict)):
            if(y_true_reg[i][0]>lim and y_true_reg[i][0]<=lim+jmp):
                conf_mat[predict[i]][y_true_cls[i]] += 1
         # print(np.array(conf_mat))

exit()

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
