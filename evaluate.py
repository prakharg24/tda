from dataloader import PPCSDataLoader
from hlstm_model import HLSTM
from utils import get_reg_strategy, get_cls_strategy, get_mae, get_rmse, get_confusion_matrix
import numpy as np
import random
import argparse
import joblib
import sys

import tensorflow as tf
from keras.models import load_model, save_model
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint

random.seed(24)
np.random.seed(24)

parser = argparse.ArgumentParser()
parser.add_argument('--reg_csv', type=str, help='CSV file containing the regression dataset', default='dataset/test_data_200_1500_regression.csv')
parser.add_argument('--cls_csv', type=str, help='CSV file containing the classification dataset', default='dataset/test_data_200_1500_classification.csv')
parser.add_argument('--model_prefix', type=str, help='Model prefix of the saved model', default='model_files/cleanup_version')

parser.add_argument('--lower_step', type=int, help='Number of steps for the lower LSTM before being reset', default=15)
parser.add_argument('--sensor_channels', type=int, help='Number of different sensors in the dataset', default=3)
parser.add_argument('--window_length', type=int, help='window_length * lower_step = Length of the sliding window', default=20)
parser.add_argument('--start_overhead', type=int, help='Number of initial outputs rejected', default=10)

parser.add_argument('--cls_strategy', type=int, help='Strategy used for the classification head (value of n)', default=1)
parser.add_argument('--reg_strategy', type=str, help='Strategy used for the regression head', default='convergence')
parser.add_argument('--reg_strategy_param', type=int, help='Parameter value for the regression strategy used (alpha for waiting time and beta for convergence)', default=2)

args = parser.parse_args(sys.argv[1:])

scalerX = joblib.load(args.model_prefix + "_scalerX.joblibdump")
scalerY = joblib.load(args.model_prefix + "_scalerY.joblibdump")

regression_data = PPCSDataLoader(args.reg_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, 'eval',
                                 scalerX=scalerX, scalerY=scalerY)

classification_data = PPCSDataLoader(args.cls_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, 'eval',
                                 scalerX=scalerX, scalerY=scalerY)

print("Data Loaded")

regX, regYc, regYr, regY_pos = regression_data.get_data()
clsX, clsYc, clsYr, clsY_pos = classification_data.get_data()

print("Combining dataset and loading model")

X = np.concatenate([regX, clsX], axis=0)
Yc = np.concatenate([regYc, clsYc], axis=0)
Yr = np.concatenate([regYr, clsYr], axis=0)
Y_pos = np.concatenate([regY_pos, clsY_pos], axis=0)

# Temporary testing on Classification dataset only
# X = np.concatenate([clsX], axis=0)
# Yc = np.concatenate([clsYc], axis=0)
# Yr = np.concatenate([clsYr], axis=0)
# Y_pos = np.concatenate([clsY_pos], axis=0)

final_model = load_model(args.model_prefix + "_complete.hdf5")

print("Starting evaluation")
fnl_pred = final_model.predict(X)
predC = fnl_pred[0]
predR = fnl_pred[1]

cls_strategy = get_cls_strategy(args.cls_strategy)
reg_strategy = get_reg_strategy(args.reg_strategy, args.reg_strategy_param)


strategy_out = []
characterisation_latency = []
for i in range(len(predC)):
    del_st = (Y_pos[i]-800)//30 + 1
    currC = predC[i][del_st:]
    currR = predR[i][del_st:]

    cls_out, cls_loc = cls_strategy.get_prediction(currC)

    if(cls_out==1):
        currR = currR[cls_loc:]
        reg_out, reg_loc = reg_strategy.get_prediction(currR)
        strategy_out.append([reg_out])
        characterisation_latency.append(cls_loc + reg_loc)
    else:
        strategy_out.append([0.])

print("\n\nRESULTS")
print("-------\n\n")

print("Average Latency (in sec) : ", np.mean(characterisation_latency)*30)

strategy_out = scalerY.inverse_transform(strategy_out)
Yr = scalerY.inverse_transform([ele[0] for ele in Yr])

acc, conf = get_confusion_matrix(strategy_out, Yc)
print("Classification accuracy : %.2f%%" % (acc*100))
print("Classification confusion matrix\n", conf)

print("Overall MAE error : ", get_mae(strategy_out, Yr))
print("Overall RMSE error : ", get_rmse(strategy_out, Yr))
