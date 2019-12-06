from dataloader import PPCSDataLoader
from hlstm_model import HLSTM
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
parser.add_argument('--scalerX', type=str, help='joblibdump of the scaler for sensor data', default='model_files/cleanup_version_scalerX.joblibdump')
parser.add_argument('--scalerY', type=str, help='joblibdump of the scaler for delay value data', default='model_files/cleanup_version_scalerY.joblibdump')
parser.add_argument('--model_complete', type=str, help='Location of the complete model file', default='model_files/cleanup_version_complete.hdf5')

parser.add_argument('--lower_step', type=int, help='Number of steps for the lower LSTM before being reset', default=15)
parser.add_argument('--sensor_channels', type=int, help='Number of different sensors in the dataset', default=3)
parser.add_argument('--window_length', type=int, help='window_length * lower_step = Length of the sliding window', default=20)
parser.add_argument('--start_overhead', type=int, help='Number of initial outputs rejected', default=10)

parser.add_argument('--cls_strategy', type=int, help='Strategy used for the classification head (value of n)', default=1)
parser.add_argument('--reg_strategy', type=str, help='Strategy used for the regression head', default='convergence')
parser.add_argument('--reg_strategy_param', type=int, help='Parameter value for the regression strategy used (alpha for waiting time and beta for convergence)', default=2)

args = parser.parse_args(sys.argv[1:])

scalerX = joblib.load(args.scalerX)
scalerY = joblib.load(args.scalerY)

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

final_model = load_model(args.model_complete)

fnl_pred = final_model.predict(X)
predC = fnl_pred[0]
predR = fnl_pred[1]
