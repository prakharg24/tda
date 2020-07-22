from dataloader import PPCSDataLoader, AGCDataLoader
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
parser.add_argument('--reg_csv', type=str, help='CSV file containing the regression dataset', default='dataset/train_data_200_1500_regression.csv')
parser.add_argument('--cls_csv', type=str, help='CSV file containing the classification dataset', default='dataset/train_data_200_1500_classification.csv')
parser.add_argument('--model_prefix', type=str, help='Added as prefix to the name while saving the model', default='model_files/cleanup_version')

parser.add_argument('--lower_step', type=int, help='Number of steps for the lower LSTM before being reset', default=15)
parser.add_argument('--sensor_channels', type=int, help='Number of different sensors in the dataset', default=3)
parser.add_argument('--window_length', type=int, help='window_length * lower_step = Length of the sliding window', default=20)
parser.add_argument('--start_overhead', type=int, help='Number of initial outputs rejected', default=10)
parser.add_argument('--sliding_step', type=int, help='Slide value while creating training dataset', default=1)

parser.add_argument('--upper_depth', type=int, help='Depth of the Upper LSTM', default=2)
parser.add_argument('--lower_depth', type=int, help='Depth of the Lower LSTM', default=2)
parser.add_argument('--dense_hidden_units', type=int, help='Number of hidden units in the dense fully connected layer', default=512)
parser.add_argument('--upper_lstm_units', type=int, help='Number of hidden units in the upper LSTM', default=512)
parser.add_argument('--lower_lstm_units', type=int, help='Number of hidden units in the lower LSTM', default=256)
parser.add_argument('--dropout', type=float, help='Dropout percentage', default=0.1)

parser.add_argument('--epoch_regression', type=int, help='Number of epochs while training the regression model', default=100)
parser.add_argument('--epoch_classification', type=int, help='Number of epochs while training the classification model', default=50)
parser.add_argument('--batch_size', type=int, help='Batch Size', default=64)

args = parser.parse_args(sys.argv[1:])

regression_data = PPCSDataLoader(args.reg_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, 'train',
                                 sliding_step=args.sliding_step)

scalerX, scalerY = regression_data.get_scalers()
joblib.dump(scalerX, model_prefix + '_scalerX.joblibdump')
joblib.dump(scalerY, model_prefix + '_scalerY.joblibdump')

classification_data = PPCSDataLoader(args.cls_csv, args.lower_step, args.sensor_channels,
                                 args.window_length, args.start_overhead, 'train',
                                 sliding_step=args.sliding_step, scalerX=scalerX, scalerY=scalerY)

print("Data Loaded")

regX, _, regY, regY_pos = regression_data.get_data()
clsX, clsY, _, clsY_pos = classification_data.get_data()

print("Setting up training environment")

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

hlstm_class = HLSTM(regX[0].shape[1:3], args.upper_depth, args.lower_depth, args.dense_hidden_units, args.lower_lstm_units, args.upper_lstm_units, args.dropout)

model = hlstm_class.get_regression_model()
print("Training Regression Model")
history_reg = model.fit(regX, regY, batch_size=args.batch_size, epochs=args.epoch_regression, validation_split=0.1,
                        callbacks=[ModelCheckpoint(filepath=args.model_prefix + "_regression.hdf5", monitor="val_loss", verbose=1, save_best_only=True)])

reg_weights = load_model(args.model_prefix + "_regression.hdf5").get_weights()
tf.reset_default_graph()
set_session(tf.Session(config=config))

model = hlstm_class.get_classification_model(reg_weights)
print("Training Classification Model")
history_cls = model.fit(clsX, clsY, batch_size=args.batch_size, epochs=args.epoch_classification, validation_split=0.1,
                        callbacks=[ModelCheckpoint(filepath=args.model_prefix + "_classification.hdf5", monitor="val_loss", verbose=1, save_best_only=True)])

cls_weights = load_model(args.model_prefix + "_classification.hdf5").get_weights()
tf.reset_default_graph()
set_session(tf.Session(config=config))

model = hlstm_class.get_final_model(reg_weights, cls_weights)
print("Saving Final Model")
save_model(model, args.model_prefix + "_complete.hdf5")
