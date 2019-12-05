import numpy as np
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential
from keras.layers.recurrent import GRU,LSTM
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.fftpack import fft, dct
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional
from keras import regularizers

from keras.models import load_model

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from keras.utils.vis_utils import plot_model

print("All Libraries Done")

# confirm TensorFlow sees the GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# confirm Keras sees the GPU
from keras import backend
print(backend.tensorflow_backend._get_available_gpus())
