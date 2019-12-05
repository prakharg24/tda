from keras.callbacks import ModelCheckpoint, LambdaCallback, LearningRateScheduler
from keras.optimizers import RMSprop,Adam
from keras.layers import Bidirectional, TimeDistributed
from keras import regularizers
from keras.layers.core import Dense, Activation, Dropout, Lambda, Flatten
from keras.layers import Input, Concatenate, Reshape
from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM

class HLSTM():
    def __init__(self, input_shape, upper_depth=2, lower_depth=2, dense_hidden_units=512,
                lower_lstm_units=256, upper_lstm_units=512, dropout=0.1):
        self.input_shape = input_shape
        self.upper_depth = upper_depth
        self.lower_depth = lower_depth
        self.dropout = dropout
        self.dense_hidden_units = dense_hidden_units
        self.lower_lstm_units = lower_lstm_units
        self.upper_lstm_units = upper_lstm_units

        assert (upper_depth>0 and lower_depth>0 and upper_lstm_units>0 and lower_lstm_units>0 and dense_hidden_units>0 and dropout<1 and dropout>=0)

    def build_classification_branch(self, x):
        x = TimeDistributed(Dense(units=self.dense_hidden_units, activation='relu'))(x)
        x = Dropout(self.dropout)(x)
        x = TimeDistributed(Dense(units=1, activation='sigmoid'), name="classification_output")(x)
        return x

    def build_regression_branch(self, x):
        x = TimeDistributed(Dense(units=self.dense_hidden_units, activation='relu'))(x)
        x = Dropout(self.dropout)(x)
        x = TimeDistributed(Dense(units=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)), name="regression_output")(x)
        return x

    def build_backbone(self, x):
        # Lower LSTM
        for i in range(self.lower_depth-1):
            x = TimeDistributed(LSTM(units=self.lower_lstm_units, return_sequences=True, activation='relu'))(x)
            x = Dropout(self.dropout)(x)

        x = TimeDistributed(LSTM(units=self.lower_lstm_units, return_sequences=False, activation='relu'))(x)
        x = Dropout(self.dropout)(x)

        # Upper LSTM
        for i in range(self.upper_depth):
            x = LSTM(units=self.upper_lstm_units,  return_sequences=True, activation='relu')(x)
            x = Dropout(self.dropout)(x)

        return x

    def hlstm_cls_train(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs)
        sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, self.upper_lstm_units))(x)
        classification_branch = self.build_classification_branch(sliced)

        model = Model(inputs=inputs, outputs=classification_branch)
        adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=adam)

        return model

    def hlstm_reg_train(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs)
        sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, self.upper_lstm_units))(x)
        regression_branch = self.build_regression_branch(sliced)

        model = Model(inputs=inputs, outputs=regression_branch)
        adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=adam)

        return model

    def hlstm_test(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs)
        sliced = Lambda(lambda x: x[:,20:,:], output_shape=(None, self.upper_lstm_units))(x)
        classification_branch = self.build_classification_branch(sliced)
        regression_branch = self.build_regression_branch(sliced)

        model = Model(inputs=inputs, outputs=[classification_branch, regression_branch])
        losses = {"classification_output": "binary_crossentropy", "regression_output": "mean_squared_error"}
        adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
        model.compile(loss=losses, optimizer=adam)
        
        return model
