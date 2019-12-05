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
        x = TimeDistributed(Dense(units=self.dense_hidden_units, activation='relu'), name="classification_hidden")(x)
        x = Dropout(self.dropout)(x)
        x = TimeDistributed(Dense(units=1, activation='sigmoid'), name="classification_output")(x)
        return x

    def build_regression_branch(self, x):
        x = TimeDistributed(Dense(units=self.dense_hidden_units, activation='relu'), name="regression_hidden")(x)
        x = Dropout(self.dropout)(x)
        x = TimeDistributed(Dense(units=1, activation='relu', kernel_regularizer=regularizers.l2(0.001)), name="regression_output")(x)
        return x

    def build_backbone(self, x, name_prefix=""):
        # Lower LSTM
        for i in range(self.lower_depth-1):
            x = TimeDistributed(LSTM(units=self.lower_lstm_units, return_sequences=True, activation='relu'), name=name_prefix + "_td_" + str(i+1))(x)
            x = Dropout(self.dropout)(x)

        x = TimeDistributed(LSTM(units=self.lower_lstm_units, return_sequences=False, activation='relu'), name=name_prefix + "_td_" + str(self.lower_depth))(x)
        x = Dropout(self.dropout)(x)

        # Upper LSTM
        for i in range(self.upper_depth):
            x = LSTM(units=self.upper_lstm_units,  return_sequences=True, activation='relu', name=name_prefix + "_lstm_" + str(i+1))(x)
            x = Dropout(self.dropout)(x)

        return x

    def hlstm_cls_train(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs, name_prefix="cls")
        sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, self.upper_lstm_units))(x)
        classification_branch = self.build_classification_branch(sliced)

        model = Model(inputs=inputs, outputs=classification_branch)
        return model

    def hlstm_reg_train(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs, name_prefix="reg")
        sliced = Lambda(lambda x: x[:,-1:,:], output_shape=(None, self.upper_lstm_units))(x)
        regression_branch = self.build_regression_branch(sliced)

        model = Model(inputs=inputs, outputs=regression_branch)
        return model

    def hlstm_test(self):
        inputs = Input(shape=(None, self.input_shape[0], self.input_shape[1]))

        x = self.build_backbone(inputs, name_prefix="comb")
        sliced = Lambda(lambda x: x[:,20:,:], output_shape=(None, self.upper_lstm_units))(x)
        classification_branch = self.build_classification_branch(sliced)
        regression_branch = self.build_regression_branch(sliced)

        model = Model(inputs=inputs, outputs=[classification_branch, regression_branch])
        return model

    def get_regression_model(self):
        # Step 1 : A randomized regression model
        reg_model = self.hlstm_reg_train()

        adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
        model.compile(loss='mean_squared_error', optimizer=adam)

        return reg_model

    def get_classification_model(self, reg_weights):
        # Step 2 : A classification head placed on the backbone trained during regression
        cls_model = self.hlstm_cls_train()
        reg_dummy = self.hlstm_reg_train()
        try:
            reg_dummy.set_weights(reg_weights)
        except ValueError as e:
            print("Error : Regression weights provided not compatible with the model")

        # Copy lower LSTM
        for i in range(self.lower_depth):
            cls_model.get_layer(name='cls_td_' + str(i+1)).set_weights(reg_dummy.get_layer(name='reg_td_' + str(i+1)).get_weights())
            cls_model.get_layer(name='cls_td_' + str(i+1)).trainable = False

        # Copy upper LSTM
        for i in range(self.lower_depth):
            cls_model.get_layer(name='cls_lstm_' + str(i+1)).set_weights(reg_dummy.get_layer(name='reg_lstm_' + str(i+1)).get_weights())
            cls_model.get_layer(name='cls_lstm_' + str(i+1)).trainable = False

        adam = Adam(clipvalue=0.5,lr=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.001, amsgrad=False)
        model.compile(loss='binary_crossentropy', optimizer=adam)

        return cls_model

    def get_final_model(self, reg_weights, cls_weights):
        # Step 3 : Comine both trained models into one
        final_model = self.hlstm_test()
        cls_dummy = self.hlstm_cls_train()
        try:
            cls_dummy.set_weights(cls_weights)
        except ValueError as e:
            print("Error : Classification weights provided not compatible with the model")
        reg_dummy = self.hlstm_reg_train()
        try:
            reg_dummy.set_weights(reg_weights)
        except ValueError as e:
            print("Error : Regression weights provided not compatible with the model")

        # Copy lower LSTM
        for i in range(self.lower_depth):
            final_model.get_layer(name='comb_td_' + str(i+1)).set_weights(reg_dummy.get_layer(name='reg_td_' + str(i+1)).get_weights())
            final_model.get_layer(name='comb_td_' + str(i+1)).trainable = False

        # Copy upper LSTM
        for i in range(self.lower_depth):
            final_model.get_layer(name='comb_lstm_' + str(i+1)).set_weights(reg_dummy.get_layer(name='reg_lstm_' + str(i+1)).get_weights())
            final_model.get_layer(name='comb_lstm_' + str(i+1)).trainable = False

        # Copy multi head weights
        final_model.get_layer(name='classification_hidden').set_weights(cls_dummy.get_layer(name='classification_hidden').get_weights())
        final_model.get_layer(name='classification_output').set_weights(cls_dummy.get_layer(name='classification_output').get_weights())
        final_model.get_layer(name='regression_hidden').set_weights(reg_dummy.get_layer(name='regression_hidden').get_weights())
        final_model.get_layer(name='regression_output').set_weights(reg_dummy.get_layer(name='regression_output').get_weights())

        return final_model
