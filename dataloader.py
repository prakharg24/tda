import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class PPCSDataLoader():
    def __init__(self, filename, lower_step, sensor_channels, window_length, start_overhead,
                sliding_step, mode, scalerX='Robust', scalerY='MinMax'):
        data_df = pd.read_csv(filename)
        # Delay value and Delay launch point
        self.delay = np.array(data_df[['delay']])
        self.delay_st = np.array(data_df[['delay_st']])

        # Removing unwanted columns such that only the sensor signal remains
        drop_terms = []
        drop_terms.extend(['delay', 'Unnamed: 0', 'delay_st'])
        self.sensors = data_df.drop(drop_terms, axis=1)

        # Setup scalers and scale data
        self.set_scalers(scalerX, scalerY)

        # Convert sensors to dataset that can be used directly by the learning model
        self.sliding_window(lower_step, sensor_channels, window_length, start_overhead, sliding_step, mode)


    def scaler_from_name(self, sc_name):
        # Return a new scaler from the name provided
        if(sc_name=='Robust'):
            return RobustScaler()
        elif(sc_name=='MinMax'):
            return MinMaxScaler()
        elif(sc_name=='Standard'):
            return StandardScaler()
        elif(sc_name=='MaxAbs'):
            return MaxAbsScaler()
        else:
            return None

    def set_scalers(self, scalerX, scalerY):
        # Define the scaler and scale the input dataset
        self.scalerX = self.scaler_from_name(scalerX)
        self.scalerY = self.scaler_from_name(scalerY)

        if(self.scalerX is not None):
            self.scalerX.fit(self.sensors)
            self.scalerY.fit(self.delay)
        else:
            self.scalerX = scalerX
            self.scalerY = scalerY

        try:
            self.sensors = self.scalerX.transform(self.sensors)
            self.delay = self.scalerY.transform(self.delay)
        except TypeError as e:
            print("Error : Scaler not recognized")
            exit()


    def sliding_window(self, step, input_size, window_length, output_st, sliding_step, mode):
        # Create training/testing dataset
        X_shape = list(self.sensors.shape)
        X_shape[-1] = int(X_shape[-1]/input_size)
        seq_length = int(X_shape[-1]/step)
        lengh = step*seq_length

        # Cut the input sensor signal such that it is perfectly divisible by the steps
        self.sensors = self.sensors.reshape((X_shape[0],input_size,-1))[:,:,:lengh]

        slide_X = []
        slide_y_cls = []
        slide_y_reg = []
        slide_y_pos = []

        for ex, ey, eyp in zip(self.sensors, self.delay, self.delay_st):
            # Additional check in case the dataset in not clean.
            if(eyp[0]<800):
                continue

            if(mode=="train"):
                # create sliding window
                for someite in range((window_length - output_st)//2, window_length - output_st//2, sliding_step):
                    end_ind = min(len(eX[0])//step, (eyp[0]-200)//(2*step) + someite)
                    st_ind = end_ind - num_outputs - output_st

                    slide_X.append(ex[:,st_ind*step:end_ind*step])
                    slide_y_cls.append([[float(ey[0])>0.]])
                    slide_y_reg.append([[ey[0]]])
                    slide_y_pos.append(eyp[0])
            else:
                # create test dataset (which is the complete trace)
                end_ind = len(eX[0])//step
                st_ind = 1

                slide_X.append(ex[:,st_ind*step:end_ind*step])
                slide_y_cls.append([[float(ey[0])>0.]])
                slide_y_reg.append([[ey[0]]])
                slide_y_pos.append(eyp[0])

        X = []
        for ele in slide_X:
            ele_temp = ele.reshape((input_size, -1, step))
            ele_temp = ele_temp.transpose((1, 2, 0))
            X.append(ele_temp)

        self.X = np.array(X)
        self.y_cls = np.array(slide_y_cls)
        self.y_reg = np.array(slide_y_reg)
        self.y_pos = np.array(slide_y_pos)

    def get_scalers(self):
        return (self.scalerX, self.scalerY)

    def get_data(self):
        return (self.X, self.y_cls, self.y_reg, self.y_pos)
