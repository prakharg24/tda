import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

class PPCSDataLoader():
    def __init__(self, filename, param_dict, mode, scalerX='Robust', scalerY='MinMax'):
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

        # Convert sensors to training data using sliding windows
        self.sliding_window(param_dict['lower_step'], param_dict['sensor_channels'], param_dict['num_outputs'], param_dict['start_overhead'], mode)


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
            print("Scaler not recognized")
            exit()


    def sliding_window(self, step, input_size, num_outputs, output_st, mode):
        # Create training examples by using sliding window
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
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    end_ind = min(len(eX[0])//step, (eyo[0]-200)//(2*step) + someite)
                    st_ind = end_ind - num_outputs - output_st
                    new_X.append(eX[:,st_ind*step:end_ind*step])
                    # break
            else:
                # end_ind = (eyo[0]-200)//30 + num_outputs
                end_ind = len(eX[0])//step
                st_ind = 1
                new_X.append(eX[:,st_ind*step:end_ind*step])

                # end_ind = (eyo[0]-200)//30 + num_outputs//2
                # if(isFixed):
                #     st_ind = end_ind - num_outputs - output_st
                # else:
                #     st_ind = 1
                # new_X.append(eX[:,st_ind*step:end_ind*step])
                # print("H3", end_ind - st_ind)


            delay_st = output_st

            y_temp_reg = []
            y_temp_cls = []
            we_temp = []
            y_temp_reg.append([ey[0]])
            if(ey[0]==0.):
                y_temp_cls.append([0.])
            else:
                y_temp_cls.append([1.])

            if(mode=="train"):
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    new_y_cls.append(np.array(y_temp_cls))
                    # break
            else:
                new_y_cls.append(np.array(y_temp_cls))
            if(mode=="train"):
                for someite in range(num_outputs//2, num_outputs + output_st//2, 1):
                    # someite = num_outputs
                    new_y_reg.append(np.array(y_temp_reg))
                    # break
            else:
                new_y_reg.append(np.array(y_temp_reg))

            new_y_pos.append(eyo[0])
            # print("Done")
            # if(len(new_X[-1][0])==40*step):
            #     print(end_ind)
            #     print(delay_st)
            #     print(np.shape(y_temp_cls))

        X = []
        shape_dict = {}
        for ele in new_X:
            # print(np.shape(ele))
            # ele_four = np.abs(np.fft.fft(ele, axis=1))
            # ele = np.concatenate((ele, ele_four), axis=0)
            # ele_temp = ele.reshape((2*input_size, -1, step))
            ele_temp = ele.reshape((input_size, -1, step))
            ele_temp = ele_temp.transpose((1, 2, 0))
            X.append(ele_temp)

        self.X = X
        print(list(np.shape(self.X)))

        self.y_cls = new_y_cls
        self.y_reg = new_y_reg
        self.y_pos = new_y_pos


    def get_data(self):
        return (self.X, self.y_cls, self.y_reg, self.y_pos)
