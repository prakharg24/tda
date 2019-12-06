import numpy as np

class Convergence_strategy():
    def __init__(self, beta):
        self.beta = beta

    def get_prediction(self, reg_arr):
        prev_pred = reg_arr[0]
        for i, ele in enumerate(reg_arr[1:]):
            if(abs(ele[0] - prev_pred)/prev_pred < self.beta/100):
                return (ele[0], i+1)
            prev_pred = ele[0]

        return (reg_arr[-1][0], len(reg_arr)-1)

class Waiting_time_strategy():
    def __init__(self, alpha):
        self.alpha = alpha

    def get_prediction(self, reg_arr):
        wait = min(len(reg_arr)-1, self.alpha//30)

        return (reg_arr[wait], wait)

class Classification_strategy():
    def __init__(self, n):
        self.n = n

    def get_prediction(self, cls_arr):
        continuous_count = 0
        for i, ele in enumerate(cls_arr):
            if(ele[0]>0.5):
                continuous_count += 1
            else:
                continuous_count = 0

            if(continuous_count==self.n):
                return (1, i)

        return (0, len(cls_arr)-1)

def get_reg_strategy(reg_name, reg_param):
    if(reg_name=='convergence'):
        return Convergence_strategy(reg_param)
    elif(reg_name=='waiting_time'):
        return Waiting_time_strategy(reg_param)
    else:
        print("Regression strategy not recognised")
        exit()

def get_cls_strategy(cls_n):

    assert(cls_n>0)
    return Classification_strategy(cls_n)
