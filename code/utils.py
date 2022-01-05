import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class Convergence_strategy():
    def __init__(self, beta):
        self.beta = beta

    def get_prediction(self, reg_arr):
        prev_pred = reg_arr[0]
        for i, ele in enumerate(reg_arr[1:]):
            if(prev_pred==0. and abs(ele[0])<self.beta/100):
                return (ele[0], i+1)
            elif(prev_pred!=0. and abs(ele[0] - prev_pred)/prev_pred<self.beta/100):
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

def get_cls_strategy(cls_n):

    assert(cls_n>0)
    return Classification_strategy(cls_n)

def get_reg_strategy(reg_name, reg_param):
    if(reg_name=='convergence'):
        return Convergence_strategy(reg_param)
    elif(reg_name=='waiting_time'):
        return Waiting_time_strategy(reg_param)
    else:
        print("Regression strategy not recognised")
        exit()

def get_mae(outs1, outs2):
    return mean_absolute_error([[int(ele[0])] for ele in outs1], [[int(ele[0])] for ele in outs2])

def get_rmse(outs1, outs2):
    return np.sqrt(mean_squared_error([[int(ele[0])] for ele in outs1], [[int(ele[0])] for ele in outs2]))

def get_confusion_matrix(outs1, outs2):
    matrix = np.zeros((2,2))
    for e1, e2 in zip(outs1, outs2):
        if(e1[0]>0 and e2[0][0]==1):
            matrix[1][1] += 1
        elif(e1[0]>0 and e2[0][0]==0):
            matrix[1][0] += 1
        elif(e1[0]==0 and e2[0][0]==1):
            matrix[0][1] += 1
        elif(e1[0]==0 and e2[0][0]==0):
            matrix[0][0] += 1

    return (matrix[0][0] + matrix[1][1])/np.sum(matrix), matrix
