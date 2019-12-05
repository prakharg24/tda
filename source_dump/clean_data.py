import os
import re
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = []

# inp_fldr1 = "../random_delay_data/"
# files1 = os.listdir(inp_fldr1)
#
# np.random.shuffle(files1)
#
# files.extend([(inp_fldr1 + ele) for ele in files1])
#
# inp_fldr2 = "../random_delay_data_v2/"
# files2 = os.listdir(inp_fldr2)
#
# np.random.shuffle(files2)
#
# files.extend([(inp_fldr2 + ele) for ele in files2])

inp_fldr3 = "../random_delay_data_class/"
files3 = os.listdir(inp_fldr3)

np.random.shuffle(files3)

files.extend([(inp_fldr3 + ele) for ele in files3])

new_arr_test = []
new_arr_train = []

par_nms = ['plant.evaporator.gasFlow.p', 'plant.generator.Pe', 'plant.superheater.gasFlow.T[7]']

arr_ele = []
arr_ele.append('delay')
arr_ele.append('delay_st')

for name in par_nms:
    for i in range(200, 1502, 2):
        arr_ele.append(name + "_" + str(i))

print(len(arr_ele))

new_arr_test.append(arr_ele)
new_arr_train.append(np.copy(arr_ele))

# files = ['case_loadCase_0_2397_ThermoPower.Examples.RankineCycle.Simulators.ClosedLoopAttackDelayPIDNoise_45_res.csv']

for j, ele in enumerate(files):
    arr_ele = []

    # print(j)
    # print(ele)

    st = re.search("PIDNoise_", ele).end()
    end = re.search("_res", ele).start()
    delay = int(ele[st:end])
    # print(delay)
    # if(delay!=0):
    #     continue
    arr_ele.append(delay)

    st = re.search("loadCase_", ele).end()
    new_ele = ele[st:]
    # print(new_ele)
    st = re.search("_", new_ele).end()
    new_ele = new_ele[st:]
    # print(new_ele)
    st = re.search("_", new_ele).end()
    # new_ele = ele[st:]
    end = re.search("_Thermo", new_ele).start()
    delay_st = int(new_ele[st:end])
    # print(delay_st)
    arr_ele.append(delay_st)

    if(delay_st<800):
        continue

    data_arr = []
    with open(ele) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                # print(row)
                line_count += 1
            else:
                try:
                    tm_ind = int(row[0])
                    if(tm_ind%2!=0):
                        continue
                    if(tm_ind<200):
                        continue
                    elif(tm_ind>1500):
                        continue
                    else:
                        data_arr.extend([row[1], row[3], row[4]])
                except Exception as e:
                    continue
                line_count += 1

    for ite in range(3):
        arr_ele.extend([float(data_arr[i]) for i in range(ite, len(data_arr), 3)])

    if(len(arr_ele)!=1955):
        print(ele)
        continue

    # arr = []
    # for i in range(210, 1502, 2):
    #     arr.append(i)

    # plt.plot(arr, arr_ele[5:(len(data_arr)//3)])
    # plt.title("Delay : " + str(delay) + " Delay Starts at :" + str(delay_st))
    # plt.show()
    # plt.clf()
    #
    # if(j==10):
    #     exit()

    someint = np.random.randint(0, 100)

    if(someint<70):
        new_arr_train.append(arr_ele)
        # print("Train : ", len(arr_ele), j)
    else:
        new_arr_test.append(arr_ele)
        # print("Test : ", len(arr_ele), j)
        # if(j%100==0):
        #     print(np.shape(new_arr_test))
# exit()
print(np.shape(new_arr_train))
print(np.shape(new_arr_test))

# signal1 = new_arr_train[1][15:len(new_arr_train[1])//3]
# y = [val for val in signal1]
# x = list(range(230, 2*len(signal1)+230, 2))
# plt.plot(x, y, color='red')
# signal1 = new_arr_train[2][15:len(new_arr_train[1])//3]
# y = [val for val in signal1]
# x = list(range(230, 2*len(signal1)+230, 2))
# plt.plot(x, y, color='blue')
# plt.show()
#
# exit()

new_arr_train = np.array(new_arr_train)
print(np.shape(new_arr_train))

dataset = pd.DataFrame(data=new_arr_train[1:,0:], columns=new_arr_train[0,0:])

print(dataset)

dataset.to_csv('train_data_200_1500_class.csv', header=True)


new_arr_test = np.array(new_arr_test)
print(np.shape(new_arr_test))

dataset = pd.DataFrame(data=new_arr_test[1:,0:], columns=new_arr_test[0,0:])

print(dataset)

dataset.to_csv('test_data_200_1500_class.csv', header=True)
