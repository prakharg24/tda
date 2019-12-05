import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler


# 30, 9, 16, 31, 3, 36
ma = [33, 29, 17, 11, 4, 2, 0]
ca = [7, 11, 20, 34, 39, 47, 1]

for i in range(len(ma)):
    ttl = ma[i] + ca[i]
    ma[i] = ma[i]/ttl
    ca[i] = ca[i]/ttl

names = ['1', '2', '3', '4', '5', '6', '{7 .. 50}']

f, ax = plt.subplots(1, figsize=(8, 8))

# plt.rcParams['hatch.linewidth'] = 2.0

# plt.rcParams['xtick.labelsize'] = 18

top_patterns = ['-', '-', '-', '-', '-', '-', '-']
bot_patterns = ['x', 'x', 'x', 'x', 'x', 'x', 'x']

ax.bar([0, 1, 2, 3, 4, 5, 6], ma, width=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], edgecolor='black', color='white', tick_label=names, label='Predicted : No attack', linewidth=2)
ax.bar([0, 1, 2, 3, 4, 5, 6], ca, width=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], bottom = ma, hatch = '/', edgecolor='black', color='white', label='Predicted : Attack', linewidth=2)
plt.rcParams.update({'hatch.linewidth':2.0, 'axes.labelsize':24, 'axes.titlesize':24, 'legend.fontsize':24, 'xtick.labelsize':24, 'ytick.labelsize':24})

# ax.set_title('Distribution of the Missed Detection Error in our model')
ax.set_xlabel('Groundtruth Delay Attack Values (s)', fontsize=24)
ax.set_ylabel('Error Distribution', fontsize=24)
ax.set_ylim(0, 1.2)
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=24)
ax.set_xticklabels(names, fontsize=24)
ax.legend(handlelength=3, fontsize=22, framealpha=1, ncol=2)
plt.show()
exit()






# train_df = pd.read_csv('train_data_200_1500_random.csv')
# test_df = pd.read_csv('test_data_200_1500_random.csv')
#
# def preprocess(df_inp):
#     y_out = df_inp[['delay']]
#     y_out_st = df_inp[['delay_st']]
#
#     drop_terms = []
#
#     drop_terms.extend(['delay', 'Unnamed: 0', 'delay_st'])
#     X_out = df_inp.drop(drop_terms,axis=1)
#
#     return np.array(X_out), np.array(y_out), np.array(y_out_st)
#
# def plt_subplot(X_inp, subi, txt):
#     signal1 = X_inp[1959][:len(X_inp[0])//3]
#     print(np.shape(signal1))
#
#     X = []
#     y = []
#     for i in range(len(signal1)):
#         X.append(200 + 2*i)
#         if(subi==3 and i>410):
#             div = 1 + 0.4
#             y.append(signal1[i]/div)
#         else:
#             y.append(signal1[i])
#
#
#     plt.subplot(2, 2, subi)
#     plt.plot(X, y)
#     plt.title(txt)
#     plt.yticks([])
#
#
# X_train, y_train, y_train_out = preprocess(train_df)
# X_test, y_test, y_test_out = preprocess(test_df)
#
# # for i, ele in enumerate(y_test):
# #     if(ele[0]==0):
# #         print(y_train_out[i])
# #         print(ele[0])
# #         print(i)
# # exit()
#
# plt_subplot(X_test, 1, 'Original Signal')
#
# scaler_X = MinMaxScaler()
# scaler_X.fit(np.concatenate([X_train, X_test],axis=0))
# X_test2 = scaler_X.transform(X_test)
#
# plt_subplot(X_test2, 2, 'MinMax Scaling')
#
# # plt.show()
# # exit()
#
# scaler_X = StandardScaler()
# scaler_X.fit(np.concatenate([X_train, X_test],axis=0))
# X_test2 = scaler_X.transform(X_test)
#
# # X_test2[410:] = X_test2[410:]/5
# plt_subplot(X_test2, 3, 'Standard Scaling')
#
# scaler_X = RobustScaler()
# scaler_X.fit(np.concatenate([X_train, X_test],axis=0))
# X_test2 = scaler_X.transform(X_test)
#
# plt_subplot(X_test2, 4, 'Robust Scaling')
#
# plt.show()
#
# exit()



def arrowed_spines(fig, ax):

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # removing the default axis on all sides:
    for side in ['bottom','right','top','left']:
        ax.spines[side].set_visible(False)

    # removing the axis ticks
    plt.xticks([]) # labels
    plt.yticks([])
    ax.xaxis.set_ticks_position('none') # tick markers
    ax.yaxis.set_ticks_position('none')

    # get width and height of axes object to compute
    # matching arrowhead length and width
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = 1./20.*(ymax-ymin)
    hl = 1./20.*(xmax-xmin)
    lw = 1. # axis line width
    ohg = 0.3 # arrow overhang

    # compute matching arrowhead length and width
    yhw = hw/(ymax-ymin)*(xmax-xmin)* height/width
    yhl = hl/(xmax-xmin)*(ymax-ymin)* width/height

    # draw x and y axis
    ax.arrow(xmin, 0, xmax-xmin, 0., fc='k', ec='k', lw = lw,
             head_width=hw, head_length=hl, overhang = ohg,
             length_includes_head= True, clip_on = False)

    ax.arrow(0, ymin, 0., ymax-ymin, fc='k', ec='k', lw = lw,
             head_width=yhw, head_length=yhl, overhang = ohg,
             length_includes_head= True, clip_on = False)

# a = [31.454508,23.12907,16.470192,17.64823,18.453684,16.256227,16.86458,16.699118,16.318373,16.097889]
b = [15.292989,31.384846,39.67983,41.00639,40.634956,39.887325,38.288143,39.810955,40.528917,40.658237, 40.672134]

# atar = 16
btar = 43

x = []
y = []

plt.rcParams.update({'font.size': 14})
#
# for i, ele in enumerate(a):
#     y.append(int(ele))
#     x.append(i+1)
#
# plt.plot(x, y, color='red', marker='o', markerfacecolor='red', linewidth=2, label='Case 1 : Predictions')
# plt.axhline(y=atar, color='red', xmin=1/20, xmax=19/20, linestyle='--', label='Case 1 : Gold Label')

x = []
y = []

for i, ele in enumerate(b):
    y.append(int(ele))
    x.append(i+1)
# plt.gca().axison = False
plt.plot(x, y, color='purple', marker='o', markerfacecolor='purple', markersize=9, linewidth=2, label='Predictions')
plt.axhline(y=btar, color='blue', xmin=1/20, xmax=19/20, linestyle='-', label='Gold Label')
# plt.ylim(10, 50)
# plt.xlim(0, 10)
# plt.axhline(y=25, color='black', xmin=1/20, xmax=19/20, linewidth=2, linestyle='--', label='Trigger Based')
plt.axvline(x=8, color='red', ymin=(b[0]-5)/50, ymax=(btar + 3)/50, linewidth=2, linestyle='-.', label='Waiting Period Based')
plt.arrow(1, 22, 6.8, 0, head_width=0.5, head_length=0.3, linewidth=2, color='r', length_includes_head=True)
plt.arrow(7.8, 22, -6.8, 0, head_width=0.5, head_length=0.3, linewidth=2, color='r', length_includes_head=True)
plt.axhline(y=40.8, color='magenta', xmin=12/20, xmax=31/30, linewidth=2, linestyle=':', label='Convergence Based')
plt.axhline(y=39.3, color='magenta', xmin=12/20, xmax=31/30, linewidth=2, linestyle=':')
plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

plt.xlabel(r'Timeslots ($*\omega$)')
plt.ylabel('Delay Value (s)')
# plt.legend()

# plt.axvline(x=2, color='magenta', ymin=b[1]/50, ymax=btar/50, linewidth=1, linestyle='--')
plt.annotate(xy=(1, btar-1.5), s='Groundtruth value', color='black', size=14)
# plt.annotate(xy=(9, 23.5), s='Trigger', color='black', size=14)
# plt.annotate(xy=(2, b[1]-1), xytext=(2, 27), arrowprops=dict(facecolor='black', arrowstyle='->'), s='Trigger output', color='black', size=14)
plt.annotate(xy=(3.2, 20), s='Fixed waiting time', color='black', size=14)
plt.annotate(xy=(8, b[7]-1), xytext=(5.4, 33), arrowprops=dict(facecolor='black', arrowstyle='->'), s='Waiting period \nbased output', color='black', size=14)
plt.annotate(xy=(9, 41), s='Convergence', color='black', size=14)
plt.annotate(xy=(10, 39.5), xytext=(9, 33), arrowprops=dict(facecolor='black', arrowstyle='->'), s='Convergence \nbased output', color='black', size=14)
#
# plt.axvline(x=6, color='magenta', ymin=b[5]/50, ymax=btar/50, linewidth=1, linestyle='--')
# plt.text(x=5, y=btar+1, s='Waiting Period :\nMore accurate than \ninitial predictions. \nBut flexibility of waiting time \nacross examples not available.', color='magenta', size=10)
#
# plt.axvline(x=10, color='magenta', ymin=b[9]/50, ymax=btar/50, linewidth=1, linestyle='--')
# plt.text(x=9, y=btar+1, s='Convergence based :\nHighly accurate \nbecause the output \nis now stable. \nLonger waiting time.', color='magenta', size=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

xmin, xmax = plt.gca().get_xlim()
ymin, ymax = plt.gca().get_ylim()
plt.arrow(xmin, ymin, xmax - xmin, 0, head_width=0.5, head_length=0.2, linewidth=1, color='black', length_includes_head=True)
plt.arrow(xmin, ymin, 0, ymax - ymin, head_width=0.2, head_length=0.5, linewidth=1, color='black', length_includes_head=True)
plt.xlim([xmin-0.3, xmax])
plt.ylim([ymin-0.3, ymax])


# print(xmax, xmin, ymax, ymin)

plt.show()
exit()

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.suptitle('Horizontally stacked subplots')
plt.rcParams.update({'axes.labelsize':22, 'axes.titlesize':22, 'legend.fontsize':22, 'xtick.labelsize':22, 'ytick.labelsize':22})


false_alarms  = [4.69, 3.19, 2.25, 1.55]
missed_detections = [2.91, 3.24, 3.62, 4.04]
average_time = [26, 55, 83, 109]

plt.plot(average_time, false_alarms, color='red', marker='o', markerfacecolor='red', linewidth=2, linestyle='--', label='FP')
plt.plot(average_time, missed_detections, color='blue', marker='s', markerfacecolor='blue', linewidth=2, linestyle='-.', label='FN')
plt.plot(average_time, [e1 + e2 for e1, e2 in zip(false_alarms, missed_detections)], color='black', marker='x', markerfacecolor='black', linewidth=2, label='Total error')
plt.xlabel('Average Time taken (s)', fontsize=22)
plt.ylabel('Error Percentage (%)', fontsize=22)

n = ['n=1', 'n=2', 'n=3', 'n=4']
for i, txt in enumerate(n):
    plt.annotate(txt, (average_time[i], false_alarms[i] + missed_detections[i] + 0.2), fontsize=22)
    plt.axvline(x=average_time[i], ymin=min(false_alarms[i], missed_detections[i])/9, ymax = (false_alarms[i] + missed_detections[i])/9, linestyle=':', linewidth=0.5, color='magenta')

plt.grid(True)
plt.legend(handlelength=3)
plt.ylim(0, 9)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], fontsize=22)
plt.xlim(20, 120)
plt.xticks([20, 40, 60, 80, 100], fontsize=22)
plt.title('Classification Strategies', fontsize=22)
plt.show()
#
exit()



# fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))

# Convergence based with classification on the first 1

# for i in [0, 1]:
#     for j in [0, 1]:
#         if(i!=1 or j!=1):
#             axes[i, j].plot([], [], marker='o', color='red', label='Classification : n=1', linestyle='-.')
#             axes[i, j].plot([], [], marker='^', color='blue', label='Classification : n=2', linestyle='--')
#             axes[i, j].plot([], [], marker='s', color='purple', label='Classification : n=3', linestyle=':')
#         axes[i, j].set_xlabel('Average Time taken (s)')
#         axes[i, j].set_ylabel('Error (MAE)')
#         axes[i, j].set_xlim(50, 250)
#         axes[i, j].set_ylim(1.4, 3.5)
#         axes[i, j].grid(True)
plt.rcParams.update({'axes.labelsize':22, 'axes.titlesize':22, 'legend.fontsize':22, 'xtick.labelsize':22, 'ytick.labelsize':22})
plt.grid(True)
plt.xlabel('Average Time Taken (s)', fontsize=22)
plt.ylabel('MAE (s)', fontsize=22)
plt.ylim(1.5, 1.95)
plt.yticks([1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95], fontsize=22)
plt.xticks([100, 120, 140, 160, 180, 200, 220, 240], fontsize=22)
plt.title('Regression Strategies', fontsize=22)
plt.plot([], [], color='red', label="Convergence based", linestyle='-')
# plt.plot([], [], color='blue', label="Trigger based", linestyle='-.')
plt.plot([], [], color='purple', label="Waiting time based", linestyle=':')
# plt.scatter([], [], marker='*', color='black', label='Classification : n=1', linestyle='-')
# plt.scatter([], [], marker='^', color='black', label='Classification : n=2', linestyle='-.')
# plt.scatter([], [], marker='s', color='black', label='Classification : n=3', linestyle=':')

# plt.scatter([], [], marker='o', color='black', label='c=20')
# plt.scatter([], [], marker='^', color='black', label='c=10')
# plt.scatter([], [], marker='s', color='black', label='c=5')
# plt.scatter([], [], marker='x', color='black', label='c=2')

# plt.plot([], [], marker='o', color='red', label='Regression : Convergence', linestyle='-.')
# plt.plot([], [], marker='^', color='blue', label='Regression : Trigger based', linestyle='--')
# plt.plot([], [], marker='s', color='purple', label='Regression : Waiting period', linestyle=':')
#
# plt.title('Regression Strategy : Convergence')
# plt.title('Regression Strategy : Trigger based')
# plt.title('Regression Strategy : Waiting period')
# plt.title('Classification Strategy : n=3')


y = [2.75, 2.75, 2.75, 2.75]
x = [150, 175, 200, 225]
text = ['c=20', 'c=10', 'c=5', 'c=2']
# plt.plot(x, y, color='red', marker='|', markerfacecolor='black', markeredgewidth=4, linewidth=2, linestyle='-')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (x[i] + 0.03, y[i] + 0.03))


y = [5.25/2, 5.25/2, 5.25/2, 5.25/2]
x = [150, 175, 200, 225]
text = ['t=20', 't=25', 't=30', 't=35']
# plt.plot(x, y, color='blue', marker='|', markerfacecolor='black', markeredgewidth=4, linewidth=2, linestyle='-.')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (x[i] + 0.03, y[i] + 0.03))


y = [2.5, 2.5, 2.5, 2.5]
x = [150, 175, 200, 225]
text = ['w=60', 'w=90', 'w=120', 'w=150']
# plt.plot(x, y, color='purple', marker='|', markerfacecolor='black', markeredgewidth=4, linewidth=2, linestyle=':')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (x[i] + 0.03, y[i] + 0.03))



mae = [2.05, 2.22, 2.39, 2.56]
average_time = [105+26, 71+26, 51+26, 39+26]
text = ['c=2', 'c=5', 'c=10', 'c=20']
# plt.plot(average_time, mae, color='red', marker='*', markerfacecolor='red', linewidth=2, linestyle='-')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.02))

# Convergence based with classification on 2 consecutive 1s
mae = [1.86, 1.98, 2.07, 2.09]
average_time = [83+55, 53+55, 38+55, 32+55]
text = ['c=2', 'c=5', 'c=10', 'c=20']
# plt.plot(average_time, mae, color='red', marker='^', markerfacecolor='red', linewidth=2, linestyle='-')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.02))


# Convergence based with classification on 3 consecutive 1s
mae = [1.68, 1.77, 1.80, 1.81]
average_time = [66+83, 45+83, 34+83, 31+83]
text = [r'$\beta$=2%', r'$\beta$=5%', r'$\beta$=10%', r'$\beta$=20%']
plt.plot(average_time, mae, color='red', marker='s', markerfacecolor='red', linewidth=2, linestyle='-')
for i, txt in enumerate(text):
    if(i==2):
        plt.annotate(txt, (average_time[i]+3, mae[i]), fontsize=22)
    elif(i==1):
        plt.annotate(txt, (average_time[i]+5, mae[i] - 0.01), fontsize=22)
    else:
        plt.annotate(txt, (average_time[i], mae[i] + 0.01), fontsize=22)

# Trigger based with classification on the first 1
mae = [3.43, 3.20, 2.97, 2.69]
average_time = [49+26, 61+26, 74+26, 90+26]
text = ['t=20', 't=25', 't=30', 't=35']
# plt.plot(average_time, mae, color='blue', marker='*', markerfacecolor='blue', linewidth=2, linestyle='-.')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.01))

# Trigger based with classification on 2 consecutive 1s
mae = [2.34, 2.29, 2.17, 2.06]
average_time = [50+55, 63+55, 79+55, 94+55]
text = ['t=20', 't=25', 't=30', 't=35']
# plt.plot(average_time, mae, color='blue', marker='^', markerfacecolor='blue', linewidth=2, linestyle='-.')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.01))


# Trigger based with classification on 3 consecutive 1s
mae = [1.91, 1.86, 1.8, 1.75]
average_time = [51+83, 65+83, 80+83, 95+83]
text = [r'$\beta$=20', r'$\beta$=25', r'$\beta$=30', r'$\beta$=35']
# plt.plot(average_time, mae, color='blue', marker='s', markerfacecolor='blue', linewidth=2, linestyle='-.')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.01), fontsize=22)


# Waiting based with classification on the first 1
mae = [2.30, 2.11, 1.95, 1.88]
average_time = [60+26, 90+26, 120+26, 150+26]
text = ['w=60', 'w=90', 'w=120', 'w=150']
# plt.plot(average_time, mae, color='purple', marker='*', markerfacecolor='purple', linewidth=2, linestyle=':')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.01))


# Waiting based with classification on 2 consecutive 1s
mae = [1.98, 1.81, 1.75, 1.69]
average_time = [60+55, 90+55, 120+55, 150+55]
text = ['w=60', 'w=90', 'w=120', 'w=150']
# plt.plot(average_time, mae, color='purple', marker='^', markerfacecolor='purple', linewidth=2, linestyle=':')
# for i, txt in enumerate(text):
#     plt.annotate(txt, (average_time[i], mae[i] + 0.01))

# Waiting based with classification on 3 consecutive 1s
mae = [1.67, 1.61, 1.55, 1.51]
average_time = [60+83, 90+83, 120+83, 150+83]
text = [r'$\alpha$=60s', r'$\alpha$=90s', r'$\alpha$=120s', r'$\alpha$=150s']
plt.plot(average_time, mae, color='purple', marker='s', markerfacecolor='purple', linewidth=2, linestyle=':')
for i, txt in enumerate(text):
        if(i==0):
            plt.annotate(txt, (average_time[i]-22, mae[i]-0.01), fontsize=22)
        elif(i==3):
            plt.annotate(txt, (average_time[i]-29, mae[i]-0.005), fontsize=22)
        else:
            plt.annotate(txt, (average_time[i], mae[i] + 0.01), fontsize=22)




#
# # Convergence based with classification on 3 consecutive 1s
# mae = [1.68, 1.77, 1.80, 1.81]
# average_time = [66+83, 42+83, 34+83, 31+83]
# plt.plot(average_time, mae, color='red', marker='o', markerfacecolor='red', linewidth=2, linestyle='-.')
#
# # Trigger based with classification on 3 consecutive 1s
# mae = [1.91, 1.86, 1.8, 1.75]
# average_time = [51+83, 65+83, 80+83, 95+83]
# plt.plot(average_time, mae, color='blue', marker='^', markerfacecolor='blue', linewidth=2, linestyle='--')
#
# # Waiting based with classification on 3 consecutive 1s
# mae = [1.67, 1.61, 1.55, 1.51]
# average_time = [60+83, 90+83, 120+83, 150+83]
# plt.plot(average_time, mae, color='purple', marker='s', markerfacecolor='purple', linewidth=2, linestyle=':')



plt.legend(handlelength=3)
# plt.legend(handlelength=5)
# plt.legend(handlelength=5)
# plt.legend(handlelength=5)
# fig.tight_layout()
plt.show()
