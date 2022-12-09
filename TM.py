import numpy as np
import math
import random
from matplotlib import pyplot as plt
from dtw import *


def epsilon_calculation(window_length, c_0, extended):
    value_array = [[0 for i in range(window_length-c_0+1)] for j in range(window_length+1)]
    for k in range(1, window_length+1):
        value_array[k][1] = 2/k
    for k in range(c_0+2, window_length+1):
        for m in range(2, window_length-c_0+1):
            sum_ = 0
            for l in range(1, m+1):
                prod_1 = 1
                prod_2 = 1
                for i in range(1, l+2):
                    prod_1 *= (k-i)/i
                for i in range(1, l):
                    prod_2 *= value_array[k-i][m-i]
                sum_ += pow(-1/c_0, l)*prod_1*prod_2
            value_array[k][m] = m/(1-sum_)
    p_0 = 1-value_array[window_length][window_length-c_0]
    sum_ = 0
    for l in range(1, window_length-c_0+1):
        prod_1 = 1
        prod_2 = 1
        for i in range(0, l):
            prod_1 *= value_array[window_length-i][window_length-c_0-i]
        for i in range(1, l+1):
            prod_2 *= (window_length-1-i)/i
        sum_ += pow(-1/c_0, l)*prod_1*prod_2
    p_1 = value_array[window_length][window_length-c_0]+sum_
    sum_ = 0
    for l in range(1, window_length-c_0+1):
        prod_1 = 1
        prod_2 = 1
        for i in range(0, l):
            prod_1 *= value_array[window_length-i][window_length-c_0-i]
        for i in range(1, l):
            prod_2 *= (1-i)/(i+1)*l
        sum_ += pow(-1/c_0, l)*prod_1*prod_2
    p_k_ = -sum_
    if extended == 1:
        return np.log(p_k_/p_1), p_0, p_1
    return 2*max(np.log(p_0/p_1), np.log(p_k_/p_1)), p_0, p_1



def c_calculation(epsilon, k):
    p_0 = None
    p_1 = None
    l = 2
    r = k-1
    extended = 0
    while l<r:
        c_0 = math.floor((l+r)/2)
        epsilon_1, _, _ = epsilon_calculation(k, c_0, extended)
        # print("epsilon1", epsilon_1, c_0)
        if epsilon_1<epsilon:
            l=c_0
        else:
            epsilon_2, _, _ = epsilon_calculation(k, c_0-1, extended)
            # print("epsilon2", epsilon_2, c_0)
            if epsilon_2>epsilon_1:
                l=c_0
            else:
                r=c_0
        if l+1==r:
            epsilon_3, _, _ = epsilon_calculation(k, l, extended)
            epsilon_4, _, _ = epsilon_calculation(k, r, extended)
            # print(epsilon_3, epsilon_4)
            if epsilon_3<=epsilon:
                c_opt = l
                break
            elif epsilon_4<=epsilon:
                c_opt = r
                break
            else:
                # print("here----------------")
                # print("l, r", l,r)
                extended = 1
                l = r
                r = k-1
                while l<r:
                    c_0 = math.floor((l+r)/2)
                    epsilon_5, p_0, p_1 = epsilon_calculation(k, c_0, extended)
                    # print(epsilon_5, c_0)
                    if epsilon_5<=epsilon:
                        r = c_0
                    else:
                        l = c_0
                    if l+1 == r:
                        c_opt = r
                        return c_opt, extended, p_0, p_1
    return c_opt, extended, p_0, p_1

def threshold_mechanism(data, k, epsilon, c_0, extended, p_0, p_1):
    index = [None for i in range(len(data)+k)]
    for item_index in range(len(data)):
        empty_list = [ i for i in range(item_index, item_index+k) if index[i]==None]
        # print(empty_list)
        c = len(empty_list)
        if c>c_0:
            noise = random.choice(empty_list)
            index[noise] = item_index
        else:
            if extended == 0:
                if index[item_index]==None:
                    index[item_index] = item_index
                else:
                    noise = random.choice(empty_list)
                    index[noise] = item_index
            elif extended == 1:
                if index[item_index]==None:
                    value = np.exp(epsilon/2)*p_1/p_0
                    sample = np.random.rand()
                    if sample<value:
                        index[item_index] = item_index
                else:
                    noise = random.choice(empty_list)
                    index[noise] = item_index
    return index


# if __name__ == "__main__":
#     window_length = [_ for _ in range(10, 210, 10)]
#     epsilon = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16]
#     # window_length = [100]
#     # epsilon = [4]
#     file = open("./epsilon_window_c_.txt", 'w')
#     for k in window_length:
#         for e in epsilon:
#             c_opt, extended, p_0, p_1 = c_calculation(e, k)
#             print(e, k, c_opt, extended, p_0, p_1)
#             file.write(str(e)+","+str(k)+","+str(c_opt)+","+str(extended)+","+str(p_0)+","+str(p_1)+"\n")
#     file.close()

def TM(data, window_length, epsilon, c_opt, extended, p_0, p_1):
    threshold_index = threshold_mechanism(data, window_length, epsilon, c_opt, extended, p_0, p_1)
    new_data = []
    for i in range(len(threshold_index)):
        if threshold_index[i]:
            new_data.append(data[threshold_index[i]])
        elif i>len(data):
            new_data.append(data[len(data)-1])
    return new_data[:len(data)]

def simple_moving_average(data, window):
    moving = []
    for i in range(len(data)-window+1):
        values = data[i:i+window]
        moving.append(round(sum(values)/window, 2))
    return np.array(moving)
        
        

if __name__ == "__main__":
    data_file = open('ibm.txt')
    line = data_file.readline()
    data = list(map(lambda x: int(x), filter(lambda x: x!='', line.strip().split(' '))))
    while line:
        line = data_file.readline()
        data.extend(list(map(lambda x:int(x), filter(lambda x: x!= '', line.strip().split(' ')))))
    min_value = min(data)
    max_value = max(data)
    data = [(i-min_value)/max_value for i in data]
    plt.plot(data)
    plt.savefig('data.png',dpi=600)
    plt.cla()
    epsilons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # epsilons = [4]
    window_length = [i for i in range(10, 110, 10)]
    window_length = [20]
    moving_window = 20
    moving_ori = simple_moving_average(data, moving_window)
    flag_10 = True
    flag_100 = True
    flag_draw = True
    for k in window_length:
        result = []
        for epsilon in epsilons:
            file_para = open('epsilon_window_c_.txt')
            lines = file_para.readlines()
            lines = [list(line.split(',')) for line in lines]
            p_0 = None
            p_1 = None
            for line in lines:
                if int(line[0])==epsilon and int(line[1])==k:
                    c_opt = int(line[2])
                    extended = int(line[3])
                    if extended == 1:
                        p_0 = float(line[4])
                        p_1 = float(line[5])
            mse_ = []
            for iter in range(500):
                threshold_data = TM(data, k, epsilon, c_opt, extended, p_0, p_1)
                threshold_moving = simple_moving_average(threshold_data, moving_window)
                if flag_draw and epsilon==4:
                    plt.plot(threshold_data, label="TM")
                    plt.plot(data, label="Original Data")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Stock Price")
                    plt.savefig("epsilon4k20.png", dpi=600)
                    flag_draw = False
                    plt.cla()
                if flag_10 and epsilon==1:
                    plt.plot(threshold_moving, label='TM')
                    plt.plot(moving_ori, label="ground truth")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Simple Moving Average Value")
                    plt.savefig("epsilon1.png", dpi=600)
                    flag_10 = False
                    plt.cla()
                if flag_100 and epsilon==10:
                    plt.plot(threshold_moving, label='TM')
                    plt.plot(moving_ori, label="ground truth")
                    plt.legend()
                    plt.xlabel("Time")
                    plt.ylabel("Simple Moving Average Value")
                    plt.savefig("epsilon10.png", dpi=600)
                    flag_100 = False
                    plt.cla()
                # mse = sum([(threshold_moving[i]-moving_ori[i])**2 for i in range(len(threshold_moving))])/len(threshold_moving)
                mse = dtw(threshold_moving, moving_ori, dist_method='euclidean', keep_internals=True).distance
                mse_.append(mse)
            result.append(np.mean(mse_))
        plt.plot(epsilons, result, 'o'+'-')
        plt.xlabel("Input Privacy Budget $\epsilon$")
        plt.ylabel("DTW Distance")
        plt.savefig('epsilon_varying.png', dpi=600)
            

