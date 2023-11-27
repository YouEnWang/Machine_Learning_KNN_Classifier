# import modules
import pandas as pd
import numpy as np
import seaborn as sns
import csv

# observing dataset
def observing(data):
    df = data
    df.head()
    df.describe().T
    sns.scatterplot(x=df['Sepal length'], y=df['Sepal width'], hue=df['Label'], style=df['Label'])
    sns.scatterplot(x=df['Sepal length'], y=df['Petal length'], hue=df['Label'], style=df['Label'])
    sns.scatterplot(x=df['Sepal length'], y=df['Petal width'], hue=df['Label'], style=df['Label'])
    sns.scatterplot(x=df['Sepal width'], y=df['Petal length'], hue=df['Label'], style=df['Label'])
    sns.scatterplot(x=df['Sepal width'], y=df['Petal width'], hue=df['Label'], style=df['Label'])
    sns.scatterplot(x=df['Petal length'], y=df['Petal width'], hue=df['Label'], style=df['Label'])

# KNN alogorithm
def KNN(initial_data, K, feature_list, change_data):
    '''split data'''
    split_data = np.split(initial_data, 6)
    if change_data:
        training_data = np.vstack((split_data[0], split_data[2], split_data[4]))
        test_data = np.vstack((split_data[1], split_data[3], split_data[5]))
    else:
        test_data = np.vstack((split_data[0], split_data[2], split_data[4]))
        training_data = np.vstack((split_data[1], split_data[3], split_data[5]))

    # 存儲最後預測所得的label
    predict = []

    '''get feature'''
    SL = feature_list[0]
    SW = feature_list[1]
    PL = feature_list[2]
    PW = feature_list[3]

    '''training'''
    for iris_test in test_data:
        # initialization
        dis = 0
        total_dis = []
        counter = 0                         # 等於training data的index，並存儲在total_dis第一個row

        # calculate distance
        for iris_train in training_data:
            if SL:
                dis_SL = round((iris_test[0] - iris_train[0]), 2) ** 2
            else:
                dis_SL = 0

            if SW:
                dis_SW = round((iris_test[1] - iris_train[1]), 2) ** 2
            else:
                dis_SW = 0
            
            if PL:
                dis_PL = round((iris_test[2] - iris_train[2]), 2) ** 2
            else:
                dis_PL = 0
            
            if PW:
                dis_PW = round((iris_test[3] - iris_train[3]), 2) ** 2
            else:
                dis_PW = 0

            dis = np.sqrt(dis_SL + dis_SW + dis_PL + dis_PW)
            total_dis.append([counter, dis])
            counter += 1

        # compare distance
        sorted_dis = sorted(total_dis, key=lambda x: x[1])
        nearest = []
        for i in range(K):
            # 搜尋前K個NN之training data的label
            neighbor = training_data[sorted_dis[i][0]][4]
            nearest.append(neighbor)

        # 求眾數
        counts = np.bincount(nearest)
        predict.append(np.argmax(counts))
        
    return predict

# 計算分類率
def classification_rate(initial_data, predict, change_data):
    '''split data'''
    split_data = np.split(initial_data, 6)
    if change_data:
        test_data = np.vstack((split_data[1], split_data[3], split_data[5]))
    else:
        test_data = np.vstack((split_data[0], split_data[2], split_data[4]))
    
    # 預測正確的資料總數
    True_prediction = 0

    # 將predict的label與test data的label做比對
    for i in range(len(predict)):
        if predict[i] == test_data[i][4]:
            True_prediction += 1
    
    # 分類率
    CR = round(True_prediction / len(test_data), 5) * 100
    return CR

# 將結果輸出為csv
def output_result(initial_data, K, feature_all):
    for i in range(len(feature_all)):
        change_data = 0                             # 前一半資料為test data
        predict0 = KNN(initial_data, K, feature_all[i], change_data)
        CR0 = classification_rate(initial_data, predict0, change_data)

        change_data = 1                             # 前一半資料為training data
        predict1 = KNN(initial_data, K, feature_all[i], change_data)
        CR1 = classification_rate(initial_data, predict1, change_data)

        Average_CR = round((CR0 + CR1)/2, 2)       # Average CR

        # 輸出至file
        with open(f"HW1_iris_K為{K}.csv", "a", newline="") as file:
            file.write(f"第{i+1}個feature\n")
            file.write("[SL SW PL PW] = ")
            for j in feature_all[i]:
                file.write(f"{j} ")
            file.write("\n")
            file.write("training data為後一半資料之predict result\n")
            for j in predict0:
                file.write(f"{j} ")
            file.write(f"\nCR = {CR0} %\n")
            file.write("training data為前一半資料之predict result\n")
            for j in predict1:
                file.write(f"{j} ")
            file.write(f"\nCR = {CR1} %\n")
            file.write(f"Average CR = {Average_CR} %\n\n")


# main function
def main():
    # import dataset
    df = pd.read_table('iris.txt', sep='\s+', header=None)
    df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Label']
    initial_data = np.loadtxt('iris.txt', dtype=float)

    # # 觀察資料並生成scatter plot
    # observing(df)

    # 15種特徵組合[SL, SW, PL, PW]如下
    feature_all = [[1, 1, 1, 1],
                   [1, 1, 1, 0], 
                   [1, 1, 0, 1], 
                   [1, 1, 0, 0], 
                   [1, 0, 1, 1], 
                   [1, 0, 1, 0], 
                   [1, 0, 0, 1], 
                   [1, 0, 0, 0], 
                   [0, 1, 1, 1], 
                   [0, 1, 1, 0], 
                   [0, 1, 0, 1], 
                   [0, 1, 0, 0], 
                   [0, 0, 1, 1], 
                   [0, 0, 1, 0], 
                   [0, 0, 0, 1]]
    
    # 設定K值 -> 設定feature -> 選定前半or後半data並計算CR -> 計算Average CR
    K = 3
    output_result(initial_data, K, feature_all)

if __name__ == "__main__":
    main()