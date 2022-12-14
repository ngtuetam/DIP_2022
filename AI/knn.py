import math
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

def predict(train_data_orange, train_data_strawberry, test_data, K_test):
    distance_array_orange = np.zeros((8))
    distance_array_strawberry = np.zeros((8))
    
    for row in range (8):
        for col in range (7):
            distance_array_orange[row] += (train_data_strawberry[row][col] - test_data[col])**2
        distance_array_orange[row] = math.sqrt(distance_array_orange[row])

    for row in range (8):
        for col in range (7):
            distance_array_strawberry[row] += (train_data_orange[row][col] - test_data[col])**2
        distance_array_strawberry[row] = math.sqrt(distance_array_strawberry[row])
        
    distance_array_orange = np.sort(distance_array_orange,kind='heapsort') 
    distance_array_strawberry = np.sort(distance_array_strawberry,kind='heapsort') 
    
    orange_count = 0
    strawberry_count = 0
    while (orange_count + strawberry_count) <= K_test:
        if (distance_array_orange[orange_count] > distance_array_strawberry[strawberry_count]):
            orange_count += 1
        else:
            strawberry_count += 1
            
    if orange_count > strawberry_count:
        return "Orange"
    else:
        return "Strawberry"

def K_fold():
    #Kfold
    input_array_test = pd.read_csv("ANN_Data_Kfold.csv").to_numpy()
    kf = KFold(n_splits=5, shuffle= True)
    kf.get_n_splits(input_array_test)
    for train_index, test_index in kf.split(input_array_test):
        print("TRAIN:", train_index, "TEST:", test_index)
        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        for i in range(4):
            y_pred = predict(train_data_orange, train_data_strawberry,input_array_test[test_index][i],3)
            print(y_pred)
            print(test_index)
            if test_index[i] < 10: #Actual
                if y_pred == "Orange":
                    TP += 1
                else:
                    FN += 1
            if test_index[i] >= 10:
                if y_pred == "Strawberry":
                    TN += 1
                else:
                    FP += 1
        print("RC = ", TP/(TP + FN))
        print("PR = ", TP/(TP + FP))
        print("ACC = ", (TP + TN)/(TP + TN + FP + FN))

        print("ACC = ", (TP + TN)/(TP + TN + FP + FN))
    
if __name__ == "__main__":
    train_data_orange  = pd.read_csv("orange_train_data.csv").to_numpy()
    train_data_strawberry = pd.read_csv("strawberry_train_data.csv").to_numpy()
    test_data_orange   = pd.read_csv("orange_test_data.csv").to_numpy()
    test_data_strawberry  = pd.read_csv("strawberry_train_data.csv").to_numpy()
    for i in range(2):
        print(predict(train_data_orange,train_data_strawberry,test_data_orange[i], 3))
    for i in range(2):
        print(predict(train_data_orange,train_data_strawberry,test_data_strawberry[i], 3))
    K_fold()