import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

##Load data and build the training set
breast_data = pd.read_csv("datasets/breast-cancer-wisconsin.data", delimiter=',', na_values='?', header=None)
breast_data = np.array(breast_data)

##Build the training set
training_set = np.delete(breast_data, 0, axis=1)
training_set = training_set[:233]

##Build the testing set
#Vector of the class of patients from the testing set
testing_set_class = np.delete(breast_data, 0, axis=1)
testing_set_class = testing_set_class[:,9]

#Testing set without the class of patients
testing_set = np.delete(breast_data, [0,10], axis=1)
testing_set = testing_set[234:]

##Parameters
#Number of columns for training (including class feature)
n = training_set.shape[1] 
#Number of rows of training set
m = training_set.shape[0] 
#Number of rows of testing set
l = testing_set.shape[0] 


def euclidian_distance(index_training_point, index_testing_point):
    dis = 0
    #Compute the euclidian distance excluding the class feature of the training set
    for i in range(n-1):
        dis = dis + (training_set[index_training_point][i] - testing_set[index_testing_point][i])**2
    return(np.sqrt(dis))

#Compute the error between the testing class and the predicted class by the k-NN algorithm
def compute_errors(Class_vector_predicted):
    errors = 0
    for i in range(l):
        if(Class_vector_predicted[i] != testing_set_class[i]):
            errors += 1
    precision = (errors/l)*100
    return(precision)

def project(testing_set, k):
    #Vector that contains the predicted classes of the points in the testing set
    Class_vector = np.zeros(l)
    
    for i in range(l):
        X_dis = np.zeros((m,2))
        #Store the index and the euclidian distance between the of every points of the training set
        for j in range(m):
            X_dis[j][0] = j
            X_dis[j][1] = euclidian_distance(j,i)
        #Sort to find the k nearest neighbors (k-NN)
        X_dis_sorted = X_dis[X_dis[:,1].argsort()]
        K_NN = X_dis_sorted[:k]
        
        #Selection of the class based on the k-NN
        benign = 0
        malign = 0
        for p in range(k):
            selected_point_index = np.int(K_NN[p][0])
            if(training_set[selected_point_index][9] == 2):
                benign += 1
            else:
                malign += 1

        if(benign < malign):
            Class_vector[i] = 2
        else:
            Class_vector[i] = 4
    
    error = compute_errors(Class_vector)
    print("The k-NN algorithm gave "+ str(error) + "% of class prediction error with k="+ str(k) )
    return(Class_vector)

print(project(testing_set, 10))


        


