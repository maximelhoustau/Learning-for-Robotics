import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

def euclidian_distance(index_training_point, index_testing_point, training_set, testing_set):
    dis = 0
    #Compute the euclidian distance excluding the class feature of the training set
    for i in range(training_set.shape[1]-1):
        dis = dis + (training_set[index_training_point][i] - testing_set[index_testing_point][i])**2
    return(np.sqrt(dis))

#Compute the error between the testing class and the predicted class by the k-NN algorithm
def compute_errors(Class_vector_predicted, testing_set_class):
    errors_benign = 0
    errors_malign = 0
    predicted_benign = 0
    predicted_malign = 0
    l = testing_set_class.shape[0]
    for i in range(l):
        if(Class_vector_predicted[i] != testing_set_class[i]):
                if(Class_vector_predicted[i] == 0):
                    errors_malign += 1
                else:
                    errors_benign += 1
        else:
                if(Class_vector_predicted[i] == 0):
                    predicted_benign += 1
                else:
                    predicted_malign += 1

    print("\n CONFUSION MATRIX: ")
    print("\t\t\t ACTUAL CLASS")
    print("\t\t Benign \t\tMalign")
    print("PREDI \t Benign  " + str(predicted_benign) + "\t\t\t" + str(errors_malign))
    print("CTION \t Malign  " + str(errors_benign) + "\t\t\t" + str(predicted_malign))

    accuracy = (predicted_malign+predicted_benign)/l*100
    return(accuracy)

def project(dataset_path, k):
    ##Load data and build the training set
    dataset = pd.read_csv(dataset_path, delimiter=',', na_values='?', header=None)
    dataset = np.array(dataset)
    np.random.shuffle(dataset)
    
    ##Build the training set
    training_size = int(dataset.shape[0]*0.8)
     
    if(dataset_path == breast):
        training_set = np.delete(dataset, 0, axis=1)
        testing_set = np.delete(dataset, [0, dataset.shape[1]-1], axis=1)
    else:
        training_set = dataset
        testing_set = np.delete(dataset, dataset.shape[1]-1, axis=1)
    
    training_set = training_set[:training_size]
    
    ##Build the testing set
    #Testing set without the class of patients
    testing_set = testing_set[training_size+1:]

    #Vector of the class of patients from the testing set
    testing_set_class = np.delete(dataset, 0, axis=1)
    testing_set_class = testing_set_class[training_size+1:]
    testing_set_class = testing_set_class[:,dataset.shape[1]-2]

    ##Parameters 
    #Number of rows of training set
    m = training_set.shape[0] 
    l = testing_set.shape[0] 
     
    if(dataset_path == breast):
        print("\nWorking on Breast Cancer Dataset")
    elif(dataset_path == haberman): 
        testing_set_class = 2*testing_set_class
        training_set[:,training_set.shape[1]-1] = 2*training_set[:,training_set.shape[1]-1]
        print("\nWorking on Haberman Dataset")
        
    #Harmonisation of the test_set_class vector
    for i in range(l):
        if(testing_set_class[i] == 2):
            testing_set_class[i] = 0
        else:
            testing_set_class[i] = 1

    #Harmonisation of training set class feature
    for i in range(m):
        if(training_set[i][training_set.shape[1]-1] == 2):
            training_set[i][training_set.shape[1]-1] = 0
        else:
            training_set[i][training_set.shape[1]-1] = 1

   #Vector that contains the predicted classes of the points in the testing set
    Class_vector = np.zeros(l)

    for i in range(l):
        X_dis = np.zeros((m,2))
        #Store the index and the euclidian distance between the of every points of the training set
        for j in range(m):
            X_dis[j][0] = j
            X_dis[j][1] = euclidian_distance(j,i,training_set, testing_set)
        #Sort to find the k nearest neighbors (k-NN)
        X_dis_sorted = X_dis[X_dis[:,1].argsort()]
        K_NN = X_dis_sorted[:k]
        #Selection of the class based on the k-NN
        benign = 0
        malign = 0
        for p in range(k):
            selected_point_index = np.int(K_NN[p][0])
            if(training_set[selected_point_index][training_set.shape[1]-1] == 0):
                benign += 1
            else:
                malign += 1
        #Keep conventions of the dataset: 2: benign, 4: malign
        if(benign < malign):
            Class_vector[i] = 1
        else:
            Class_vector[i] = 0

    error = compute_errors(Class_vector, testing_set_class)
    print("\nThe k-NN algorithm gave "+ str(error) + "% of accuracy with k="+ str(k) )

    fig = plt.figure()
    fig.suptitle(dataset_path + ", k =  " +str(k))

    if(dataset_path == breast):
        AxisNames = wisconsinAxisNames
    else:
        AxisNames = habermanAxisNames
    
    ax = fig.add_subplot(321, projection='3d')
    ax.scatter(testing_set[:,0], testing_set[:,1], testing_set[:,2], c=getColors(testing_set_class))
    ax.set_xlabel(AxisNames[0])
    ax.set_ylabel(AxisNames[1])
    ax.set_zlabel(AxisNames[2])
    ax.set_title("Testing set initial")
 
    ax = fig.add_subplot(322, projection='3d')
    ax.scatter(testing_set[:,0], testing_set[:,1], testing_set[:,2], c=getColors(Class_vector))
    ax.set_xlabel(AxisNames[0])
    ax.set_ylabel(AxisNames[1])
    ax.set_zlabel(AxisNames[2]) 
    ax.set_title("Prediction by the k-nn algorithm")

    if(dataset_path == breast):

        ax = fig.add_subplot(323, projection='3d')
        ax.scatter(testing_set[:,3], testing_set[:,4], testing_set[:,5], c=getColors(testing_set_class))
        ax.set_xlabel(wisconsinAxisNames[3])
        ax.set_ylabel(wisconsinAxisNames[4])
        ax.set_zlabel(wisconsinAxisNames[5])
     
        ax = fig.add_subplot(324, projection='3d')
        ax.scatter(testing_set[:,3], testing_set[:,4], testing_set[:,5], c=getColors(Class_vector))
        ax.set_xlabel(wisconsinAxisNames[3])
        ax.set_ylabel(wisconsinAxisNames[4])
        ax.set_zlabel(wisconsinAxisNames[5])  
   
        ax = fig.add_subplot(325, projection='3d')
        ax.scatter(testing_set[:,6], testing_set[:,7], testing_set[:,8], c=getColors(testing_set_class))
        ax.set_xlabel(wisconsinAxisNames[6])
        ax.set_ylabel(wisconsinAxisNames[7])
        ax.set_zlabel(wisconsinAxisNames[8])
     
        ax = fig.add_subplot(326, projection='3d')
        ax.scatter(testing_set[:,6], testing_set[:,7], testing_set[:,8], c=getColors(Class_vector))
        ax.set_xlabel(wisconsinAxisNames[6])
        ax.set_ylabel(wisconsinAxisNames[7])
        ax.set_zlabel(wisconsinAxisNames[8])  
   

    plt.show()

    return(Class_vector)


def getColors(class_vector):
    c = []
    for i in range(len(class_vector)):
        if (class_vector[i] == 0 or class_vector[i] == 2):
            c.append('b')
        if(class_vector[i] == 1 or class_vector[i] == 4):
            c.append('r')

    return(c)


breast = "datasets/breast-cancer-wisconsin.data"
haberman = "datasets/haberman.data"
habermanAxisNames = ["Age", "YearOperation", "AuxilliaryNodes", "Class"]
wisconsinAxisNames = ["SampleNumber", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]

project(breast, 3)
project(breast, 6)
project(haberman, 3)
project(haberman, 6)


        


