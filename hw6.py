from sklearn.linear_model import SGDClassifier, LogisticRegression 
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from matplotlib.pyplot import cm
#Author: YongBaek Cho
# Date : 11/01/2018
#This program read the matlab format file which consists of 70,000 digitized handwritten digits and classify them using SGDClassifier and LogisticRegression
def get_data():
    '''
    This funciton takes no arguments
    Read the mat file
    '''

    mnist = sio.loadmat('mnist-original.mat')
    
    return np.transpose(mnist['data']), mnist['label'][0]

def get_train_and_test_sets(X,y):
    '''
    This function takes X and y from previous function.
    The first 60,000 instances for training , the rest for testing
    This function breaks into separate traning and testing X and y_test
    '''
    
    index = np.random.permutation(len(X[:60000]))

    X_train = X[index[:60000]]
    X_test = [X[i] for i in range(len(X)) if i >= 60000 ]

    y_train = y[index[:60000]]
    y_test = [y[i] for i in range(len(y)) if i >= 60000 ]
   
    return X_train, X_test, y_train, y_test

def train_to_data(X_Training, Y_Training, model_name):
    '''
    This function takes a traing X and y and model name
    If its on SGDClassifier, maximum of 50 iterations and tolerance of 
    .001 otherwise, make LogisticRegression
    '''
    if model_name in 'SGDClassifier':
        model = SGDClassifier(max_iter = 50, tol = 0.001)
        model = model.fit(X_Training, Y_Training)
    else:
        model = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
        model = model.fit(X_Training, Y_Training)

    return model

def get_confusion_matrix(model, X_Training, Y_Training):
    '''
    This function takes a model, X_Training and Y_Training
    pass the results of 5-fold cross validation predicting on the training data
    to the confucion matrix
    '''

    cross_val = cross_val_predict(model, X_Training, Y_Training, cv = 5)

    return confusion_matrix(Y_Training, cross_val)

def probability_matrix(confusion_matrix):
    '''
    This function takes probability matrix as parameter.
    With parameter, it will figure out the probabilities, then return the probability matrix.
    '''

    prob_matrix = np.zeros((10,10))
    
    for i in range(len(confusion_matrix)):
        tot = sum(confusion_matrix[i])
        for j in range(len(confusion_matrix[0])):
            prob_matrix[i][j] = round(confusion_matrix[i][j] / tot, 3)

    return prob_matrix
    

def plot_probobality_matrices(prob_matrix1, prob_matrix2):
    '''
    This function takes 2 probability matrices
    and display the heatmap
    '''
    fig, ax = plt.subplots(ncols=2)
    im1 = ax[0].imshow(prob_matrix1, cmap = cm.Greys)
    ax[0].xaxis.tick_top()
    ax[0].set_title('SGDClassifier', pad = 30)
    im2 = ax[1].imshow(prob_matrix2, cmap = cm.Greys)
    ax[1].xaxis.tick_top()
    ax[1].set_title('SoftmaxRegression', pad = 30)
    



def main():
    '''
    Get the data split into training and testing, train SGDClassifier and a LogisticRegression
    '''

    X, y = get_data()

    X_Training, testing_X, Y_Training, testing_y = get_train_and_test_sets(X,y)
    SGDClassifier = train_to_data(X_Training, Y_Training, 'SGD')
    LogisticRegression = train_to_data(X_Training, Y_Training, 'Logistic')
    confusion_matrix_SGD = get_confusion_matrix(SGDClassifier, X_Training, Y_Training)
    confusion_matrix_Logistic = get_confusion_matrix(LogisticRegression, X_Training, Y_Training)
    prob_matix_SGD = probability_matrix(confusion_matrix_SGD)
    prob_matrix_Logistic = probability_matrix(confusion_matrix_Logistic)
    plot_probobality_matrices(prob_matix_SGD, prob_matrix_Logistic)
    plt.figure()
    for mod in (('SGDClassifier:', probability_matrix(confusion_matrix_SGD)), ('Softmax:', probability_matrix(confusion_matrix_Logistic))):
        print(*mod, sep = '\n')
    plt.show()

if __name__ == "__main__":
    main()