
'''Hyperparameter Optimization
'''


import numpy

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
import sys
#Specification of event log
log = sys.argv[1]
timestepsFrom = int(sys.argv[2])
timestepsTo = int(sys.argv[3])
num_classes = 2
batch_size = 128



for i in range(timestepsFrom, timestepsTo):
    num_timesteps = i
    #import dataset
    dataset = numpy.loadtxt("Data/"+log+"/"+log+"_transformedNumberOfEvents_"+str(num_timesteps)+".csv", delimiter=",")
    X_train = dataset[:, :-1]
    Y_train = dataset[:, -1]
    if sys.argv[1] != "ProductionLog":
        Y_train[Y_train == 1] = 0
        Y_train[Y_train == 2] = 1
        Y_train[Y_train == 3] = 1

    num_features = int((dataset.shape[1]-1)/num_timesteps)

    if __name__ == '__main__':

            y_train = Y_train
            scoring = {'AUC': 'roc_auc',
                       'Accuracy': 'accuracy',
                       'F1': 'f1',
                       'Precision': 'precision',
                       'Recall': 'recall'}


            #Fixed random seed for reproducibility
            seed = 7
            numpy.random.seed(seed)

            # define grid search parameters
            param_dist = {'n_estimators': [200, 600, 1000, 1400, 1800],
                          "max_depth": [3, None],
                          "max_features": sp_randint(1, num_features),
                          "min_samples_split": sp_randint(2, 11),
                          "min_samples_leaf": sp_randint(1, 11),
                          "bootstrap": [True, False],
                          "criterion": ["gini", "entropy"]}
            #create model

            clf = RandomForestClassifier(n_estimators=20)

            randomizedSearch = RandomizedSearchCV(estimator=clf, param_distributions=param_dist, scoring=scoring, cv=10, n_jobs=-1, refit='F1', n_iter=20)
            grid_search_result = randomizedSearch.fit(X_train, y_train)

            #output
            print("Best: %f using %s" % (grid_search_result.best_score_, grid_search_result.best_params_))


            metrics = numpy.empty(5)
            metrics[0] = grid_search_result.cv_results_['mean_test_AUC'][0]
            metrics[1] = grid_search_result.cv_results_['mean_test_Accuracy'][0]
            metrics[2] = grid_search_result.cv_results_['mean_test_F1'][0]
            metrics[3] = grid_search_result.cv_results_['mean_test_Precision'][0]
            metrics[4] = grid_search_result.cv_results_['mean_test_Recall'][0]



            numpy.savetxt("Results/RF/"+log+"_"+str(num_timesteps)+".csv", numpy.atleast_2d(metrics), delimiter = ',', fmt='%6f',header = "AUC, Accuracy, F1, Precision, Recall")
            text_file = open("Results/RF/Hyperparameters"+log+"_"+str(num_timesteps)+".txt", "w")
            text_file.write(str(grid_search_result.best_params_))
            text_file.close()