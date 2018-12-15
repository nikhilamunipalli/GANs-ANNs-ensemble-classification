#importing libraries
import numpy as np
import pandas as pd

#importing dataset
dataset = pd.read_csv('out.csv') 
dataset = dataset.replace('?',np.NaN)         
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values           
 
#missing data handling
from sklearn.preprocessing import Imputer
imputer = Imputer()
X = imputer.fit_transform(X)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X) 

X_train = X
y_train = y


#importing libraries
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense,Dropout
import keras.backend as K
import tensorflow as tf
import keras

def tfauc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc


def create_baseline(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = len(X_train[0]) , output_dim = 100, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.5))
    
    #internal layers
    classifier.add(Dense(output_dim =256, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 256, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 100, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 30, init = 'uniform', activation = 'relu'))
    classifier.add(Dropout(0.2))
    
    #output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = keras.optimizers.Adam(lr=0.001,amsgrad=True),
                       metrics =[tfauc],loss = 'binary_crossentropy' )
    return classifier

classifier = KerasClassifier(build_fn=create_baseline, epochs= 200)

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier

estimators = []

model1 = RandomForestClassifier(n_estimators = 50
                                )
estimators.append(('rf',model1))
model2 = BaggingClassifier(base_estimator = classifier, n_estimators = 20)
estimators.append(('bagging',model2))

# create the ensemble model
model = VotingClassifier(estimators,voting = 'soft')

#kforld cross validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train ,y_train, cv=10, scoring='roc_auc')
scores.mean()




            
            
            
            
            
            
            
            
            
            
            
            
            
            
