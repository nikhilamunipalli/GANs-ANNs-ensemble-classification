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

#deep learning

#importing libraries
   
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from keras.models import Model,Sequential
from keras.layers import Dense,Dropout
from keras import initializers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input
import keras.backend as K
import tensorflow as tf
import keras



input_size = 10
batch_size = 25
epoch_size = 10
sample_size = 20000

def create_generator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = input_size,output_dim=100,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    
    #internal layers
    classifier.add(Dense(output_dim =32))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dense(output_dim = 32))
    classifier.add(LeakyReLU(0.2))
    
    #output layer
    classifier.add(Dense(output_dim =len(one[0]), activation = 'tanh'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier

def create_discriminator(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = len(one[0]) , output_dim = 10,
                         kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.5))
    
    #internal layers
    classifier.add(Dense(output_dim =20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))
    classifier.add(Dense(output_dim = 20))
    classifier.add(LeakyReLU(0.2))
    classifier.add(Dropout(0.2))
    
    #output layer
    classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'adam',loss = 'binary_crossentropy' )
    return classifier  
  
def create_gan_network():
    
    discriminator.trainable = False
    gan_input = Input(shape=(input_size,))
    
    x = generator(gan_input)
    gan_output = discriminator(x)
    
    #model network gan    
    gan = Model(inputs=gan_input, outputs=gan_output)
    
    #compiling
    gan.compile(loss='binary_crossentropy',optimizer = keras.optimizers.Adam(lr=0.001,amsgrad=True))
    return gan

def train_gan():
    
   for epoch in range(1,epoch_size+1):
       j=0
       
       for i in range(batch_size,len(one),batch_size):
            
            #discriminator training
            X_curr = one[j:i,:]
            noise = np.random.normal(0, 1, size=[batch_size,input_size])
            generated_output = generator.predict(noise)
        
            X_dis = np.concatenate([X_curr, generated_output])
            y_dis = np.zeros(len(X_dis))
            y_dis[:batch_size]=0.9
            
            discriminator.trainable = True
            discriminator.train_on_batch(X_dis,y_dis)
            
            #generator training
            discriminator.trainable = False
            gan_input = np.random.normal(0, 1, size=[batch_size,input_size])
            gan_output = np.ones(batch_size)
            gan.train_on_batch(gan_input, gan_output)
            
            j=i      
            

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


#model building 
def build_model():
    
    estimators = []
    
    #neural networks
    classifier = KerasClassifier(build_fn=create_baseline, epochs= 5)
    model2 = BaggingClassifier(base_estimator = classifier, n_estimators = 1)
    estimators.append(('bagging',model2))
    
    #random forest
    model1 = RandomForestClassifier(n_estimators = 50)
    estimators.append(('rf',model1))
    
    # create the ensemble model
    model = VotingClassifier(estimators,voting = 'soft')
    
    return model

#k fold cross validation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

scores = []
kf = StratifiedKFold(n_splits = 10)

for train_index, test_index in kf.split(X,y):
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    one = []
    for i,v in enumerate(X_train):
        if y[i] == 1:
            one.append(v)
    
    one=np.vstack(one)
    
    generator = create_generator()
    discriminator = create_discriminator()
    gan = create_gan_network()
    
    train_gan()

    generated = generator.predict(np.random.normal(0, 1, size=[sample_size,input_size]))
    X_train = np.concatenate((X_train,generated))
    y_train = np.concatenate((y_train,np.ones(sample_size)))
    
    model = build_model()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    scores.append(roc_auc_score(y_test,y_pred))
