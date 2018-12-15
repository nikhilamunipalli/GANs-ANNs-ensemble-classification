#----- CLASSIFICATION -------#

#initializing timer
import time
start_time = time.time()           

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

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test, y_train,y_test = train_test_split(X,y,test_size=0.2)

#over sampling
'''from imblearn.over_sampling import ADASYN as p 
sampler= p()
X_train,y_train = sampler.fit_sample(X_train, y_train)'''

#importing libraries
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dense

def create_baseline(): 
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = len(X_train[0]) , output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #internal layers
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'rmsprop' , metrics =['accuracy'],loss = 'binary_crossentropy' )
    return classifier

classifier = KerasClassifier(build_fn=create_baseline, epochs=1, batch_size=5, verbose=0)
#fitting the model
classifier.fit(X_train,y_train)

#predicting results
y_pred = classifier.predict(X_test)
y_pred = np.hstack(y_pred)

'''#classification
from sklearn.ensemble import RandomForestClassifier as D
classifier = D(n_estimators = 10)
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)'''          
            
#auc under roc
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(y_pred, y_test)
roc_auc = auc(fpr, tpr)

#stopping timer
time = (time.time() - start_time)
      
#tabulation
'''import csv
with open('results.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow([method,roc_auc,time])'''
    
#printing
print(time)
print(roc_auc)


#----- CONVERTING ARFF TO CSV -------#

'''import os
files =[file for file in os.listdir('.') if file.endswith('.arff')]
# Function for converting arff list to csv list
def toCsv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent

# Main loop for reading and writing files
for file in files:
    with open(file , "r") as inFile:
        content = inFile.readlines()
        name,ext = os.path.splitext(inFile.name)
        new = toCsv(content)
        with open(name+".csv", "w") as outFile:
            outFile.writelines(new)'''
            
#merging csv    
'''import os
csv_dir = os.getcwd()

dir_tree = os.walk(csv_dir)
for dirpath, dirnames, filenames in dir_tree:
   pass

csv_list = []
for file in filenames:
   if file.endswith('.csv'):
      csv_list.append(file)
      
fout=open("out.csv","a")
# first file:
for line in open(csv_list[0],'r+'):
    fout.write(line)
# now the rest:    
for num in csv_list[1:]:
    f = open(num)
    next(f) # skip the header
    for line in f:
         fout.write(line)
    f.close() # not really needed
fout.close()'''
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
