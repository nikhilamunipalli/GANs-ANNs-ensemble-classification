'''from pylab import bone,pcolor,colorbar,plot,show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o','s']
colors = ['r','g']
for i,x in enumerate(X_train):
    print('marking')
    print(i)
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth =2 )
show()'''

#tabulation
'''import csv
with open('results.csv', 'a') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerow([method,roc_auc,time])'''

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
            