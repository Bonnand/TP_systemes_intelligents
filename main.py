import sklearn
from sklearn.neural_network import MLPClassifier
import numpy as np

# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ''' Etape 1 '''
    NUMBER=10

    ''' Etape 2 '''
    x = []
    for i in range(NUMBER) :
        for j in range(NUMBER) :
            x.append([i, j])

    #x = np.array(x)

    # Affichage des couples
    print("Affichage des couples")
    print(x)
    print("")


    ''' Etape 3 '''
    y =[]

    for index in range(NUMBER**2) :
            if(( (x[index][0]*x[index][1]) )%2==0):
                y.append(1)
            else :
                y.append(0)

    # Affichage des parités
    print("Affichage des parités")
    print(y)
    print("")

    dataset_entrainement=[]
    dataset_test = []

    ''' Etape 4 '''
    for i in range(NUMBER**2) :

            if(x[i][1]< int(NUMBER*0.80) ):
                dataset_entrainement.append( [x[i],y[i]] )

            elif( x[i][1] >= int(NUMBER*0.80)):
                dataset_test.append( [x[i],y[i]] )

    # Affichage des datasets
    print("Affichage des datasets")
    print(dataset_entrainement)
    print(dataset_test)
    print("")

    ''' Etape 5 '''
    model = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000, random_state=42)

    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
