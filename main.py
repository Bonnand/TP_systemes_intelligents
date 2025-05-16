import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


if __name__ == '__main__':

    ''' Etape 1 '''
    NUMBER=100

    ''' Etape 2 '''
    x = []

    # Création de la matrice de couples
    for i in range(NUMBER) :
        for j in range(NUMBER) :
            x.append([i, j])

    # Affichage des couples
    print("Affichage des couples")
    print(x)
    print("")


    ''' Etape 3 '''
    y =[]

    # Création de la matrice de parités
    for index in range(NUMBER**2) :
            if(( (x[index][0]*x[index][1]) )%2==0):
                y.append(1)
            else :
                y.append(0)

    # Affichage des parités
    print("Affichage des parités")
    print(y)
    print("")


    ''' Etape 4 '''

    # Solution 1 pour séparer les datasets
    dataset_entrainement_x=[]
    dataset_entrainement_y = []
    dataset_test_x = []
    dataset_test_y = []

    # Répartition manuelle des données d'entraînement (80%) et de test (20%) selon la valeur de j
    for i in range(NUMBER**2) :

            if(x[i][1]< int(NUMBER*0.80) ):
                dataset_entrainement_x.append( x[i] )
                dataset_entrainement_y.append(y[i])


            elif( x[i][1] >= int(NUMBER*0.80)):
                dataset_test_x.append( x[i] )
                dataset_test_y.append(y[i])

    # Solution 2 pour séparer les datasets (séparation plus aléatoire)
    dataset_entrainement_x, dataset_test_x, dataset_entrainement_y, dataset_test_y = train_test_split(x, y, test_size=0.2, random_state=42)

    # Affichage des datasets
    print("Affichage des datasets")
    print("dataset entrainement entrées : ", dataset_entrainement_x)
    print("dataset entrainement sorties : ",dataset_entrainement_y)
    print("dataset test entrées : ", dataset_test_x)
    print("dataset test sorties : ", dataset_test_y)
    print("")


    ''' Etape 5 '''
    model = MLPClassifier(hidden_layer_sizes=(15,), max_iter=1000, random_state=42)

    ''' Etape 6 '''
    model.fit(dataset_entrainement_x, dataset_entrainement_y)

    ''' Etape 7 '''
    print("Nombre d'époques utilisées pour converger : ", model.n_iter_)

    ''' Etape 8-9 '''
    accuracy = model.score(dataset_test_x, dataset_test_y)
    print("Précision sur les données de test : ", accuracy)

    ''' Etape 10 '''
    y_pred = model.predict(dataset_test_x)
    print("Valeurs d'entrées ", dataset_test_x[:10])
    print("Valeurs de sorties prédites :     ", y_pred[:10])
    print("Valeurs de sorties attendues :", dataset_test_y[:10])


