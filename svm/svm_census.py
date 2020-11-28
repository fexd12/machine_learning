from sklearn.svm import SVC
from pre_process.pre_process_census import pre_process_census
from sklearn.metrics import accuracy_score, confusion_matrix

def svm_census_main():
    previsor_treinamento,previsor_teste, classe_treinamento,classe_teste = pre_process_census()


    classificador = SVC(C=2.0,kernel='rbf',random_state=1) #kernel gaussiano
    classificador.fit(previsor_treinamento,classe_treinamento)

    resultado = classificador.predict(previsor_teste)
    accuracy = accuracy_score(classe_teste,resultado)
    matriz = confusion_matrix(classe_teste,resultado)

    print(accuracy,matriz)
