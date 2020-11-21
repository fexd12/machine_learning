import sklearn
from pre_process.pre_process_census import pre_process_census
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score,confusion_matrix

def nb_census_main():


    previsor_treinamento, previsor_teste, classe_treinamento, classe_teste = pre_process_census()

    classificador = GaussianNB()
    classificador.fit(previsor_treinamento,classe_treinamento)
    resultado = classificador.predict(previsor_teste)

    accuracy = accuracy_score(classe_teste,resultado) # porcetagem de acertos
    matriz = confusion_matrix(classe_teste,resultado) # saber quantos acertos e erros
    print(accuracy,matriz)

