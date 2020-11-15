from pre_process.pre_process_credit import pre_process_credit
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score

def nb_credit_main():

    previsor_treinamento, previsor_teste, classe_treinamento, classe_teste = pre_process_credit()


    classificador = GaussianNB()
    classificador.fit(previsor_treinamento,classe_treinamento)
    resultado = classificador.predict(previsor_teste)

    accuracy = accuracy_score(classe_teste,resultado)
    # matriz = confusion_matrix(classe_teste,resultado)

    print(accuracy)