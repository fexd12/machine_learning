from pre_process.pre_process_credit import pre_process_credit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

def svm_credit_main():
    previsor_treinamento,previsor_teste, classe_treinamento,classe_teste = pre_process_credit()


    classificador = SVC(kernel='linear',random_state=1) #kernel gaussiano
    classificador.fit(previsor_treinamento,classe_treinamento)

    resultado = classificador.predict(previsor_teste)
    accuracy = accuracy_score(classe_teste,resultado)
    matriz = confusion_matrix(classe_teste,resultado)

    print(accuracy,matriz)
