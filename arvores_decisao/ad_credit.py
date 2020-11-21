from pre_process.pre_process_credit import pre_process_credit
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix

def ad_credit_main():
    previsor_treinamento,previsor_teste, classe_treinamento,classe_teste = pre_process_credit()


    classificador = DecisionTreeClassifier(criterion='entropy',random_state=0)
    classificador.fit(previsor_treinamento,classe_treinamento)
    # export_graphviz(   decision_tree = classificador,
    #                    out_file = 'arvore_risco.dot',
    #                    feature_names = ['historia', 'divida', 'garantias', 'renda'],
    #                    class_names = ['alto','moderado','baixo'],
    #                    filled = True,
    #                    leaves_parallel=True) # exportar arvore para visualizacao,app graphviz
    resultado = classificador.predict(previsor_teste)
    accuracy = accuracy_score(classe_teste,resultado)
    matriz = confusion_matrix(classe_teste,resultado)

    print(accuracy,matriz)
