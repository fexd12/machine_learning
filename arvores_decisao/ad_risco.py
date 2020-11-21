from pre_process.pre_process_risco import pre_process_risco
from sklearn.tree import DecisionTreeClassifier,export_graphviz

def ad_risco_main():
    previsor_treinamento, classe_treinamento = pre_process_risco()


    classificador = DecisionTreeClassifier(criterion='entropy')
    classificador.fit(previsor_treinamento,classe_treinamento)
    # export_graphviz(   decision_tree = classificador,
    #                    out_file = 'arvore_risco.dot',
    #                    feature_names = ['historia', 'divida', 'garantias', 'renda'],
    #                    class_names = ['alto','moderado','baixo'],
    #                    filled = True,
    #                    leaves_parallel=True) # exportar arvore para visualizacao,app graphviz
    resultado = classificador.predict([[
        0,0,1,2
    ],[
        3,0,0,0
    ]])
    print(resultado)
