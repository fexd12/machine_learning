from pre_process.pre_process_credit import pre_process_credit
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import numpy as np

def ad_credit_main(seed=0):
    previsor,classe = pre_process_credit()


    kfold = StratifiedKFold(n_splits=10,shuffle=True,random_state=seed)
    resultado1 = []

    for indice_treinamento,indice_teste in kfold.split(previsor,np.zeros(shape=(previsor.shape[0],1))):

        classificador = DecisionTreeClassifier(criterion='entropy')
        classificador.fit(previsor[indice_treinamento],classe[indice_treinamento])
        # export_graphviz(   decision_tree = classificador,
        #                    out_file = 'arvore_risco.dot',
        #                    feature_names = ['historia', 'divida', 'garantias', 'renda'],
        #                    class_names = ['alto','moderado','baixo'],
        #                    filled = True,
        #                    leaves_parallel=True) # exportar arvore para visualizacao,app graphviz
        resultado = classificador.predict(previsor[indice_teste])
        resultado1.append(accuracy_score(classe[indice_teste],resultado))
    
    return np.asfarray(resultado1).mean()
