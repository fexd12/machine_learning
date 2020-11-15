# from pre_process.pre_process_risco import pre_process_risco

from pre_process.pre_process_risco  import pre_process_risco

from sklearn.naive_bayes import GaussianNB

def nb_risco_main():

    previsor,classe = pre_process_risco()

    classificador = GaussianNB()

    #verificar se é necesario criar variaveis dummy

    classificador.fit(previsor,classe) #ajusta de tabela de probabilidades
    resultado = classificador.predict([[
        0,0,1,2
    ],[
        3,0,0,0
    ]])
    # print(classificador.classes_) #ver classes
    # print(classificador.class_prior_) #probabilidades prior das classes
    # print(resultado)
