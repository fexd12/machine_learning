import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def pre_process_census():

    df = pd.read_csv('dados\census.csv')

    #separacao de previsores e classes

    previsor = df.iloc[:,0:14].values # 0 ao 13
    classe = df.iloc[:,14].values # somente coluna 14

    #converter variaveis categoricas
    label_previsor = LabelEncoder()
    previsor[:,1] = label_previsor.fit_transform(previsor[:,1])
    previsor[:,3] = label_previsor.fit_transform(previsor[:,3])
    previsor[:,5] = label_previsor.fit_transform(previsor[:,5])
    previsor[:,6] = label_previsor.fit_transform(previsor[:,6])
    previsor[:,7] = label_previsor.fit_transform(previsor[:,7])
    previsor[:,8] = label_previsor.fit_transform(previsor[:,8])
    previsor[:,9] = label_previsor.fit_transform(previsor[:,9])
    previsor[:,13] = label_previsor.fit_transform(previsor[:,13])

    # onehot = ColumnTransformer(transformers = [('OneHot',OneHotEncoder(),[1,3,5,6,7,8,9,13])],remainder = 'passthrough') # transformar em column dummy
    # previsor = onehot.fit_transform(previsor).toarray()

    # classe_label = LabelEncoder() # mudar variavel categorica 
    # classe = classe_label.fit_transform(classe)

    scaler = StandardScaler() # fazer escalonamento de dados 
    previsor = scaler.fit_transform(previsor)

    # previsor_treinamento, previsor_teste, classe_treinamento, classe_teste = train_test_split(previsor,classe,test_size=0.15,random_state=0)
    
    # return previsor_treinamento, previsor_teste, classe_treinamento, classe_teste
    return previsor,classe