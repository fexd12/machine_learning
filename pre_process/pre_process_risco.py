import pandas as pd

from sklearn.preprocessing import LabelEncoder

def pre_process_risco():
    df = pd.read_csv('dados\\risco_credito.csv')
    
    previsor =  df.iloc[:,0:4].values
    classe = df.iloc[:,4].values

    label_previsor  = LabelEncoder()
    
    previsor[:,0] = label_previsor.fit_transform(previsor[:,0])
    previsor[:,1] = label_previsor.fit_transform(previsor[:,1])
    previsor[:,2] = label_previsor.fit_transform(previsor[:,2])
    previsor[:,3] = label_previsor.fit_transform(previsor[:,3])


    return previsor,classe