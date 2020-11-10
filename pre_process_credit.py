import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import   StandardScaler
df = pd.read_csv('dados\credit_data.csv')

print(df)

# df.drop(df.loc[df['age'] <0 ].index,inplace=True)

# print(df)

# df['age'][df.age > 0].mean()
df.loc[df.age < 0,'age' ] = df['age'][df.age > 0].mean() # trocar valores negativos pela media dos positivos

# print(df.loc[pd.isnull(df['age'])]) # verififcar se existe valore NaN

previsor = df.iloc[:,1:4].values
classe = df.iloc[:,4].values

imputer  = SimpleImputer(missing_values=np.nan,strategy='mean') #trocar valores faltantes ( NaN ) 
imputer.fit(previsor[:,0:3])
previsor[:,0:3] = imputer.transform(previsor[:,0:3])
print(previsor)
scaler_previsor = StandardScaler() # manter mesma grandeza
previsor = scaler_previsor.fit_transform(previsor)

print(previsor)