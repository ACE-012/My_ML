import pandas as pd
import numpy as np
data=pd.read_csv('./diabetes.csv')

input_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
output_cols = ["Outcome"]

X_orig=data[input_cols].to_numpy()
Y=data[output_cols].to_numpy()

X_1=X_orig[:,1]>129.5
X=X_orig[X_1]
Y=Y[X_1]
X_5=X[:,5]>27.85
X=X[X_5]
Y=Y[X_5]
X_0=X[:,0]<7.5
X=X[X_0]
Y=Y[X_0]
print(X.shape,(Y==1).sum(),(Y==0).sum())
