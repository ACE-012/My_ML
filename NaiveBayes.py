data={'age':['youth','youth','middle-aged','senior','senior','senior','middle-aged','youth','youth','senior','youth','middle-aged','middle-aged','senior'],
      'income':['high','high','high','medium','low','low','low','medium','low','medium','medium','medium','high','medium'],
      'student':['No','No','No','No','Yes','Yes','Yes','No','Yes','Yes','Yes','No','Yes','No'],
      'credit_rating':['fair','excellent','fair','fair','fair','excellent','excellent','fair','fair','fair','excellent','excellent','fair','excellent'],
      'buys_computer':['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']
      }
import pandas as pd
import numpy as np
from pandas.core.dtypes.common import classes
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
data=pd.DataFrame(data)


data=data.apply(preprocessing.LabelEncoder().fit_transform).to_numpy()
X=data[:,:-1]
Y=data[:,-1]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)

class NaiveBayes():
    def __init__(self) -> None:
        self.P_Y={}
        self.P_X={}
    def train(self,X:np.ndarray,Y:np.ndarray):
        X=X.T
        Y_count=np.bincount(Y.reshape(-1))
        for i in range(len(Y_count)):
            self.P_Y[i]=Y_count[i]/Y_count.sum()
        for i in range(X.shape[0]):
            X_unique=np.unique(X[i])
            for j in np.unique(Y):
                X_col=X[i][Y==j]
                X_count=np.bincount(X_col)
                for each in X_unique:
                    if each<len(X_count):
                        self.P_X[f'{i}_{j}_{each}']=X_count[each]/Y_count[j]
    def predict(self,X:np.ndarray):
        classes=list(self.P_Y.keys())
        out=[]
        for each in X:
            prob_Each_class=dict(self.P_Y)
            for i in range(len(each)):
                for j in classes:
                    prob_Each_class[j]*=self.P_X.get(f'{i}_{j}_{each[i]}',0)
            out.append(max(prob_Each_class,key=prob_Each_class.get))
        return out

clf=NaiveBayes()
clf.train(X_train,Y_train)
Y_pred=clf.predict([[2,2,1,1]])
print(Y_pred,Y_test)
