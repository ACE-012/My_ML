import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ExtraFunctions.GraphvizExtra import visualize_tree
data=pd.read_csv('./diabetes.csv')
input_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age"]
output_cols = ["Outcome"]

X = data[input_cols]
Y = data[output_cols]

print(X.shape,Y.shape)
print(type(X))

X_train,X_test,Y_train,Y_test =train_test_split(X.values,Y.values,test_size=0.2)


class DecisionTree():
    def __init__(self,class_names:list[str],out_labels:list[str],algo="gini",min_thres=0.0) -> None:
        self.__tree={}
        self.class_names=class_names
        self.out_labels=out_labels
        self.min_thres=min_thres
        self.algo=algo
    def train(self,X:np.ndarray,Y:np.ndarray):
        self.__tree=self.__create__gini_tree(X,Y,labels=self.class_names) if self.algo=="gini" else self.__create__IfGain_tree(X,Y,labels=self.class_names)

    def __create__IfGain_tree(self,X:np.ndarray,Y:np.ndarray,prev=-1,labels=[])->dict|str:
        if prev!=-1:
            X=np.delete(X,prev,1)
            #print(labels,prev)
            labels=[element for i, element in enumerate(labels) if i != prev]
        max_IfGain=0
        max_val=0
        max_indx=0
        max_leaf={"Left":-1,"Right":-1}
        for i in range(X.shape[1]):
            if X.size==0:
                return "unk"
            IfGain,val,_,leaf=self.__calculate_IfGain(X,Y,i)
            if max_IfGain<IfGain:
                max_IfGain=IfGain
                max_indx=i
                max_val=val
                max_leaf=leaf
        try:
            X_indx=X[:,max_indx]
        except IndexError:
            return self.out_labels[1] if (Y==1).sum()>(Y==0).sum() else self.out_labels[0]
        sub_tree= {"Label":labels[max_indx],"val":max_val,"IfGain":max_IfGain,
                   "Left":self.out_labels[max_leaf["Left"]] if max_leaf["Left"]!=-1 else self.__create__IfGain_tree(X[X[:,max_indx]<max_val],Y[X_indx<max_val],max_indx,list(labels)),
                   "Right":self.out_labels[max_leaf["Right"]] if max_leaf["Right"]!=-1 else self.__create__IfGain_tree(X[X[:,max_indx]>=max_val],Y[X_indx>=max_val],max_indx,list(labels))}
        if type(sub_tree["Left"])==str and type(sub_tree["Right"])==str and sub_tree["Left"]==sub_tree["Right"]:
            return sub_tree["Left"]
        if sub_tree["Left"]=="unk" or sub_tree["Right"]=="unk":
            return self.out_labels[1 if (Y==0).sum()<(Y==1).sum() else 0]
            
        return sub_tree

    def __calculate_IfGain(self,X:np.ndarray,Y:np.ndarray,col_id):
        X=X[:, col_id]
        X_argsort=X.argsort()
        X=X[X_argsort]
        Y=Y[X_argsort]
        X_old=np.unique(X)
        min_X = np.min(X)
        max_X = np.max(X)
        window = np.array([1, 1]) / 2
        X_un = np.convolve(X_old, window, mode='valid')
        max_IfGain=0
        max_indx=0
        pure_leaf={"Left":-1,"Right":-1}
        if not (min_X==0 and max_X==1):
            _type="continous"
            for each in X_un:
                Left=Y[np.where(X<each)]
                Right=Y[np.where(X>=each)]                
                Total_ent=self.__ENtropy(Y)
                Left_ent=self.__ENtropy(Left)
                Right_ent=self.__ENtropy(Right)
                IfGain=Total_ent-((len(Left)/len(Y))*Left_ent)-((len(Right)/len(Y))*Right_ent)
                if max_IfGain<IfGain:
                    max_IfGain=IfGain
                    max_indx=each
                    if Left_ent==1 or (Left==1).sum()<self.min_thres or (Left==0).sum()<self.min_thres:
                        #print(Left_Y,Left_N)
                        pure_leaf["Left"]=1 if (Left==1).sum()>(Left==0).sum() else 0
                    if Right_ent==1 or (Right==1).sum()<self.min_thres or (Right==0).sum()<self.min_thres:
                        pure_leaf["Right"]=1 if (Right==1).sum()>(Right==0).sum() else 0
        else:
            _type="bool"
        return max_IfGain,max_indx,_type,pure_leaf

    def __ENtropy(self,Y:np.ndarray):
        Y=Y.reshape(-1)
        counts = np.bincount(Y)
        probabilities = counts / len(Y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


    def __create__gini_tree(self,X:np.ndarray,Y:np.ndarray,prev=-1,labels=[])->dict|str:
        if prev!=-1:
            X=np.delete(X,prev,1)
            #print(labels,prev)
            labels=[element for i, element in enumerate(labels) if i != prev]
        min_gini=1
        min_val=0
        min_indx=0
        min_leaf={"Left":-1,"Right":-1}
        #print(X.shape)
        #print(X,Y)
        for i in range(X.shape[1]):
            if X.size==0:
                return "unk"
            gini,val,_,leaf=self.__calculate_gini(X,Y,i)
            if min_gini>gini:
                min_gini=gini
                min_val=val
                min_indx=i
                min_leaf=leaf
        #print(min_gini,min_val,min_indx,min_leaf)
        try:
            X_indx=X[:,min_indx]
        except IndexError:
            return self.out_labels[1] if (Y==1).sum()>(Y==0).sum() else self.out_labels[0]
        #print(min_leaf)
        
        sub_tree= {"Label":labels[min_indx],"val":min_val,"gini":min_gini,
                   "Left":self.out_labels[min_leaf["Left"]] if min_leaf["Left"]!=-1 else self.__create__gini_tree(X[X[:,min_indx]<min_val],Y[X_indx<min_val],min_indx,list(labels)),
                   "Right":self.out_labels[min_leaf["Right"]] if min_leaf["Right"]!=-1 else self.__create__gini_tree(X[X[:,min_indx]>=min_val],Y[X_indx>=min_val],min_indx,list(labels))}
        if type(sub_tree["Left"])==str and type(sub_tree["Right"])==str and sub_tree["Left"]==sub_tree["Right"]:
            return sub_tree["Left"]
        if sub_tree["Left"]=="unk" or sub_tree["Right"]=="unk":
            return self.out_labels[1 if (Y==0).sum()<(Y==1).sum() else 0]

        return sub_tree


    def __calculate_gini(self,X:np.ndarray,Y:np.ndarray,col_id=0)->tuple[np.float64|int,np.float64|int,str,dict[str,int]]:
        X=X[:, col_id]
        X_argsort=X.argsort()
        X=X[X_argsort]
        Y=Y[X_argsort]
        X_old=np.unique(X)
        min_X = np.min(X)
        max_X = np.max(X)
        window = np.array([1, 1]) / 2
        X_un = np.convolve(X_old, window, mode='valid')

        #print(X_un)
        
        min_gini=1
        min_indx=0
        pure_leaf={"Left":-1,"Right":-1}
        if not (min_X==0 and max_X==1):
            _type="continous"
            for each in X_un:
                Left=Y[np.where(X<each)]
                Right=Y[np.where(X>=each)]
                Left_Y=(Left==1).sum()
                Left_N=(Left==0).sum()
                Right_Y=(Right==1).sum()
                Right_N=(Right==0).sum()
                Left_gini=self.__gini(Left_Y,Left_N)
                Right_gini=self.__gini(Right_Y,Right_N)
                Total=Left_N+Left_Y+Right_N+Right_Y
                total_gini:int|np.float64=(((Left_N+Left_Y)/Total)*Left_gini)+(((Right_Y+Right_N)/Total)*Right_gini)
                #print(Left_Y,Left_N,Right_Y,Right_N,each)
                #print(Left_gini,Right_gini,total_gini)
                if(min_gini>total_gini):
                    min_gini=total_gini
                    min_indx=each
                    if Left_gini==0 or Left_Y<self.min_thres or Left_N<self.min_thres:
                        #print(Left_Y,Left_N)
                        pure_leaf["Left"]=1 if Left_Y>Left_N else 0
                    if Right_gini==0 or Right_Y<self.min_thres or Right_N<self.min_thres:
                        pure_leaf["Right"]=1 if Right_Y>Right_N else 0
                    
        else:
            _type="bool"
        return min_gini,min_indx,_type,pure_leaf
    def __gini(self,Y,N):
        return 0 if (Y==0 and N==0) else 1-((Y/(Y+N))**2)-((N/(Y+N))**2)


    def predict(self,X:np.ndarray):
        ret=[]
        if len(X.shape)==2:
            for each in X:
                ret.append(self.__predict_helper(self.__tree,each))
        #print(ret)
        return np.array(ret).reshape(-1,1)
    def __predict_helper(self,tree,X):
        if type(tree)==str:
            return self.out_labels.index(tree)
        if tree["val"]>X[self.class_names.index(tree["Label"])]:
            return self.__predict_helper(tree["Left"],X)
        return self.__predict_helper(tree["Right"],X)
    def visualize(self):
        attributes=['Label','val',self.algo]
        visualize_tree(self.__tree,attributes=attributes)


clf=DecisionTree(input_cols,["No","Yes"],min_thres=0,algo="IfGain")
clf.train(X_train,Y_train)
Y_pred=clf.predict(X_test)
clf.visualize()
print(((Y_pred==Y_test).sum()/Y_pred.shape[0])*100)

