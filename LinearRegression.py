from matplotlib import pyplot as plt
import numpy as np
from ExtraFunctions.extrafn import line_with_slope_and_intercept

X=np.array([3,8,9,13,3,6,11,21,1,16])
Y=np.array([30,57,64,72,36,43,59,90,20,83])

plt.scatter(X,Y)


X_=X.mean()
Y_=Y.mean()

A = np.round(((X-X_)*(Y-Y_)).sum()/np.square(X-X_).sum(),1)
B = np.round(Y_-(A*X_),1)


line_with_slope_and_intercept(A,B)
plt.show()
