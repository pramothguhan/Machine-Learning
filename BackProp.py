import numpy as np
import math
x1=1;x2=0;x3=1
bias=np.array([1,1,1])
I4 = np.array([bias[0],x1,x2,x3])
I5 = np.array([bias[1],x1,x2,x3])
Wi4 = np.array([-0.4,0.2,0.4,-0.5])
Wi5 = np.array([0.2,-0.3,0.1,0.2])
Wjk = np.array([-0.3,-0.2])
print(Wi4,Wi5,Wjk)
Wbk = 0.1
eta = 0.9
Ij = np.array([0.0,0.0,0.0],dtype="double")
Oj = np.array([0.0,0.0],dtype="double")
Err = np.array([0.0,0.0,0.0],dtype="double")
t=1
delWjk = np.array([0.0,0.0])
delWi4 = np.array([0.0,0.0,0.0],dtype="double")
delWi5 = np.array([0.0,0.0,0.0],dtype="double")
Ok=0.0
for i in range(5):
 print("Epoch!!!! ",(i+1))
 Ij[0] = x1*Wi4[1]+x2* Wi4[2]+x3*Wi4[3]+bias[0]*Wi4[0]
 Ij[1] = x1*Wi5[1]+x2* Wi5[2]+x3*Wi5[3]+bias[1]*Wi5[0]
 Oj[0] = 1/(1+math.exp(-Ij[0]))
 Oj[1] = 1/(1+math.exp(-Ij[1]))
 Ij[2] = Oj[0]*Wjk[0]+Oj[1]*Wjk[1]+Wbk*bias[2]
 Ok= 1/(1+math.exp(-Ij[2]))
 Err[2] = Ok*(1-Ok)*(t-Ok)
 Err[1] = Oj[1]*(1-Oj[1])*Err[2]*Wjk[1]
 Err[0] = Oj[0]*(1-Oj[0])*Err[2]*Wjk[0]
 delWjk = (eta*Err[2])*Oj
 delWbk = eta*Err[2]*bias[2]
 Wbk = Wbk+delWbk
 Wjk = Wjk+delWjk
 delWi4 = (eta*Err[0])*I4
 delWi5 = (eta*Err[1])*I5
 Wi4 = Wi4 + delWi4
 Wi5 = Wi5 + delWi5
 print("New Weights:!!!!")
 print("W46:",Wjk[0],"W56:",Wjk[1])
 print("W14:",Wi4[1],"W15:",Wi5[1])
 print("W24:",Wi4[2],"W25:",Wi5[2])
 print("W34:",Wi4[3],"W35:",Wi5[3])
 print("Wb4:",Wi4[0],"Wb5:",Wi5[0],"Wb6:",Wbk)
