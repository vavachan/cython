#here I have given a sample coupled stochastic differential equation.
#Its example 3 in Joshua Wilkie's paper. the program will generate a file called t_X'1_X'2_X1_X2.dat 
#in which column 1 and column 2 are the values found by integration and column 3 and 4 are the corresponding value 
#from analytic solution.
m=0.5
omega=1
eta=1
a=1
KB=1.38*10**(-2)
T=300#
dt=.0001
M=3
D=2
N=40000
Xzero=0
import time 
import numpy as np
import sde
def true(i,w1,w2,w3):
    X=np.zeros(shape=D)
    X[0]=np.exp(-2*i*dt+w1-w2)*np.cos(w3)
    X[1]=np.exp(-2*i*dt+w1-w2)*np.sin(w3)
    return X 
def gamma():
    return 100
    #return 6*3.14*eta*a
#def drift(X):
#    temp=np.zeros(shape=D)
#    temp[0]=-(3.0/2)*X[0]
#    temp[1]=-(3.0/2)*X[1]
#    return temp
#def diffusion(X):
#    temp=np.zeros(shape=(D,M))
#    temp[0][:]=[X[0],-X[0],-X[1]]
#    temp[1][:]=[X[1],-X[1],X[0]]
    #print X,temp
#    return temp
#3def jac_diff(X):
#    temp=np.zeros(shape=(D,M,D))
#    temp[0][:][:]=[[1,0],[-1,0],[0,-1]]
#    temp[1][:][:]=[[0,1],[0,-1],[1,0]]
#    return temp
A=sde.D1()
B=sde.diff()
C=sde.JD()
#print A.rhs([1.0,1.0])
X_ini=[1.0,0.0]
choice=2
solver = sde.ito(N,dt,A,B,C,D,M,choice) 
solver.set_initial_condition(X_ini)
start_time=time.time()
x=solver.solve()
print("--- %s seconds ---" % (time.time() - start_time))
X=np.zeros(shape=N)
V=np.zeros(shape=N)
infile=open("t_X'1_X'2_X1_X2.dat","w")
infile2=open("Histogram.dat","w")
infile3=open("Correlation.dat","w")
for i in range (0,N):
    X[i]=x[i][0]
    V[i]=x[i][1]
hist=np.histogram(X,1000)
#print hist    
for i in range (0,N):
    infile.write(str(i*dt)+'\t') # the following 3 lines writes all the information to a file.
    infile.write(str(solver.X_all[i][0])+'\t'+str(solver.X_all[i][1])+'\t'+str(true(i,solver.W[i][0],solver.W[i][1],solver.W[i][2])[0])+'\t'+str(true(i,solver.W[i][0],solver.W[i][1],solver.W[i][2])[1]))
    infile.write('\n')
exit()
for i in range (0,1000):
    infile2.write(str(hist[0][i]*1.0/N)+'\t'+str(hist[1][i])+'\n') 
for T in range (0,N/2,100):
    corr=0
#    if T/1000 > 20:
#        exit()
    if T%1000 == 0:
        print T/1000
        for i in range (0,N/2):
            corr=corr+(solver.X_all[i][0]-solver.X_all[i+T][0])**2
        infile3.write(str(T*dt)+'\t'+str(corr/(N/2))+'\n')
exit()

