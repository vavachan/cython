import numpy as np
cimport numpy as np
#cimport cython
#@cython.boundscheck(False)
cdef class drift:
    cdef np.ndarray temp
    cdef np.ndarray X
    def __init__(self):
        self.temp = np.zeros(2)

    def rhs(self,np.ndarray X):
        return 0
cdef class diffusion:
    cdef np.ndarray temp

    def __init__(self):
        self.temp = np.zeros(shape=(2,3))

    def rhs(self,np.ndarray X):
        return 0
cdef class jac_diff:
    cdef np.ndarray temp

    def __init__(self):
        self.temp = np.zeros(shape=(2,3,2))

    def rhs(self,np.ndarray X):
        return 0

cdef class D1(drift):
    def rhs(self,np.ndarray X):
        self.temp[0]=-(3.0/2)*X[0]
        self.temp[1]=-(3.0/2)*X[1]
        return self.temp
cdef class diff(diffusion):
    def rhs(self,np.ndarray X):
        self.temp[0][:]=[X[0],-X[0],-X[1]]
        self.temp[1][:]=[X[1],-X[1],X[0]]
        return self.temp
cdef class JD(jac_diff):
    def rhs(self,np.ndarray X):
        self.temp[0][:][:]=[[1,0],[-1,0],[0,-1]]
        self.temp[1][:][:]=[[0,1],[0,-1],[1,0]]
        return self.temp


#cpdef unpack(f_type f):
#    return f
cdef class ito (object):
    cdef public int M,D,N,choice
    cdef public double dt
    cdef public drift A
    cdef public diffusion B
    cdef public jac_diff B_der
    cdef public np.ndarray X_all,dW,W
    def __init__(self,int N,float dt,drift P,diffusion Q,jac_diff R,int D,int M,int choice):
        self.choice=choice
        self.M=M
        self.D=D
        self.A=P
        self.B=Q
        self.B_der=R
        self.N=N
        self.dt=dt
        self.X_all=np.zeros(shape=(self.N,self.D))
        self.dW=np.zeros(shape=(self.N,self.M))
        self.W=np.zeros(shape=(self.N+1,M))
    cdef multiply(self, A, B):
        #cdef np.ndarray[double,ndim=1] temp
        #cdef np.ndarray[double,ndim=2] ten,ten1
        temp=np.zeros(shape=self.D)
        cdef int j,k,i
        for j in range (0,self.D):
            ten=(B[j][:][:])
            ten1=np.transpose(ten)
            sum2=0
            for k in range (0,self.M):
                sum1=0
                for i in range (0,self.D):
                    sum1=sum1+A[k][i]*ten1[i][k]
                sum2=sum2+sum1
            temp[j]=sum2
        return temp    
            
    def F(self,X,W):
        D=self.D
        M=self.M
        B1=np.zeros(shape=(D,M))
        K=np.zeros(shape=(D,M))
        K=self.B.rhs(X)
        B1=np.dot(K,W)
        F1=(self.A.rhs(X)-0.5*self.multiply((np.transpose(self.B.rhs(X))),self.B_der.rhs(X)))*self.dt+B1
        return F1
    def weiner(self):
        M=self.M
        cdef int i,j
        for i in range (0,self.N):
            for j in range (0,self.M):    
                self.dW[i][j]=self.dt**(.5)*np.random.normal(0,1,1)
        for j in range (0,M):
            sum1=0
            for i in range (1,self.N+1):
                sum1=sum1+self.dW[i-1][j]
                self.W[i][j]=sum1
    
    def RK(self):
        self.weiner()
        cdef int D=self.D
        cdef np.ndarray[double,ndim=1] K1= np.zeros(shape=D)
        cdef np.ndarray[double,ndim=1] K2= np.zeros(shape=D)
        cdef np.ndarray[double,ndim=1] K3= np.zeros(shape=D)
        cdef np.ndarray[double,ndim=1] K4= np.zeros(shape=D)
        cdef np.ndarray[double,ndim=1] X= np.zeros(shape=D)
        cdef int i,j 
        for i in range (1,D):
            X[i]=self.X_all[0][i]
        for i in range (1,self.N):
            for j in range (1,self.D):
                K1[j]=self.F(X,self.dW[i-1][:])[j]
                K2[j]=self.F(X+0.5*K1,self.dW[i-1][:])[j]
                K3[j]=self.F(X+0.5*K2,self.dW[i-1][:])[j]
                K4[j]=self.F(X+K3,self.dW[i-1][:])[j]
                X[j]=X[j]+1.0/6*(K1[j]+2*K2[j]+2*K3[j]+K4[j])
            for j in range (1,D):
                self.X_all[i][j]=X[j]   #print self.dW[i][:]
       
    def em(self):
        self.weiner()
        cdef np.ndarray B1,K,winc
        cdef int i,j,D,M,N
        #cdef double winc[10]
        D=self.D
        M=self.M
        N=self.N
        #B1=np.zeros(shape=D)
        #winc=np.zeros(shape=M)
        #K=np.zeros(shape=(D,M))
        for i in range (1,N):
            for j in range (0,M):
                winc[j]=self.W[i][j]-self.W[i-1][j]
            K=self.B.rhs(self.X_all[i-1][:])
            B1=np.dot(K,winc)
            for j in range (0,D):
                 self.X_all[i][j] = self.X_all[i-1][j] + self.dt*self.A.rhs(self.X_all[i-1][:])[j] + B1[j]
    def solve(self):
        if self.choice == 1:
            print 'RK'
            self.RK()
        if self.choice == 2:
            print 'Euler-Maruyama'
            self.em()
        if self.choice != 1 and self.choice != 2: 
            print 'wrong choice',self.choice
            exit()
        return self.X_all

    def set_initial_condition(self,X_ini):
        self.X_all[0][:]=X_ini
