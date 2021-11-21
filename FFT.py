import numpy as np;
import matplotlib.pyplot as plt

def F(X):
    Func = []
    for x in X:
        f = F_uni(x)
        Func.append(f)
    return Func;

def F_uni(x):
    Pi = np.pi
    e = np.exp(-x)
    f = round(x*(x-2*Pi)*e,3)
    return f;

def X(a,b):
    Pi = np.pi
    x = []
    for i in range (a,b+1):
        x.append(round(i*Pi/5,3))
    return x;

def F_m(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = 2 * np.pi * n / N
    W = []
    for i in n:
        W1 = []
        for j in k:
            W1.append(complex(round(np.cos(i * j),3),
                              round(np.sin(-i * j),3)))
        W.append(W1)
    return np.dot(W,x);

def FFT(x):
    N = len(x)
    if (N % 2 > 0):
        raise ValueError("size of x must be a power of 2")
    if(N > 2):
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])

        W_N1 = np.cos(2 * np.pi * np.arange(N) / N)
        W_N2 = np.sin(-2 * np.pi * np.arange(N) / N)
        W_N3 = [complex (0,y) for y in W_N2]
        W_N = W_N1 + W_N3
        
        return np.concatenate([X_even + W_N[:int(N / 2)] * X_odd,
                                X_even + W_N[int(N / 2):] * X_odd])

    else:
        t = F_m(x)
        print(t)
        return t;

def coef(x, y):
    '''x : array of data points
       y : array of f(x)  '''
    n = len(x)
    a = []
    for i in range(n):
        a.append(y[i])

    for j in range(1, n):

        for i in range(n-1, j-1, -1):
            a[i] = float(a[i]-a[i-1])/float(x[i]-x[i-j])

    return np.array(a) # return an array of coefficient

def Eval_uni(a, x, r):
##     ''' a : array returned by function coef()
##        x : array of data points
##        r : the node to interpolate at  '''
    f = [u for u in x]
    f[0] = 1
    f[1] = r - x[0]
    for i in range(2,len(a)):
        f[i] = f[i-1] * (r - x[i-1])
    return sum(f * a);
    
def Eval(a,x,R):
    ev = []
    for r in R:
        ev.append(Eval_uni(a,x,r))
    return ev;    

def Plot(x,y):
    plt.plot(x,y)
    plt.show()

a = np.round(coef(X(0,15),F(X(0,15))),3)
#print(a)
R = [np.pi*x/30 for x in np.arange(60)]
y = Eval(a, X(0,15),R)
#print(y)
z = np.round(FFT(X(0,15)),3)
#z2 = np.round(np.fft.fft(X(0,15)),3)
z3 = np.round(F_m(X(0,15)),3)
w = np.round(F(X(0,15)),3)
t1 = np.linspace(0, 2*np.pi, len(y))
t2 = np.linspace(0,2*np.pi, len(z))

f=plt.plot(t1, y)
g=plt.plot(t2, z, 'g--')
u=plt.plot(t2,w, 'r-')
#plt.plot(t2, z2, 'black')
v=plt.plot(t2, z3, 'orange')
plt.legend([f[0],g[0],u[0],v[0]],("interpolating polynomial","M_FFT","F(x)","T_FFT"),loc="best")

plt.axis([0,2*np.pi,-20,6])

plt.show()
#Plot(t, y)
#Plot(t, z)
#print(Eval_uni(coef([-1,1,2,3],[-2,0,7,26]),[-1,1,2,3]))
#print(Eval(coef(X(0,15),F(X(0,15))),X(0,15),np.pi))
#print(np.round_(FFT(X(0,15)),1))
#print(np.round_(F_m(X(0,15)),1))


