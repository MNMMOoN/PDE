import numpy as np;
import matplotlib.pyplot as plt

def F(X):
    Func = []
    for x in X:
        f = F_in_point(x)
        Func.append(f)
    return Func;

def F_in_point(x):
    Pi = np.pi
    e = np.exp(-x)
    f = np.round(x*(x-2*Pi)*e,3)
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

def Interpolating_Polynomial_in_point(a, x, r):
#     ''' a : array returned by function coef()
#        x : array of data points
#        r : the node to interpolate at  '''
    f = [u for u in x]
    f[0] = 1
    f[1] = r - x[0]
    for i in range(2,len(a)):
        f[i] = f[i-1] * (r - x[i-1])
    return sum(f * a);
    
def Interpolating_Polynomial(a,x,R):
    ev = []
    for r in R:
        ev.append(Interpolating_Polynomial_in_point(a,x,r))
    return ev;    

def Plot(x,y):
    plt.plot(x,y)
    plt.show()

def Fourier(Fast_FT, X_Points):
    N = len(Fast_FT)
    n = np.arange(N)
    #e^ i n x
    D = n * X_Points
    w1 = np.cos(D)
    w2 = np.sin(D)
    w3 = [complex(0,y) for y in w2]
    w = w1 + w3
    Fast_coef = np.round(w * Fast_FT)
    print(Fast_coef)
    plt.plot(X_Points, Fast_coef, 'purple')



##data:
X_Points = X(0,15)

X_Points_ip = X(0,15)

F_Points = F(X_Points)

F_Points_ip = F(X_Points_ip)
    
interp_coef = np.round(coef(X_Points_ip, F_Points_ip), 3)

interp_points = [np.pi*x/30 for x in np.arange(60)]

interp_polynomial = Interpolating_Polynomial(interp_coef, X_Points_ip,interp_points)

Fast_FT = np.round(FFT(X_Points),3)

Discrete_FT = np.round(F_m(X_Points),3)

Python_FFT = np.round (np.fft.fft(X_Points),3)

Function = np.round(F_Points, 3)

Domain_points_IP = np.linspace(0, 2*np.pi, len(interp_polynomial))

Domain_points_FFT = np.linspace(0,2*np.pi, len(Fast_FT))

##prints:
####matrix = np.array(np.round(FFT(X_Points),2))
####for mat in range (matrix.shape[0]):
####    print('F_hat(',mat,'): ', matrix[mat])
####
Fourier(Fast_FT, X_Points)

##plot:
IP_plot = plt.plot(Domain_points_IP, interp_polynomial)

FFT_plot = plt.plot(Domain_points_FFT, Fast_FT, 'orange')

F_plot = plt.plot(Domain_points_FFT, Function, 'r-')

Py_plot = plt.plot(Domain_points_FFT, Python_FFT, 'black')

TrueFFT_plot = plt.plot(Domain_points_FFT, Discrete_FT, 'y--')

plt.legend([IP_plot[0], FFT_plot[0], F_plot[0], TrueFFT_plot[0], Py_plot[0]],("Interpolating Polynomial","FFT","F(x)","DFT","python FFT"),loc="best")

plt.axis([0,2*np.pi,-8,8])

plt.show()


