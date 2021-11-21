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
    F1 = np.asarray(F(x), dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = 2 * np.pi * n / N
    W = []                                          #W = [exp(-i * n * x_k)]
    for i in n:
        W1 = []
        for j in k:
            W1.append(complex(round(np.cos(i * j),3),
                              round(np.sin(-i * j),3)))
        W.append(W1)
    return np.dot(W,F1);

def FFT(x):
    N = len(x)
    if (N % 2 > 0):
        raise ValueError("size of x must be a power of 2")
    if(N > 2):
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
                                                    #W^n = [exp(-2 * i * n * Pi / N)]
        W_N1 = np.cos(2 * np.pi * np.arange(N) / N)
        W_N2 = np.sin(-2 * np.pi * np.arange(N) / N)
        W_N3 = [complex (0,y) for y in W_N2]
        W_N = W_N1 + W_N3
        
        return np.concatenate([X_even + W_N[:int(N / 2)] * X_odd,
                                X_even - W_N[:int(N / 2)] * X_odd])

    else:
        t = F_m(x)
        return t;
                                                    ## Interpolating Polynomial using FFT:
def Interpolating_Polynomial_in_point(Fast_FT, x, r):
                                                    #        Fast_FT : array returned by function FFT()
                                                    #        x : array of data points
                                                    #        r : the node to interpolate at 

    N = len(Fast_FT)
    n = int((N+1)/2)
    z = Fast_FT / N
    
    A_N = 2 * z[1:n].real                              #a(n) = c(n) + c(-n) 
    B_N =-2 * z[1:n].imag                              #b(n) = i(c(n) - c(-n))

    k = np.arange(1,len(A_N)+1)
    C = sum(A_N * np.cos( k * r ) )
    S = sum(B_N * np.sin( k * r ) )
    result = np.round(z[0] + C + S,3)

    return result;
    
def Interpolating_Polynomial(Fast_FT,x,R):
    ev = []
    for r in R:
        ev.append(Interpolating_Polynomial_in_point(Fast_FT,x,r))
    return ev;    

##data:
X_Points = X(0,15)

X_Points_ip = X(0,15)

F_Points = F(X_Points)

F_Points_ip = F(X_Points_ip)

interp_points = [np.pi*x/300 for x in np.arange(600)]

Fast_FT = np.round(FFT(X_Points),3)

interp_polynomial = Interpolating_Polynomial(Fast_FT, X_Points_ip, interp_points)

Discrete_FT = np.round(F_m(X_Points),3)

Python_FFT = np.round (np.fft.fft(F_Points),3)

Function = np.round(F_Points, 3)

Domain_points_IP = np.linspace(0, 2*np.pi, len(interp_polynomial))

Domain_points_FFT = np.linspace(0,2*np.pi, len(Fast_FT))

##prints:
##for xp in X_Points:
##    print('F_using_FFT =(',xp,') = ',Interpolating_Polynomial_in_point(Fast_FT, X_Points, xp).real,
##          '----->','F(',xp,') = ',F_in_point(xp))
for i in Fast_FT:
    print(i)

##plot:
IP_plot = plt.plot(Domain_points_IP, interp_polynomial, 'b')

FFT_plot = plt.plot(Domain_points_FFT, Fast_FT, 'orange')

F_plot = plt.plot(Domain_points_FFT, Function, 'r-')

Py_plot = plt.plot(Domain_points_FFT, Python_FFT, 'black')

TrueFFT_plot = plt.plot(Domain_points_FFT, Discrete_FT, 'y--')

plt.legend([IP_plot[0], FFT_plot[0], F_plot[0], TrueFFT_plot[0], Py_plot[0]],("Interpolating Polynomial","FFT","F(x)","DFT","python FFT"),loc="best")

plt.axis([0,2*np.pi,-8,8])

plt.show()


