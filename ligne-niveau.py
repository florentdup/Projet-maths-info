import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect
import autograd

f=lambda x,y:x**2+y**2-np.exp(-(x-1)**2-(y-1)**2)
f=lambda x,y:x**2+y**2

X,Y=np.linspace(0,1,100),np.linspace(0,1,100)

X,Y=np.meshgrid(X,Y)

Z=f(X,Y)

plt.pcolormesh(X, Y, Z,cmap='PuBu_r')
plt.colorbar()

def dichotomie(f,a,b,epsilon):
    g,d=a,b
    while (d-g)>epsilon:
        c=(d+g)/2
        if f(a)*f(c)<0:
            d=c
        else:
            g=c
    return c

def grad_f(x, y):
    g = autograd.grad
    return np.r_[g(f, 0)(x, y), g(f, 1)(x, y)]


def find_seed(g, c=0.0, eps=2**(-26)):
    if (g(0,0)-c)*(g(0,1)-c)<0:
        tmp=lambda x : g(0,x)-c  
        return dichotomie(tmp,0,1,2**-26)
    else:
        return None



def simple_contour(f, c=0.0, delta=0.01):
    X,Y=[],[]
    d0=find_seed(f,c)
    if d0!=None:
        x,y=0.0001,d0
        gradient=grad_f(x,y)
        s = 1
        if gradient[1] < 0:
            s = -1
        while 0<=x<1 and 0<=y<1:
            X.append(x)
            Y.append(y)
            gradient=grad_f(x,y)
            n=[gradient[1],-gradient[0]]
            norme=s*np.sqrt(n[0]**2+n[1]**2)
            if norme!=0:
                x+=delta/norme*n[0]
                y+=delta/norme*n[1]
            else:
                break
        
    return X,Y

X_l,Y_l=simple_contour(f,0.01)
plt.plot(X_l,Y_l,"*",color="black")



def find_seed_4(g, c=0.0, eps=2**(-26)):
    if (g(0,0)-c)*(g(0,1)-c)<0:
        tmp=lambda x : g(0,x)-c  
        return dichotomie(tmp,0,1,2**-26)
    else:
        return None

def case(f,c,x0,y0,x1,y1,delta):
    return X,Y


def contour(f, c=0.0, xc=[0.0,1.0], yc=[0.0,1.0], delta=0.01):
    X, Y = [], []
    n = len(xc) - 1
    for i in range(n):
        for j in range(n):
            a,b = case(f,c,xc[i],yc[j],xc[i+1],yc[j+1],delta)
            X += a
            Y += b

    return X,Y

 

plt.show()

