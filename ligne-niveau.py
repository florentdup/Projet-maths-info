import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect

f=lambda x,y:x**2+y**2-np.exp(-(x-1)**2-(y-1)**2)

X,Y=np.linspace(-1,1,100),np.linspace(-1,1,100)

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



def find_seed(g, c=0.0, eps=2**(-26)):
    if g(0)*g(1)<=0:
        tmp=lambda x : g(0,x)-c  
        return dichotomie(tmp,0,1,2**-26)
    else:
        return None

def simple_contour(f, c=0.0, delta=0.01):
    X,Y=np.array([]),np.array([])
    d0=find_seed(f,c)
    if d0!=None:
        
    if (pass):
        pass
    return x, y




plt.show()

