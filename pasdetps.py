import matplotlib.pyplot as plt

nbDt = 1000

def solve_euler_explicit(f, x0, dt):
    x = [x0[0]]
    t = [x0[1] + i*dt for i in range(0,nbDt)]

    for i in range(1,nbDt):
        xp1 = x[i-1] + dt * f(t[i-1], x[i-1])
        x.append(xp1)

    return t, x

def f(t,x):
    return -0.02*x

t, x = solve_euler_explicit(f,[1,0], 0.1)

plt.plot(t,x)
plt.show()