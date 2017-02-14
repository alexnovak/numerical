import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

"""
Hello grader! The following functions represent the code that
would be run for the corresponding problem. At the top are a series
of function abstractions of the method that will accept a function
pointer to perform the brunt of the work. Hope this works!
"""

def bisect(a, b, f, arr=False, ntrials=5):
    x = []
    fa = f(a)
    fb = f(b)
    for i in range(ntrials):
        c = (a + b)/2
        fc = f(c)
        x.append(c)
        if(fc*fb > 0):
            b = c
        elif(fc*fb < 0):
            a = c
        else:
            a = b = c
    if(arr):
        return x
    return x[-1]

def fixedpoint(x0, f, arr=False, ntrials=5):
    x = []
    currpoint = x0
    for i in range(ntrials):
        currpoint = f(currpoint)
        x.append(currpoint)
    if(arr):
        return x
    return x[-1]

def newtonPoly(x0, coeff, arr=False, ntrials=5):
    p = Polynomial(coeff)
    d = p.deriv()
    x = []
    currpoint = x0
    for i in range(ntrials):
        currpoint = currpoint - p(currpoint)/d(currpoint)
        x.append(currpoint)
    if(arr):
        return x
    return x[-1]

def secant(a, b, f, arr=False, ntrials=5):
    x = [a, b]
    for i in range(ntrials):
        val = x[-1] - f(x[-1])*(x[-1] - x[-2])/(f(x[-1]) - f(x[-2]))
        x.append(val)
    if(arr):
        return x[2:]
    return x[-1]

def p1_a():
    def f(x):
        return np.exp(x) - 3*x - 1
    x = np.arange(-1, 3, .01)
    fig, ax = plt.subplots()
    ax.plot(x, f(x))
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.show()

def p1_b():
    def f(x):
        return np.exp(x) - 3*x - 1
    res = bisect(1, 3, f, arr=True)
    print(res)

def p1_d():
    def f(x):
        return np.log(3*x + 1)
    res = fixedpoint(1, f, arr=True)
    print(res)

def p1_e():
    def f(x):
        return np.exp(x) - x*3 -1
    def g(x):
        return np.log(3*x + 1)

    seq1 = np.array(bisect(-1, 3, f, arr=True, ntrials=100))
    seq2 = np.array(fixedpoint(1, g, arr=True, ntrials=100))
    fix, ax = plt.subplots()
    bis = plt.plot(seq1, f(seq1), 'ro', label="Bisection")
    fix = plt.plot(seq2, f(seq2), 'b^', label="Fixed Point")
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.title("Bisection vs Fixed Point Iteration")
    plt.legend([bis, fix], labels=['Bisection', 'Fixed Point'])
    print(seq1[-10:])
    print(seq2[-10:])
    plt.show()

def p2_b():
    def f(x):
        return (x-2)**2 - np.log(x)
    res = bisect(1, 2, f, arr=True)
    print(res)

def p3():
    def f(x):
        return np.tan(x)/2
    def g(x):
        return np.arctan(2*x)
    fix, ax = plt.subplots()
    x = np.arange(-np.pi/2+.2, np.pi/2-.2, .01)
    t = plt.plot(x, f(x))
    at = plt.plot(x, g(x))
    li = plt.plot(x, x)
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.legend([t, at, li], labels=["y = tan(x)/2", "y = arctan(2x)", "y = x"])
    plt.show()

def p5():
    def f(x):
        return x + np.sin(x)
    ar = fixedpoint(2.55, f, arr=True, ntrials=100)
    print(ar)

def p4_c():
    def f(x):
        return x**3 - 2*x**2 + 2*x
    ar = fixedpoint(.00001, f, arr=True, ntrials=100)
    print(ar)

def p6_a():
    def f(x):
        return x**3 - 3*x**2 + 3
    res = secant(1, 2, f, arr=True)
    print(res)

def p6_b():
    res = newtonPoly(1.5, [3, 0, -3, 1], arr=True)
    print(res)

def p6_c():
    res = newtonPoly(2.1, [3, 0, -3, 1], arr=True)
    print(res)


