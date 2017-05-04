import numpy as np
from numpy.linalg import norm, cond, lstsq, qr
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

## Utility functions
def vande_cond(n):
    mat = []
    for i in range(n+1):
        x = -1 + (2*i)/n
        mat.append([x**k for k in range(n+1)])
    print(cond(mat))

def qralg(mat, tol=None):
    mat = np.array(mat)
    if tol:
        Z = np.eye(len(mat))
        true_e = np.linalg.eig(mat)[0]
        runs = 0
        while max(abs(true_e - np.diag(Z.dot(mat).dot(Z)))) > tol:
            Z = qr(mat.dot(Z))[0]
            runs += 1
        return (np.diag(Z.dot(mat).dot(Z)), runs)
    else:
        Z = np.eye(len(mat))
        for i in range(10):
            Z = qr(mat.dot(Z))[0]
        return (np.diag(Z.dot(mat).dot(Z)), 10)

#Specific problems

def p2_c():
    for i in [5, 10, 20, 30]:
        print("Condition for %d by %d is" % (i, i))
        vande_cond(i)
## Response:
# Condition for 5 by 5 is
# 63.8272825964
# Condition for 10 by 10 is
# 13951.6269315
# Condition for 20 by 20 is
# 831377051.624
# Condition for 30 by 30 is
# 5.64223880998e+13

def p3_a():
    x = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    mat = [[i**k for k in range(3, -1, -1)] for i in x]
    y = [0.0, .2, .27, .3, .32, .33]
    title_count = 1
    for end in [None, 5, 4]:
        tmp_mat = mat[:end]
        tmp_y = y[:end]
        sol = lstsq(tmp_mat, tmp_y)[0]
        p = poly.Polynomial(list(reversed(sol)))
        print ("Solution in this case is: ")
        print(p)
        plt.plot(x[:end], tmp_y, 'ro')
        x_lin = np.arange(-.2, 2.7, .01)
        plt.plot(x_lin, p(x_lin))
        plt.title("Least squares solution to problem "+ "i"*title_count)
        plt.savefig("p3a" + "i"*title_count+".png")
        plt.clf()
        title_count += 1

def p4_a():
    p = poly.Polynomial([1, -5.15878, 24.2443])
    x = np.arange(-.2, 1.2, .01)
    plt.plot(x, np.exp(3*x), label="Real function")
    plt.plot(x, p(x), "--", label="Poly approximation")
    plt.legend()
    plt.title("Problem 4, a")
    plt.savefig("p4a.png")

def p4_b():
    L0 = poly.Polynomial([1, -3, 2])
    L1 = poly.Polynomial([0, 4, -4])
    L2 = poly.Polynomial([0, -1, 2])
    x = np.arange(-.1, 1.1, .01)
    i = 0
    for l, s in [(L0, '-'), (L1, '--'), (L2, '-.')]:
        plt.plot(x, l(x), s, label="L"+str(i))
        i+=1
    plt.plot(x, np.zeros(len(x)), 'k')
    plt.title("Problem 4, b")
    plt.legend()
    plt.savefig("p4b.png")

def p4_d():
    L0 = poly.Polynomial([1, -2])
    L1 = poly.Polynomial([0, 2])
    H0 = L0**2*(1 - 2*L0.deriv()*poly.Polynomial([0, 1]))
    H1 = L1**2*(1 - 2*L1.deriv()*poly.Polynomial([-1/2, 1]))
    K0 = L0**2*poly.Polynomial([0, 1])
    K1 = L1**2*poly.Polynomial([-1/2, 1])
    x = np.arange(-.1, 1.1, .01)
    i = 0
    for h, k, s, d in [(H0, K0, '-', '--'), (H1, K1, '-.', ':')]:
        plt.plot(x, h(x), s, label="H"+str(i))
        plt.plot(x, k(x), d, label="K"+str(i))
        i+= 1
    plt.title("Problem 4, d")
    plt.legend()
    plt.savefig("p4d.png")

def p5_d():
    x = np.arange(0, 1, .05)
    res = [qralg([[1, l],[l, 1]], 1e-10)[1] for l in x[1:]]
    plt.plot(x[1:], res, "ro")
    plt.title("Problem 5, d")
    plt.savefig("p5d.png")


if __name__=="__main__":
    p4_d()
