import numpy as np
from numpy.linalg import norm, inv
import matplotlib.pyplot as plt
from scipy.linalg import hilbert, qr, solve
from timeit import default_timer as timer

# Utility Functions
def m1norm(matrix):
    return max(abs(matrix).sum(axis=0))

def minfnorm(matrix):
    return max(abs(matrix).sum(axis=1))

def p1_a():
    print("%2.20f" % ( (1 - 1) + 1e-16, ))
    print("%2.20f" % ( 1 - (1 + 1e-16),) )

def p1_b():
    h5 = hilbert(5)
    e5 = np.ones(5)
    print("N = 5: ")
    print(norm(h5.dot(inv(h5).dot(e5)) - e5))
    print()
    h10 = hilbert(10)
    e10 = np.ones(10)
    print("N = 10: ")
    print(norm(h10.dot(inv(h10).dot(e10)) - e10))
    print()
    h20 = hilbert(20)
    e20 = np.ones(20)
    print("N = 20: ")
    print(norm(h20.dot(inv(h20).dot(e20)) - e20))
    print()

def p1_c():
    def f(x):
        return np.exp(x)/(np.cos(x)**3 + np.sin(x)**3)
    x = np.pi/4
    x_ax = [10**(-h) for h in range(1, 17)]
    y_ax = [3.101766393836051 - (f(x + h) - f(x))/h for h in x_ax]
    plt.loglog(x_ax, y_ax, '-ro')
    plt.show()

def p2_c():
    n = 100
    for i in range(1, 8):
        mat = np.random.rand(n, n)
        print("Beginning L1 norm for matrix of size %dx%d"%(n, n))
        start = timer()
        print(m1norm(mat))
        end = timer()
        print("Took %6.5f seconds" % (end-start,))
        print("Beginning Linf norm for matrix of size %dx%d"%(n, n))
        start = timer()
        print(minfnorm(mat))
        end = timer()
        print("Took %6.5f seconds" % (end-start,))
        n *=2

# RESPONSE
# Beginning L1 norm for matrix of size 100x100
# 56.184219261
# Took 0.00013 seconds
# Beginning Linf norm for matrix of size 100x100
# 58.4569858113
# Took 0.00006 seconds
# Beginning L1 norm for matrix of size 200x200
# 113.634217414
# Took 0.00018 seconds
# Beginning Linf norm for matrix of size 200x200
# 110.175226153
# Took 0.00011 seconds
# Beginning L1 norm for matrix of size 400x400
# 219.45716295
# Took 0.00108 seconds
# Beginning Linf norm for matrix of size 400x400
# 216.313387767
# Took 0.00053 seconds
# Beginning L1 norm for matrix of size 800x800
# 427.436100478
# Took 0.00209 seconds
# Beginning Linf norm for matrix of size 800x800
# 425.719609626
# Took 0.00170 seconds
# Beginning L1 norm for matrix of size 1600x1600
# 843.151631626
# Took 0.00592 seconds
# Beginning Linf norm for matrix of size 1600x1600
# 835.986053255
# Took 0.00584 seconds
# Beginning L1 norm for matrix of size 3200x3200
# 1651.73892227
# Took 0.02743 seconds
# Beginning Linf norm for matrix of size 3200x3200
# 1659.31169426
# Took 0.02292 seconds
# Beginning L1 norm for matrix of size 6400x6400
# 3282.54283564
# Took 0.09184 seconds
# Beginning Linf norm for matrix of size 6400x6400
# 3294.88652228
# Took 0.09514 seconds


def p7():
    A = np.matrix([[ 9, -6],
                   [12, -8],
                   [ 0, 20]])
    b = np.matrix([300, 600, 900]).T
    Q, R = qr(A)
    #Remove the 0 vec from R
    R = R[0:2, 0:2]
    #Remove unnecessary bit from solvable part of b
    bs = Q.T.dot(b)[0:2]
    sol = solve(R, bs)
    print(sol)
    x = np.arange(0, 100, .1)
    def f(x):
        return (300 - 9*x)/(-6)
    def g(x):
        return (600 - 12*x)/(-8)
    plt.plot(x, f(x))
    plt.plot(x, g(x))
    plt.plot(x, [900/20]*len(x))
    plt.plot(sol[0][0], sol[1][0], 'ro')
    plt.show()

# RESPONSE:
# [[ 74.]
#  [ 45.]]

def p8():
    A = np.matrix([[  .125,   .25,  .5],
                   [     1,     1,   1],
                   [ 3.375,  2.25, 1.5],
                   [     8,     4,   2],
                   [15.625,  6.25, 2.5]])
    b = np.matrix([[.2, .27, .3, .32, .33]]).T

    B = A.T.dot(A)
    bp = A.T.dot(b)

    print("Normal equation is: ")
    print(B)
    print("by [a, b, c]^T] = ")
    print(bp)
    print("Which is solved by: ")
    print(solve(B, bp))

# RESPONSE
# Normal equation is:
# [[ 320.546875  138.28125    61.1875  ]
#  [ 138.28125    61.1875     28.125   ]
#  [  61.1875     28.125      13.75    ]]
# by [a, b, c]^T] =
# [[ 9.02375]
#  [ 4.3375 ]
#  [ 2.285  ]]
# Which is solved by:
# [[ 0.05239669]
#  [-0.27987013]
#  [ 0.50547816]]


if __name__ == "__main__":
    p7()
