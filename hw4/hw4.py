import numpy as np
from numpy import *
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

S = np.random.randn(100, 100)
S = vectorize(lambda x: max(2.0, x) - 2.0)(S)
S -= diag(diag(S))
S = S.dot(diag([1.0 / max(1e-10, x) for x in sum(S, 1)]))

## UTILITY FUNCTIONS

def pow_method(matrix, vec, iter=5):
    res = []
    for i in range(iter):
        vec = matrix.dot(vec)
        vec = vec/norm(vec)
        res.append(vec)
    return res

def inv_pow_method(matrix, vec, val, iter=5):
    n = matrix.shape[0]
    inv = np.linalg.inv(matrix - val*np.eye(n))
    return [x.T.dot(matrix).dot(x).A[0][0] for x in pow_method(inv, vec, iter=iter)]


## PROBLEMS

def p2_b():
    mat = np.matrix([[-2, 1, 4],
                     [ 1, 1, 1],
                     [ 4, 1, -2]])
    vec1 = np.matrix([[1, 2, -1]]).T
    vec2 = np.matrix([[1, 1, 1]]).T
    ans1 = pow_method(mat, vec1)
    ans2 = pow_method(mat, vec2)
    print("Results for first vector")
    for x in ans1:
        print(x)
    print()
    print("Results for second vector")
    for x in ans2:
        print(x)
    print()
    print("Results for eig")
    print(np.linalg.eig(mat)[0])
    print(np.linalg.eig(mat)[1])

## Response
# Results for first vector
# [[-0.43643578]
#  [ 0.21821789]
#  [ 0.87287156]]

# [[ 0.80829038]
#  [ 0.11547005]
#  [-0.57735027]]

# [[-0.64483142]
#  [ 0.05862104]
#  [ 0.7620735 ]]

# [[ 0.73561236]
#  [ 0.02942449]
#  [-0.67676337]]

# [[-0.69215012]
#  [ 0.0147266 ]
#  [ 0.72160331]]
#
# Results for second vector
# [[ 0.57735027]
#  [ 0.57735027]
#  [ 0.57735027]]

# [[ 0.57735027]
#  [ 0.57735027]
#  [ 0.57735027]]

# [[ 0.57735027]
#  [ 0.57735027]
#  [ 0.57735027]]

# [[ 0.57735027]
#  [ 0.57735027]
#  [ 0.57735027]]

# [[ 0.57735027]
#  [ 0.57735027]
#  [ 0.57735027]]
#
# Results for eig
# [ -6.00000000e+00   3.00000000e+00   2.77080206e-16]
# [[  7.07106781e-01  -5.77350269e-01   4.08248290e-01]
#  [ -4.70543743e-17  -5.77350269e-01  -8.16496581e-01]
#  [ -7.07106781e-01  -5.77350269e-01   4.08248290e-01]]



def p2_d():
    mat = np.matrix([[-2, 1, 4],
                     [ 1, 1, 1],
                     [ 4, 1, -2]])
    vec1 = np.matrix([[4, 5, 6]]).T
    ans1 = inv_pow_method(mat, vec1, -5.4, iter=5)
    ans2 = inv_pow_method(mat, vec1, 2.5, iter=5)
    ans3 = inv_pow_method(mat, vec1, .0001, iter=5)
    print("Finding largest eig")
    print(ans1)
    print("Finding another eig")
    print(ans2)
    print("Finding last eig")
    print(ans3)

## RESPONSE
# Finding largest eig
# [-4.5546038543897236, -5.9912231656416211, -5.9999551767440735, -5.9999997713087847, -5.9999999988332071]
# Finding another eig
# [2.9991696267933756, 2.9999971264721039, 2.9999999900569945, 2.9999999999655946, 2.9999999999998819]
# Finding last eig
# [2.9404032715438877, 2.9850279481550097, 2.9962520438409799, 0.015470096331001375, 1.7282018546008047e-11]

def p4_b():
    a = np.poly([x for x in range(1, 16)])
    m = np.diag(np.ones(len(a)-2), -1)
    m[:, len(a)-2] = -np.flipud(a[1:])
    print("Results of eig method")
    e = np.linalg.eig(m)[0]
    print(e)
    print("Resuts of root")
    r = np.roots(a)
    print(r)
    print("Error in terms of L2 norm is:")
    print(norm(np.flipud(e) - r))

## RESPONSE
# Results of eig method
# [  1.           2.           3.           4.00000001   4.99999997
#    6.00000016   6.99999932   8.00000224   8.99999481  10.00000797
#   10.99999225  12.0000043   12.99999906  13.9999998   15.00000011]
# Resuts of root
# [ 15.00000008  13.99999929  13.00000283  11.99999361  11.00000924
#    9.99999088   9.00000635   7.99999684   7.00000113   5.99999972
#    5.00000005   3.99999999   3.           2.           1.        ]
# Error in terms of L2 norm is:
# 2.95810084304e-05
#

def p5_a():
    plt.spy(S)
    plt.title("Sparsity matrix")
    plt.savefig("5a.png")

def p5_b():
    eigs = linalg.eigvals(S)
    print(eigs)
    t = np.arange(0.0, 2*pi, .01)
    x = [z.real for z in eigs]
    y = [z.imag for z in eigs]
    plt.plot(x, y, 'ro')
    plt.plot(cos(t), sin(t), 'b', lw=2)
    plt.title("Problem 5 part B")
    plt.savefig("5b.png")

def p5_c():
    t = np.arange(0.0, 2*pi, .01)
    for k in [0.0, .25, .5, .85, .9, 1.0]:
        eigs = linalg.eigvals(k*S + (1-k)/100*ones([100, 100]))
        x = [z.real if abs(z.real) < 1.5 else 0 for z in eigs]
        y = [z.imag if abs(z.imag) < 1.5 else 0 for z in eigs]
        plt.plot(x, y, 'ro')
        plt.plot(cos(t), sin(t), 'b', lw=2)
        plt.title("Eigenvalues with k set to %.2f"%(k))
        plt.savefig("5c%d"%(int(k*100)))

if __name__ == "__main__":
    p5_b()
