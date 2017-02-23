import numpy as np

A = [[6, 2, 1, -1],
     [2, 4, 1, 0],
     [1, 1, 4, -1],
     [-1, 0, -1, 3]]

def LUfactor(A):
    if(len(A) != len(A[0])):
        print("Not a square matrix!")
        return
    n = len(A)
    L = np.identity(n)
    U = np.zeros([n, n])
    for j in range(n):
        for i in range(j + 1):
            U[i][j] = A[i][j] - sum([L[i][k]*U[k][j] for k in range(i)])
        if abs(U[i][i]) < 1e-8:
            print("Coefficients are too small to proceed!")
            return
        for i in range(j, n):
            L[i][j] = (A[i][j] - sum([L[i][k]*U[k][j] for k in range(j)]))/U[j][j]

    return (L, U)

if __name__=="__main__":
    L, U = LUfactor(A)
    print("L is: \n")
    print(L)
    print("\nU is: \n")
    print(U)
    print("\nProduct is: \n")
    print(L.dot(U))

### OUTPUT ###
# L is:
#
# [[ 1.          0.          0.          0.        ]
#  [ 0.33333333  1.          0.          0.        ]
#  [ 0.16666667  0.2         1.          0.        ]
#  [-0.16666667  0.1        -0.24324324  1.        ]]
#
# U is:
#
# [[ 6.          2.          1.         -1.        ]
#  [ 0.          3.33333333  0.66666667  0.33333333]
#  [ 0.          0.          3.7        -0.9       ]
#  [ 0.          0.          0.          2.58108108]]
#
# Product is:
#
# [[  6.00000000e+00   2.00000000e+00   1.00000000e+00  -1.00000000e+00]
#  [  2.00000000e+00   4.00000000e+00   1.00000000e+00   0.00000000e+00]
#  [  1.00000000e+00   1.00000000e+00   4.00000000e+00  -1.00000000e+00]
#  [ -1.00000000e+00   5.55111512e-18  -1.00000000e+00   3.00000000e+00]]
