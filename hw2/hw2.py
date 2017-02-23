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
