import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly

## Utility Functions

def trap(f, a, b, m):
    h = (b - a)/m
    s = (f(a) + f(b))/2
    for i in range(1, m):
        s += f(a+i*h)
    return s*h

def simp(f, a, b, m):
    h = (b-a)/(2*m)
    s = f(a) + f(b)
    c = 4
    for i in range(1, 2*m):
        s += c*f(a+i*h)
        c = 8//c
    return s*h/3

def quad_01(f):
    s = 0
    s+= f(1/2)*4/9
    s+= f(1/2 - np.sqrt(3/5)/2)*5/18
    s+= f(1/2 + np.sqrt(3/5)/2)*5/18
    return s

## Problems:

def p1_a():
    f = lambda x: np.cos(3*np.pi*x)**4
    g = lambda x: np.sqrt(x)
    m_arr = [x for x in range(10, 1000, 5)]
    cos_res_trap = [abs(3/8 - trap(f, 0, 1, m)) for m in m_arr]
    sqt_res_trap = [abs(2/3 - trap(g, 0, 1, m)) for m in m_arr]
    cos_res_simp = [abs(3/8 - simp(f, 0, 1, m)) for m in m_arr]
    sqt_res_simp = [abs(2/3 - simp(g, 0, 1, m)) for m in m_arr]
    #Cos first
    plt.loglog(m_arr, cos_res_trap, "-", label="Trapezoidal error")
    plt.loglog(m_arr, cos_res_simp, "--", label="Simpson's error")
    plt.title("Error in computing cosine integral")
    plt.xlabel("m value (log)")
    plt.ylabel("Error (log)")
    plt.legend()
    plt.savefig("p1a-cos.pdf")
    plt.clf()

    plt.loglog(m_arr, sqt_res_trap, "-", label="Trapezoidal error")
    plt.loglog(m_arr, sqt_res_simp, "--", label="Simpson's error")
    plt.xlabel("m value (log)")
    plt.ylabel("Error (log)")
    plt.title("Error in computing sqrt integral")
    plt.legend()
    plt.savefig("p1a-sqt.pdf")

def p1_b():
    g = lambda x: np.sqrt(x)
    m_arr = [x for x in range(10, 1000, 5)]
    one = np.ones(len(m_arr))
    m_log = np.log10(m_arr)
    trap_err = np.log10([abs(2/3 - trap(g, 0, 1, m)) for m in m_arr])
    print(trap_err[-10:])
    simp_err = np.log10([abs(2/3 - simp(g, 0, 1, m)) for m in m_arr])
    print(simp_err[-10:])

    A = np.transpose([one, m_log])
    print("Coefficients based on simpson's error")
    print(lstsq(A, simp_err)[0])
    print("Coefficients based on trap error")
    print(lstsq(A, trap_err)[0])

def p3_c():
    ks = [k for k in range(8)]
    simps = [simp(lambda x: x**k, 0, 1, 1) for k in range(8)]
    quad = [quad_01(lambda x: x**k) for k in range(8)]
    print("Simpsons values:")
    for i in range(8): print("%d\t%f" %(i, simps[i]))
    print("Quad values: ")
    for i in range(8): print("%d\t%f" %(i, quad[i]))
    simp_err = [abs(1/(k+1) - simps[k]) for k in range(8)]
    quad_err = [abs(1/(k+1) - quad[k]) for k in range(8)]
    print("Simpsons' errors:")
    for i in range(8): print("%d\t%f" %(i, simp_err[i]))
    print("Quad errors:")
    for i in range(8): print("%d\t%f" %(i, quad_err[i]))
    plt.plot(ks, simp_err, "ro", markersize=10)
    plt.title("Simpsons error on $\int_0^1x^k\,dx$")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.savefig("p3c_simp.pdf")
    plt.clf()
    plt.plot(ks, quad_err, "bo", markersize=10)
    plt.title("Quadrature error on $\int_0^1x^k\,dx$")
    plt.xlabel("k")
    plt.ylabel("Error")
    plt.savefig("p3c_quad.pdf")

#Response
# Simpsons values:
# 0	1.000000
# 1	0.500000
# 2	0.333333
# 3	0.250000
# 4	0.208333
# 5	0.187500
# 6	0.177083
# 7	0.171875
# Quad values:
# 0	1.000000
# 1	0.500000
# 2	0.333333
# 3	0.250000
# 4	0.200000
# 5	0.166667
# 6	0.142500
# 7	0.123750
# Simpsons' errors:
# 0	0.000000
# 1	0.000000
# 2	0.000000
# 3	0.000000
# 4	0.008333
# 5	0.020833
# 6	0.034226
# 7	0.046875
# Quad errors:
# 0	0.000000
# 1	0.000000
# 2	0.000000
# 3	0.000000
# 4	0.000000
# 5	0.000000
# 6	0.000357
# 7	0.001250


def p4_a():
    l0 = poly.Polynomial([1])
    l1 = poly.Polynomial([-1, 1])
    l2 = poly.Polynomial([2, -4, 1])
    l3 = poly.Polynomial([-6, 18, -9, 1])
    i = 0
    x = np.arange(0, 10, .01)
    for l in [l0, l1, l2, l3]:
        plt.plot(x, l(x))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Laguerre polynomial L%d on [0, 10]"%(i,))
        plt.savefig("p4a"+str(i)+".pdf")
        i+=1
        plt.clf()

def p4_b():
    l2 = poly.Polynomial([2, -4, 1])
    l3 = poly.Polynomial([-6, 18, -9, 1])
    print("Roots of l2: ")
    print(l2.roots())
    print("Roots of l3: ")
    print(l3.roots())

## Results
# Roots of l2:
# [ 0.58578644  3.41421356]
# Roots of l3:
# [ 0.41577456  2.29428036  6.28994508]

def p4_c():
    x2 = [.585786, 3.41421]
    W2 = [.853553, .146447]
    x3 = [.415775, 2.29428, 6.28995]
    W3 = [.711093, .278518, .0103893]
    x4 = [.322548, 1.74576, 4.53662, 9.39507]
    W4 = [.603154, .357419, .0388879, .000539295]

    i = 1
    for f, r in [(lambda x: np.exp(-x), 1/2),
                (lambda x: np.exp(-x**2 + x), np.sqrt(np.pi)/2)]:
        print("N=2 result for integral %d"%(i,))
        r2 = np.dot([f(x) for x in x2], W2)
        print("\t",r2)
        print("N=3 result for integral %d"%(i,))
        r3 = np.dot([f(x) for x in x3], W3)
        print("\t",r3)
        print("N=4 result for integral %d"%(i,))
        r4 = np.dot([f(x) for x in x4], W4)
        print("\t",r4)
        print()
        print("Error results for integral %d"%(i,))
        print("\tN=2 error: ", abs(r-r2))
        print("\tN=3 error: ", abs(r-r3))
        print("\tN=4 error: ", abs(r-r4))
        print()
        i+=1
#Result
# N=2 result for integral 1
# 	 0.479964224493
# N=3 result for integral 1
# 	 0.497302926217
# N=4 result for integral 1
# 	 0.499655676123
#
# Error results for integral 1
# 	N=2 error:  0.0200357755071
# 	N=3 error:  0.00269707378286
# 	N=4 error:  0.000344323877425
#
# N=2 result for integral 2
# 	 1.0879862911
# N=3 result for integral 2
# 	 0.920904172616
# N=4 result for integral 2
# 	 0.847679138611
#
# Error results for integral 2
# 	N=2 error:  0.201759365651
# 	N=3 error:  0.034677247163
# 	N=4 error:  0.0385477868413


if __name__=="__main__":
    p4_c()
