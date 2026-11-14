import numpy as np

def tridiagonal(a, b, c, d, n):
    P = [0.] * n  
    Q = [0.] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        digit = a[i] * P[i - 1] + b[i]   
        P[i] = -c[i] / digit             
        Q[i] = (d[i] - a[i] * Q[i - 1]) / digit 

    x = [0.] * n          
    x[-1] = Q[-1]  

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]
    return x


def main():
    a = [0, -9, 1, -7, 6]      
    b = [-14, 15, -11, 12, -7]   
    c = [-6, -1, 1, 3, 0]      
    d = [-78, -73, -38, 77, 91]    

    n = 5  

    x = tridiagonal(a, b, c, d, n)
    print("Значения х:", x)

    A = np.array([
        [-14, -6,   0,   0,  0],
        [ -9, 15,  -1,   0,  0],
        [  0,  1, -11,   1,  0],
        [  0,  0,  -7,  12,  3],
        [  0,  0,   0,   6, -7],
    ], dtype=float)

    d = np.array([-78, -73, -38, 77, 91], dtype=float)

    x_ref = np.linalg.solve(A, d)
    print("Значения х (numpy):", x_ref)



if __name__ == '__main__':
    main()
