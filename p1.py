
A = eval(input("Enter the matrix A as a numpy array : "))
#Q2


def get_eigen(mat):
    M = np.array(mat)
    e, v = np.linalg.eig(M)
    ev = []
    for i in range(len(e)):
        ev.append([e[i], v[i].tolist()])
    return ev

#Q2 - A
mat_a = [[0, 55, -10], 
         [0, 22, 16], 
         [0, -9, -2]]
res_a = get_eigen(mat_a)
for i in res_a:
    print("Eigenvalue:", i[0], "Eigenvector:",i[1])


#Q3
def power_method(A, n, dim=3):
    X = np.ones(dim)
    def normalize(x):
        fac = abs(x).max()
        x_n = x / x.max()
        return fac, x_n
    for i in range(n):
        X = np.dot(A, X)
        lambda_l, X = normalize(X)
    print("Power Method Result:\n\tEigenvalue:", lambda_l, "\n\tEigenVector:", X)

#Q3
def rayleigh_quotient(B, x):
    l=np.dot((B@x),x)
    ans=l/np.dot(x,x)
    return ans


def rayleigh_quotient(B, n, dim = 3):
    def large_eig(x, i, n):
        for j in range(n):
            y = np.matmul(x, i)
            i = y
        a = i
        b = np.matmul(x, a)
        bt = b.transpose()
        b_res = np.matmul(bt, a)
        c = np.matmul(a.transpose(), a)
        return b_res / c
    I = np.ones(dim)
    K = I.transpose()
    for i in range(n):
        e = large_eig(B, K, i)
    print("Rayleigh Quotient Result:\n\tEigenvalue:", e)

#Q3 A
mat_3a = np.array([[1, 2, 0], 
                   [-2, 1, 2], 
                   [1, 3, 1]])
power_method(mat_3a, 10)
rayleigh_quotient(mat_3a, 10)

mat_3a = np.array([[1, 2, 0], 
                   [-2, 1, 2], 
                   [1, 3, 1]])
x=power_method(mat_3a, 10)
ans=rayleigh_quotient(mat_3a, x)


#Q4
def inv_pow_method(A, n, dim=3):
    a_inv = np.linalg.inv(A)
    x = np.ones(dim)
    def normalize(x):
        fac = abs(x).max()
        x_n = x / x.max()
        return fac, x_n
    for i in range(n):
        x = np.dot(a_inv, x)
        lambda_l, x = normalize(x)
    print("Inverse Power Method Result:\n\tEigenvalue:", lambda_l, "\n\tEigenvector:", x)

#Q4
def rayleigh_small(A, n, dim=3):
    a = np.linalg.inv(A)
    rayleigh_quotient(a, n, dim)

#Q4 A
mat_4a = [[5, -10, -5], [2, 14, 2], [-4, -8, 6]]
inv_pow_method(mat_4a, 100)
rayleigh_small(mat_4a, 100)

#Q4 B
mat_4b = [[2, 3, 1], [0, -1, 2], [0, 0, 3]]
x=inv_pow_method(mat_4b, 100)
rayleigh_small(mat_4b, 100,x)



"""# SPARSE MATRIX"""

import numpy as np
import math

# Q1
def Jacobi(A,tol = 1.0e-9): # Jacobi method
    # Find largest off-diagonal element a[k,l]
    def maxElem(A):
        n = len(A)
        Amax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(A[i,j]) >= Amax:
                    Amax = abs(A[i,j])
                    k = i; l = j
        return Amax,k,l
    # Rotate to make A[k,l] = 0 and define the rotation matrix
    def rotate(A,p,k,l):
        n = len(A)
        Adiff = A[l,l] - A[k,k]
        if abs(A[k,l]) < abs(Adiff)*1.0e-36: 
            t = A[k,l]/Adiff
        else:
            phi = Adiff/(2.0*A[k,l])
            t = 1.0/(abs(phi) + math.sqrt(phi**2 + 1.0))
            if phi < 0.0: 
                t = -t
        c = 1.0/math.sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = A[k,l]
        A[k,l] = 0.0
        A[k,k] = A[k,k] - t*temp
        A[l,l] = A[l,l] + t*temp
        for i in range(k): # Case of i < k
            temp = A[i,k]
            A[i,k] = temp - s*(A[i,l] + tau*temp)
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(k+1,l): # Case of k < i < l
            temp = A[k,i]
            A[k,i] = temp - s*(A[i,l] + tau*A[k,i])
            A[i,l] = A[i,l] + s*(temp - tau*A[i,l])
        for i in range(l+1,n): # Case of i > l
            temp = A[k,i]
            A[k,i] = temp - s*(A[l,i] + tau*temp)
            A[l,i] = A[l,i] + s*(temp - tau*A[l,i])
        for i in range(n): # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
    n = len(A)
    maxRot = 5*(n**2) # Set limit on number of rotations
    p = np.identity(n)*1.0 # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop
        Amax,k,l = maxElem(A)
        if Amax < tol: 
            return np.diagonal(A),p
        rotate(A,p,k,l)
    print('Jacobi method did not converge')

# Q1
A = np.array([[1,1.3,2], [1.3,3,1.3], [2,1.3,1]])
print('Eigenvalues and Eigenvectors of matrix:\n', A)
print('is\n', Jacobi(A,tol = 1.0e-9)) # set the tolerance as your wish

# Q2
def is_sparse(mat):
    n,m = len(mat), len(mat[0])
    nz = 0
    for i in range(n):
        for j in range(m):
            if mat[i][j] != 0:
                nz += 1
    if nz < (n * m) / 2:
        return 1, nz, (m*n - nz)
    else:
        return 0, nz, (m*n - nz)

# Q2
M2 = [[0, 0, 3, 0, 4], [0, 0, 5, 7, 0], [0, 0, 0, 0, 0], [0, 2, 6, 0, 0]]
res2 = is_sparse(M2)
if res2[0] == 1:
    print("Matirx is Sparse")
else:
    print("Matrix is Dense")
print("No of Non-Zero Elements: {}\nNo of Zero Elements: {}".format(res2[1], res2[2]))

# Q3
class sparse_mat:
    def __init__(self, mat):
        self.mat = mat
        self.m = len(mat)
        self.n = len(mat[0])

    def get_comp(self, arr=1):
        array = []
        for i in range(self.m):
            for j in range(self.n):
                if self.mat[i][j] != 0:
                    array.append([i, j, self.mat[i][j]])
        if arr == 1:
            return np.array([x for x in array]).T
        else:
            return array

    def get_dict(self):
        dict1 = {}
        for i in range(self.m):
            for j in range(self.n):
                if self.mat[i][j] != 0:
                    dict1[(i, j)] = self.mat[i][j]
        return dict1

# Q3
M3 = sparse_mat([[0, 0, 3, 0, 4], [0, 0, 5, 7, 0], [0, 0, 0, 0, 0], [0, 2, 6, 0, 0]])
comp3 = M3.get_comp()
print(comp3)
print(M3.get_comp(0))
print(M3.get_dict())

# Q4
def sparse_to_mat(sparse):
    max_j = max(max(sparse[1]), max(sparse[0])) + 1
    mat = [[0 for i in range(max_j)] for j in range(max_j)]
    for k in range(len(sparse[0])):
        i, j = sparse[0][k], sparse[1][k]
        mat[i][j] = sparse[2][k]
    return mat

# Q4
print(sparse_to_mat(comp3))

# Q5
class Sparse_Ops:
    def __init__(self, sparse, m, n):
        self.sparse = sparse
        self.m = m
        self.n = n
    def add_sparse(self, s1):
        if self.m != s1.m or self.n != s1.n:
            print("Cannot Add!")
            return
        res = {}
        for i in range(self.m):
            for j in range(self.n):
                res[(i, j)] = 0
        for a in self.sparse.keys():
            if a in s1.sparse.keys():
                res[a] = self.sparse[a] + s1.sparse[a]
        return res
    def tra_sparse(self):
        res = {}
        for a in self.sparse.keys():
            res[a[::-1]] = self.sparse[a]
        return res
    def mul_sparse(self, s1):
        if self.m != s1.m or self.n != s1.n:
            print("Cannot Multiply!")
            return
        res = {}
        for a in self.sparse.keys():
            if a in s1.sparse.keys():
                res[a] = self.sparse[a] * s1.sparse[a]
        return res

# Q5
M5_1 = sparse_mat([[0, 0, 3, 0, 4], [0, 0, 5, 7, 0], [0, 0, 0, 0, 0], [0, 2, 6, 0, 0]]).get_dict()
M5_2 = sparse_mat([[1, 0, 3, 0, 9], [0, 2, 5, 7, 0], [0, 0, 3, 0, 0], [0, 3, 6, 0, 1]]).get_dict()
M1 = Sparse_Ops(M5_1, m=4, n=5)
M2 = Sparse_Ops(M5_2, m=4, n=5)
print(M1.tra_sparse())
print(M1.add_sparse(M2))
print(M1.mul_sparse(M2))

"""# INTERPOLATION"""


import numpy as np

# Q1
def linear_inter(fx0, fx1, x0, x1, x):
    return fx0 + (((fx1 - fx0) / (x1 - x0)) * (x - x0))

# Q1 A
x0 = -1
fx0 = -8
x1 = 2
fx1 = 1
x = 0
fx = linear_inter(fx0, fx1, x0, x1, x)
print("f({}) =".format(x), fx)



# Q2
def quadratic_inter(fx0, fx1, fx2, x0, x1, x2, x):
    b0 = fx0
    b1 = (fx1 - fx0) / (x1 - x0)
    b2 = (((fx2 - fx1) / (x2 - x1)) - b1) / (x2 - x0)
    return b0 + (b1 * (x - x0)) + (b2 * (x - x0) * (x - x1))

# Q2 A
x0 = 0
fx0 = 659
x1 = 2
fx1 = 705
x2 = 3
fx2 = 729
x = 2.75
fx = quadratic_inter(fx0, fx1, fx2, x0, x1, x2, x)
print("f({}) =".format(x), fx)



# Q3
def divided_diff_table(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
def product_term(i, val, x):
    pro = 1
    for j in range(i):
        pro *= val - x[j]
    return pro
def divided_diff(val, x, y, n):
    sum = y[0][0]
    for i in range(1, n):
        sum += product_term(i, val, x) * y[0][i]
    return sum


# Q3 A
n = 4
y = [[0 for i in range(10)] for j in range(10)]
x = [654, 658, 659, 661]
y[0][0] = 2.8156
y[1][0] = 2.8182
y[2][0] = 2.8189
y[3][0] = 2.8202
y = divided_diff_table(x, y, n)
val = 656
print("f({}) =".format(val), divided_diff(val, x, y,n))



# Q4 A
n = 5
x = [45, 50, 55, 60, 65]
y = [[0 for i in range(10)] for j in range(10)]
y[0][0] = 114.84
y[1][0] = 96.16
y[2][0] = 83.32
y[3][0] = 74.48
y[4][0] = 68.48
y = divided_diff_table(x, y, n)
val = 46
print("f({}) =".format(val), divided_diff(val, x, y,n))




# Q5
def forward_diff_table(y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = y[j + 1][i - 1] - y[j][i - 1]
    return y
def calc_u(u, n, f=1):
    temp = u
    for i in range(1, n):
        temp *= u - i
    return temp
def get_factorial(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f
def forward_diff(val, x, y, n):
    sum = y[0][0]
    u = (val - x[0]) / (x[1] - x[0])
    for i in range(1, n):
        sum += (calc_u(u, i) * y[0][i]) / get_factorial(i)
    return sum
def u_cal(u, n):
	temp = u;
	for i in range(1, n):
		temp = temp * (u + i);
	return temp
def backward_diff(value, x, y, n):
    for i in range(1, n):
        for j in range(n - 1, i+1, -1):  
            y[j][i] = y[j][i - 1] - y[j - 1][i - 1]
    sum = y[n - 1][0];
    u = (value - x[n - 1]) / (x[1] - x[0]);
    for i in range(1, n):
        sum = sum + (u_cal(u, i) * y[n - 1][i]) / get_factorial(i);  
    return y, sum

# Q5 A
n = 7
x = [i for i in range(3, 10)]
y = [[0 for i in range(n)] for j in range(n)]
y[0][0] = 4.8
y[1][0] = 8.4
y[2][0] = 14.5
y[3][0] = 23.6
y[4][0] = 36.2
y[5][0] = 52.8
y[6][0] = 73.9
y1 = forward_diff_table(y, n)
val = 3.5
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)
val = 8.5
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)


# Q5
def lagrange_interpolate(f, x1):
    res = 0.0
    n = len(f)
    for i in range(n):
        term = f[i][1]
        for j in range(n):
            if j != i:
                term *= (x1 - f[j][0]) / (f[i][0] - f[j][0])
        res += term
    return res

# Q5 A
f = [[5, 12], [6, 13], [9, 14], [11, 16]]
val = 10
print("Lagrange's Interpolation: f({}) =".format(val), lagrange_interpolate(f, val))