#%%
import numpy as np
import math
from matplotlib import pyplot as plt
# %%

vector = [1,2,3,4,5]

print(vector*3)

# In python the vector multiplied by scalar doesn't work the way we intend it
# We get a result of [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5] which is basically the vector repeated 3 times
# Instead we have to use numpy array to make a vector 




# %%
# Now we get the intended scalar multiplication of vector [ 3  6  9 12 15]
vector = np.array(vector)
print(vector*3)

# %%


vector_2d = np.array([1,2])
scalar_1 = 2
scalar_2 = 0.5
scalar_3 = -1

#%%
vector_2d1 = vector_2d*scalar_1
vector_2d2 = vector_2d*scalar_2
vector_2d3 = vector_2d*scalar_3

plt.plot([0, vector_2d[0]], [0, vector_2d[1]], 'ro', label="original vector" )
plt.plot([0, vector_2d1[0]], [0, vector_2d1[1]], label="positive scalar")
plt.plot([0, vector_2d2[0]], [0, vector_2d2[1]], label="fractional scalar")
plt.plot([0, vector_2d3[0]], [0, vector_2d3[1]], label="negative scalar")
plt.title("scalar multiplications of a vector")
plt.grid()
plt.legend()
plt.show()

# %%
## DOT PRODUCT ##

v1 = np.array([1,3,-4,6])
v2 = np.array([2,-3,5,2])

v_dot = np.dot(v1,v2)

print("DOT PRODUCT : ", v_dot)

# %%
def is_it_orthogonal(v1,v2):
    
    if(len(v1)==len(v2)):
        v_dot = np.dot(v1,v2)
        if(v_dot==0):
            print("The vectors are orthogonal")
        else:
            print("The vectors are not orthogonal")
    else:
        print("please input two vectors of matching lengths")

# %%

is_it_orthogonal(np.array([-4,3,1]), np.array([2,3,-1]))

# %%
## IDENTITY MATRIX ##
# create a 3x3 identity matrix
identity_matrix = np.eye(3)
print(identity_matrix)

# %%
# Zeros and Full np functions to create matrices

zeros_matrix = np.zeros((5,2))
print("Zeros Matrix : \n", zeros_matrix)

full_matrix = np.full((2,3), 9)
print("Full Matrix : \n", full_matrix)


# %%

M = np.array([[1,2,3],
             [2,3,4],
             [3,4,5]])

print("3x3 matrix created in numpy : \n", M)

print("3x3 scalar multiplied matrix created in numpy : \n", M*3)

# %%
vector = np.array([1,2,3], ndmin=2)

Matrix = np.array([[1,2,3],
             [2,6,4],
             [7,4,9]])

# %%
vector_T = vector.T
print("vector : \n", vector)
print("vector Transposed : \n", vector_T)

Matrix_T = Matrix.T
Matrix_T_T = Matrix_T.T

print("Matrix : \n", Matrix)
print("Matrix Transposed : \n", Matrix_T)
print("Matrix Double Transposed : \n", Matrix_T_T)


# %%
## MATRIX MULTIPLICATION ##

## two matrices can be multiplied only when the number of columns of matrix 1 is equal to number of rows of matrix 2

## mat 1 -> AXB and mat 2 -> BXN (B == B)

## Resulting matrix will be of shape AXN



def matrix_multiplication(mat1, mat2):
    if(mat1.shape[1] != mat2.shape[0]):
        print("Invalid shapes for multiplication")
    else:
        return np.matmul(mat1,mat2)


mat1 = np.random.randn(3,4)
mat2 = np.random.randn(4,5)

matrix_multiplied = matrix_multiplication(mat1,mat2)

print(matrix_multiplied.shape)

# we can multiply matrices using the @ symbol not *
matrix_multiplied_method2 = mat1 @ mat2

# We subtract multiplied matrices from two methods to get zeros to confirm its the same
print(matrix_multiplied - matrix_multiplied_method2)

# %%
# Solving for set of linear equations with x matrix of unkowns and A matrix of coefficients
# b matrix with the right hand side values 
# Ax = b 
# The solution for this linear equations will be obtained by x = inverse of A * b

A = np.round(np.random.randn(3,3))
try:
    Ainv = np.linalg.inv(A)

except:
    print("not invertible")

print("A : \n", A)
print("A inverse : \n", Ainv)

A_times_Ainv = A@Ainv

print("multiplication of A and Ainverse to yield Identity matrix : \n", A_times_Ainv)

# %%


# Solve a linear equation
A = [[1, 2], [3, 4]]
b = [5, 6]
x = np.linalg.solve(A, b)

print(x)

# %%
