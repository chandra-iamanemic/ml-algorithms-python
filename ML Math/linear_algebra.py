#%% 
import numpy as np
import math
from matplotlib import pyplot as plt

#%%
# Vectors: A vector is a one-dimensional array of numbers. It can be represented as a list or a NumPy array.

# Create a vector
vector = [1, 2, 3]

# Print the vector
print(vector)


#%%

# Matrices: A matrix is a two-dimensional array of numbers. It can be represented as a list of lists or a NumPy array.

# Create a matrix
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

# Print the matrix
print(matrix)

# %%

# Linear equations: A linear equation is an equation of the form Ax = b, where A is a matrix, x is a vector, and b is a vector.

# Solve a linear equation
A = [[1, 2], [3, 4]]
b = [5, 6]
x = np.linalg.solve(A, b)

print(x)

#%%

# Eigenvalues and eigenvectors: Eigenvalues and eigenvectors are important concepts in linear algebra. They are used to solve problems such as finding the direction of maximum change and the natural frequencies of a system.

# Calculate the eigenvalues and eigenvectors of a matrix
A = [[2, 1], [1, 2]]
w, v = np.linalg.eig(A)

print(w)
print(v)

# %%

# Linear transformations: A linear transformation is a function that maps vectors to vectors. It can be represented as a matrix.

# Create a linear transformation
T = np.array([[1, 2], [3, 4]])

# Apply the linear transformation to a vector
x = np.array([1, 2])
y = T @ x

print(y)

# %%

# Norms: A norm is a way of measuring the size of a vector. There are many different norms, but the most common ones are the 1-norm, the 2-norm, and the infinity norm.

# Calculate the 1-norm of a vector
vector = [1, 2, 3]
norm1 = np.linalg.norm(vector, 1)

print(norm1)

# Calculate the 2-norm of a vector

norm2 = np.linalg.norm(vector, 2)

print(norm2)

# Calculate the inf-norm of a vector
norm_inf = np.linalg.norm(vector, np.inf)

print(norm_inf)

# %%

# Determinants: The determinant of a matrix is a number that describes the properties of the matrix. It can be used to solve linear equations, find eigenvalues and eigenvectors etc.

# Calculate the determinant of a matrix
A = np.array([[1, 2], [3, 4]])
determinant = np.linalg.det(A)

print(determinant)

# %%

# Rank: The rank of a matrix is the number of linearly independent rows or columns in the matrix. It can be used to determine if a system of linear equations has a unique solution.

# Calculate the rank of a matrix
A = np.array([[1, 2], 
              [3, 4], 
              [5, 6]])
rank = np.linalg.matrix_rank(A)

print(rank)

# %%

# Orthogonality: Two vectors are orthogonal if their dot product is zero. Orthogonality is used in many different areas of linear algebra, including least squares regression, principal component analysis, and singular value decomposition.

# Check if two vectors are orthogonal
vector1 = np.array([1, 2])
vector2 = np.array([3, 4])

is_orthogonal = np.allclose(vector1.dot(vector2), 0)

print(is_orthogonal)

# %%


# Span: The span of a set of vectors is the set of all vectors that can be created by linear combinations of the vectors in the set.
# Linear independence: A set of vectors is linearly independent if no vector in the set can be written as a linear combination of the other vectors in the set.


vectors = np.array([[1, 2], 
                    [3, 4], 
                    [5, 6]])
is_linearly_independent = np.linalg.lstsq(vectors, vectors)[0].all()

print(is_linearly_independent)


#%%
# Dimension: The dimension of a vector space is the number of vectors in a basis for the vector space.
# OR the rank
vectors = np.array([[1, 2], 
                    [3, 4], 
                    [5, 6]])
dimension = np.linalg.matrix_rank(vectors)

print(dimension)

#%%
# Subspace: A subspace of a vector space is a vector space that is contained within the original vector space.
# Inner product: An inner product is a way of measuring the similarity between two vectors. It can be used to define norms and distance metrics.

import numpy as np

vector1 = np.array([1, 2])
vector2 = np.array([3, 4])
inner_product = np.dot(vector1, vector2)

print(inner_product)

# %%
