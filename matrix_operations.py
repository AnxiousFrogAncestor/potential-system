from collections import deque
import math
from logging import raiseExceptions
from itertools import chain
import copy

def shape(matrix):
    """Calculates shape of arbitrary matrix size, 
    but requires same length across dimension.
    e.g.: 2 rows with equal length."""
    assert type(matrix) == list
    mat = list(matrix)
    ls = []
    ls.append(len(mat))
    first_val = mat[0]
    while type(first_val) == list:
        ls.append(len(mat[0]))
        mat = flatten_matrix(mat, 1)
        first_val = mat[0]
    return tuple(ls)


def vector(len):
    """ Returns an empty 1-d list of len(x)."""
    return len*[0]

def product_reduce(vector):
    """Reduces a vector to a product."""
    z = 0
    for val in vector:
        print("Multiplying", val)
        z*=val
    return z

def add_dimension(matrix_list):
    """Adds a dummy dimension in matrix."""
    return [matrix_list]

def flatten_matrix(matrix, levels):
    """Flattens a level in an n-d matrix."""
    mat = matrix
    ls = []
    for i in range(levels):
        for row in mat:
            ls+=row
            mat=ls
    return mat

def reshape_matrix(matrix, num_column):
    """Creates a row x column matrix from a 1-d (row, column) matrix."""
    return  [matrix[el:el+num_column] for el in range(0,len(matrix),num_column)]

def empty_matrix(num_row, num_column):
    "Creates empty matrix with num_row, num_column."
    return reshape_matrix(num_row*num_column*[0], num_column)

def matrix(shape=[], reshape=False, in_matrix=[]):
    "Forms a shape-sized matrix from an equivalent-sized 1-d matrix. OR convert matrix into another shape."
    if not reshape:
        mat = vector(product_reduce(shape))
    else:
        mat = flatten_matrix(in_matrix, len(shape))
    i=0
    while i < len(shape):
        out = reshape_matrix(mat, shape[i+1])
        mat = out
        i+=1
    return mat


def concatenate(matrix_list):
    "Joins matrix_list values in a single list."
    return matrix_list.join(",")

def repeat_matrix(matrix, times):
    """Repeats a matrix across a dimension."""
    ls = deque([])
    for i in range(times):
        ls.append(matrix)
    return ls

def vector_add(v1, v2):
    """ Adds two vectors."""
    z = vector(len(v1))
    if len(v1)!=len(v2):
        raise Exception("Vectors need same to be length.")
    idx = 0
    for x, y in zip(v1, v2):
        z[idx] = x+y
        idx+=1
    return z

def vector_subtract(v1, v2):
    """ Subtract two vectors."""
    print("input_subtract", v1, v2)
    if len(v1)!=len(v2):
        raise Exception("Vectors need to be same length.")
    z = vector(len(v1))
    idx = 0
    for x, y in zip(v1, v2):
        z[idx] = x-y
        idx+=1
    return z

def vector_el_multiply(v1, v2):
    """ Element-wise multiply vectors."""
    print("input_multiply", v1, v2)
    if len(v1)!=len(v2):
        raise Exception("Vectors need same length.")
    z = vector(len(v1))
    idx = 0
    for x, y in zip(v1, v2):
        z[idx] = x*y
        idx+=1
    return z

def vector_el_divide(v1, v2):
    """ Element-wise divide vectors."""
    #TODO [X]
    z = vector(len(v1))
    if len(v1)!=len(v2):
        raise Exception("Vectors need same length.")
    #if any(v2) == 0:
        #raise Exception("Division by zero")
    idx = 0
    for x, y in zip(v1, v2):
        try:
            z[idx] = x/y
            idx+=1
        except:
            print(x, y)
            print("Division by zero")
            print("Replaced value with zero.")
            z[idx] = 0
            idx+=1
            continue
        finally:
            print("loop done.")
    print("result of division",z)
    return z

def dot_product(v1, v2):
    """Dot product between two vectors."""
    if len(v1)!=len(v2):
        raise Exception("Vectors need same length.")
    z = 0
    for x, y in zip(v1, v2):
        z+=x*y
    return z

def matrix_product_reduce(matrix_list):
    """Applies matrix product list-wise sequentially and returns a matrix."""
    res = matrix_list[0]
    for val in matrix_list[0:]:
        out = matrix_product(res, val)
        res = out
    return res

def sum_reduce(vector):
    """Reduces a vector to a sum."""
    z = 0
    for val in vector:
        z+=val
    return z

def product_reduce(*args):
    """Reduces a vector to a product."""
    print("len of args", len(*args))
    print(type(*args))
    x = list(*args)
    z = x[0]
    for i, val in enumerate(x):
        z*=val
    return z

def max_reduce(vector):
    """Reduces a vector to a max."""
    curr_max = 0
    curr_ix = 0
    for i, val in enumerate(vector):
        if val > curr_max:
            curr_max=val
            curr_ix=i
    return curr_max, curr_ix

def min_reduce(vector):
    """Reduces a vector to a min."""
    curr_min = 0
    curr_ix = 0
    for i, val in enumerate(vector):
        if val < curr_min:
            curr_min=val
            curr_ix=i
    return curr_min, curr_ix

def mean_reduce(vector):
    """Reduces a vector to a mean or average."""
    return sum_reduce(vector)/len(vector)

def standard_deviation_reduce(vector):
    """Reduces a vector to standard deviation."""
    mean = mean_reduce(vector)
    sigma_squared = 0
    for val in vector:
        sigma_squared+=(val-mean)**2
    return math.sqrt(sigma_squared)/len(vector)

def cosine_similarity(v1, v2):
    """Calculates the cosine distance between two vectors."""
    norm_v1=standard_deviation_reduce(v1)
    norm_v2=standard_deviation_reduce(v2)
    return dot_product(v1, v2)/(norm_v1*norm_v2)

def median_reduce(vector):
    """Calculates median of vector."""
    sorted_vector = vector.sort()
    return sorted_vector[math.floor(len(sorted_vector/2))]

def transpose_vector(vector):
    """Reverses/transposes a vector: x^T."""
    return vector[::-1]

def roll(vector, diff):
    """Rolls the vector, shifts the vector by diff to the left."""
    return vector[:diff]+vector[diff:]

def outer_product(v1, v2):
    """Outer product of two vectors v1^T.dot(v2). Returns a matrix by
    multiplying each element from the two vectors."""
    if len(v1)!=len(v2):
        raise Exception("Vectors need same length.")
    flattened_vec = vector(v1*v2)
    for i in range(len(v2)):
        flattened_vec[i:i+v2] = dot_product(roll(v1, i), v2)
    return reshape_matrix(flattened_vec,len(v2))

def identity_matrix(n):
    """Creates a square identity matrix size nxn."""
    matrix = vector(n*n)
    matrix_chunk = [matrix[el:el+n] for el in range(0,len(matrix),n)]
    j = 0
    for i, elem in enumerate(chain(matrix_chunk)):
        elem[j] = 1
        j+=1
    return matrix_chunk

def vector_equality(v1, v2):
    """Compares vectors, equals if all scalars are the same."""
    out = 0
    for x, y in zip(v1, v2):
        if x != y:
            out = False
            break
        else:
            out=True
    return out
def matrix_equality(m1, m2):
    """Compares 2d matrices, equals if all vectors are the same."""
    out = 0
    for x, y in zip(m1, m2):
        res = vector_equality(x, y)
        if res == False:
            out = res
            break
        else:
            out=True
    return out
    
def vector_cross_product(v1, v2, z=[1,1,1]):
    """Calculates cross-product of two vectors with respect to (z) in 3-dimensions."""
    #TODO Use Determinant to calculate cross-product [X]
    # src: https://en.wikipedia.org/wiki/Cross_product 
    assert len(v1) == 3
    assert len(v2) == 3
    assert len(z) == 3
    i, j, k = z
    def _2d_det(mat):
        print("input_mat", mat)
        a = 0
        b = 0
        c = 0
        d = 0
        for i, row in enumerate(mat):
            if i == 0:
                a = row[0]
                b = row[1]
            if i == 1:
                c = row[0]
                d = row[1]
        res = a*d-b*c
        print(res)
        return res
    det_i = i*_2d_det([v1[1:], v2[1:]])
    det_j = -j*_2d_det([[v1[0], v2[0]], [v1[2], v2[2]]])
    det_k = k*_2d_det([v1[:2], v2[:2]])
    return [det_i, det_j, det_k]


def create_grid(n,m):
    """Creates an n x m grid numbered from 1 to n x m."""
    ls = range(1,n*m)
    res = [ls[i:i+m] for i in range(0, ls, m)]
    return res

def scalar_broadcast(s, v, operation):
    """Applies a scalar operation on the vector or matrix."""
    assert type(v) == list
    assert (type(s) == int or type(s) == float)
    new_vec = len(v)*[0]
    for i, val in enumerate(v):
        if operation == "+":
            new_vec[i] = val+s
        elif operation == "-":
            new_vec[i] = val-s
        elif operation == "*":
            new_vec[i] = val*s
            print("multiplying", val*s)
            print(new_vec[i])
        elif operation == "/":
            new_vec[i] = val/s
    print("result of scalar broadcast", new_vec)
    return new_vec

def vector_broadcast(v, m, operation, in_data=False):
    """Applies a vector operation on the flattened matrix.
    If not, then set in_data==True, which means 
    input matrix is 2d."""
    if in_data == False:
        vector_chunks = [m[el:el+len(v)] for el in range(0,len(m),len(v))]
        #flatten one level
        vector_chunks = flatten_matrix(vector_chunks,1)
        print("Vector_chunks", vector_chunks)
        print("len_vector_chunks", len(vector_chunks))
        print("Vector_chunks", vector_chunks)
        if len(vector_chunks)%len(v)!=0:
            raise Exception("Vector size must be multiple of matrix size.")
    else:
        vector_chunks=m
    z = []
    print("Vector_chunks", vector_chunks)
    if in_data == True:
        for val in vector_chunks:
            print(val)
            if operation == "+":
                z.append(vector_add(val,v))
            elif operation == "-":
                z.append(vector_subtract(val,v))
            elif operation == "*":
                z.append(vector_el_multiply(val,v))
            elif operation == "/":
                z.append(vector_el_divide(val,v))
            elif operation == "@":
                z.append(dot_product(v,val))
            elif operation == "(x)":
                z.append(outer_product(v, val))
            elif operation == "x":
                z.append(vector_cross_product(v, val))
    else:
        if operation == "+":
            z = vector_add(v,m)
        elif operation == "-":
            z = vector_subtract(v,m)
        elif operation == "*":
            z = vector_el_multiply(v,m)
        elif operation == "/":
            z = vector_el_divide(v,m)
        elif operation == "@":
            z = dot_product(v,m)
        elif operation == "(x)":
            z = outer_product(v,m)
        elif operation == "x":
            z = vector_cross_product(v,m)
    return z

        
def diagonal(matrix):
    """ Finds the diagonal values of a 2d matrix object. """
    j = 0
    diag = []
    for i, elem in enumerate(chain(matrix)):
        diag.append(elem[j])
        j+=1
    return diag

def trace(matrix):
    """Finds the sum of diagonal elements in a 2d matrix object."""
    return sum_reduce(diagonal(matrix))

def transpose_matrix(matrix):
    """Transposes a 2d matrix."""
    res = []
    z = shape(matrix)
    c = z[-1]
    r = z[-2]
    print("C", c)
    mat = flatten_matrix(matrix,1)
    print(mat)
    i = 0
    while c > 0 and i<=r:
        col_vals = mat[i::c]
        i=i+1 
        print(col_vals)
        res.append(col_vals)
    return res[:c]


def matrix_product(m1, m2):
    """Matrix product of two matrices."""
    t_m2 = transpose_matrix(m2)
    res = deque([])
    for row, row_2 in zip(t_m2, m1):
        res.append(dot_product(row,row_2))
    return res

def matrix_add(m1, m2):
    """Matrix sum of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(vector_add(row,row_2))
    return res

def matrix_subtract(m1, m2):
    """Matrix difference of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(vector_subtract(row,row_2))
    return res

def matrix_el_divide(m1, m2):
    """Matrix division of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(vector_el_divide(row,row_2))
    return res

def matrix_el_multiply(m1, m2):
    """Matrix elementwise multiply of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(vector_el_multiply(row,row_2))
    return res

def matrix_cross_product(m1, m2):
    """Matrix cross_product of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(vector_cross_product(row,row_2))
    return res

def matrix_outer_product(m1, m2):
    """Matrix outer_product of two n-d matrices."""
    res = deque([])
    for row, row_2 in zip(m1, m2):
        res.append(outer_product(row,row_2))
    return res

def matrix_broadcast(m1, m2, operation):
    """Applies a matrix operation on a n-d matrix."""
    z = 0
    if operation == "+":
        z = matrix_add(m1,m2)
    elif operation =="-":
        z = matrix_subtract(m1,m2)
    elif operation =="/":
        z = matrix_el_divide(m1, m2)
    elif operation =="*":
        z = matrix_el_multiply(m1,m2)
    elif operation =="@":
        z = matrix_product(m1,m2)
    elif operation =="x":
        z = matrix_cross_product(m1,m2)
    elif operation =="(x)":
        z = matrix_outer_product(m1,m2)
    return z

def Gauss_Jordan_elimination(matrix, calc_determinant=False):
    """Does Gauss-Jordan elimination of an n-d matrix."""
    """TODO backward substitution [X]"""
    """https://en.wikipedia.org/wiki/Gaussian_elimination"""
    shape_mat = shape(matrix)
    #matrix = list(matrix)
    print("Matrix", matrix)
    print(shape_mat)
    def swap_row(matrix, i, j):
        temp = matrix[i]
        matrix[i] = matrix[j]
        matrix[j] = temp
        return matrix
    # col, row
    c = shape_mat[-1]
    r = shape_mat[-2]
    #determinant_coeff = 1
    i = 0
    def extract_1st_element(matrix_2d, ix):
        print("input matrix", matrix_2d)
        first_els = []
        for i, val in enumerate(matrix_2d):
            print("VAL", val)
            first_els.append(val[ix])
        return first_els
    fin_result = []
    mat = flatten_matrix(matrix, 1)
    while i<r:
        #chunk flattened matrix into col-width
        print("curr_mat", mat)
        print("input: first element of rows", mat[i::c])
        col_vals = mat[i::c]
        pivot_val_max, imax = max_reduce(col_vals)
        pivot_val_min, imin = min_reduce(col_vals)
        pivot_val = 0
        ix = 0
        if abs(pivot_val_max)>=abs(pivot_val_min):
            pivot_val = pivot_val_max
            ix = imax
        else:
            pivot_val= pivot_val_min
            ix = imin
        print("current C", c)
        r_mat = reshape_matrix(mat, c)
        print("tall_matrix", r_mat)
        print("1st elements of rows", col_vals)
        pivot_row = r_mat[ix]
        print("pivot_row", pivot_row)
        if calc_determinant == False:
            if abs(pivot_val) > 1:
                pivot_row = scalar_broadcast(pivot_val, pivot_row, "/") # TODO
                #determinant_coeff*=pivot_val
                if pivot_val < 0:
                    pivot_row = scalar_broadcast(-1, pivot_row, "*")
            elif abs(pivot_val) < 1 and pivot_val!=0:
                print("pivot val", pivot_val)
                pivot_row = scalar_broadcast(1/abs(pivot_val), pivot_row,"*")
                #determinant_coeff*=1/abs(pivot_val)
                if pivot_val < 0:
                    pivot_row = scalar_broadcast(-1, pivot_row, "*")
                    #determinant_coeff*=(-1)
        print("remaining rows", r_mat[1:])
        print("piv_row", pivot_row) 
        fin_result.append(pivot_row)
        swap_row(r_mat, ix, 0)
        #determinant_coeff=determinant_coeff*(-1)
        r_mat[0] = pivot_row
        print("ROW_SWAPPED_MAT", r_mat)
        pivot_coeffs = vector_broadcast(pivot_row, r_mat[1:], "/", in_data=True)
        for ind, x in enumerate(col_vals):
            if x!=pivot_val:
                col_vals[ind]=0
        print("Replaced first elements of remaining rows", col_vals)
        print(pivot_coeffs)
        pivot_coeffs = extract_1st_element(pivot_coeffs, i)
        print("Extracted 1st elements of pivot_coeff matrix", pivot_coeffs)
        mat2 = copy.deepcopy(mat)
        i=i+1
        print("pivot coeff", pivot_coeffs)
        #weight columns
        print("Copied flattened matrix", mat2)
        rrow, rcol = shape(r_mat)
        r2_mat = reshape_matrix(matrix=mat2, num_column=rcol)
        print("Reshaped copied matrix", r2_mat)
        pivot_res = []
        for el in pivot_coeffs:
            pivot_res.append(scalar_broadcast(v=pivot_row, s=el, operation="*"))
        print("multiplied pivot rows", pivot_res)
        r2_mat[1:] = matrix_broadcast(r2_mat[1:], pivot_res, operation="-")
        print("copied matrix after subtraction", r2_mat)
        mat = flatten_matrix(r2_mat, 1)
        mat = mat[c:]
        print("flattened matrix for the next iteration", mat)
        print("Current result", fin_result)

    print("Final result after all iterations:", fin_result)
    #print("determinant_coeff", determinant_coeff)
    print(diagonal(fin_result))
    diag = diagonal(fin_result)
    diag_prod = product_reduce(diag)
    print("Diag prod", diag_prod)
    #determinant_res = determinant_coeff*diag_prod
    determinant_res = None
    if calc_determinant == True:
        determinant_res = diag_prod
    print(determinant_res)
    return fin_result, determinant_res

def backward_substitution(matrix, aug_col_ix):
    """Input: "matrix": Full augmented matrix result from Gauss-elimination.
    "aug_col_ix": size/number of augmented_columns, column index (starting from zero), which seperates the augmented part, from the substitution part.
    Out: Identity matrix, and solution to the augmented part."""
    #non_augmented_matrix = copy.deepcopy(matrix[:][:aug_col_ix])
    r, c = shape(matrix)
    empty_mat = empty_matrix(r, c) #full matrix result
    print("r, c", r, c)
    last_row = matrix[r-1]
    identity = identity_matrix(c-aug_col_ix)
    is_identity = False
    print("is_identity", is_identity)
    print(empty_mat)
    def extract_non_augmented_mat(full_mat, ix):
        non_augmented_mat = []
        for row in full_mat:
            non_augmented_mat.append(row[:ix])
        return non_augmented_mat
    while is_identity == False:
        print("R", r)
        print("last row", last_row)
        last_row_last_el = last_row[c-1-aug_col_ix]
        remaining_rows = copy.deepcopy(matrix[:r-1])
        if len(remaining_rows) == 0:
            break
        print("remaining_rows", remaining_rows)
        remaining_rows_last_els = []
        for i, el in enumerate(remaining_rows):
            remaining_rows_last_els.append(el[c-1-aug_col_ix])
        print("remaining_rows", remaining_rows)
        print("remaining rows_last_el", remaining_rows_last_els)
        coeffs = scalar_broadcast(last_row_last_el, remaining_rows_last_els, "/")
        print("coeffs", coeffs)
        mult_last_rows = []
        for val in coeffs:
            mult_last_rows.append(scalar_broadcast(val, last_row, "*"))
        remaining_rows = matrix_broadcast(remaining_rows, mult_last_rows, "-")
        print("remaining rows after subtraction", remaining_rows)
        empty_mat[:r-1] = remaining_rows
        empty_mat[r-1] = last_row
        last_row = remaining_rows[-1]
        r -=1
        c -=1
        print("OG matrix", matrix)
        print("current_result", empty_mat)
        non_aug_mat = extract_non_augmented_mat(empty_mat, aug_col_ix)
        print("non_augmented", non_aug_mat)
        is_identity = matrix_equality(non_aug_mat, identity)
        print(is_identity)
    print("final result of backward substitution", empty_mat)
    return empty_mat

#https://math.stackexchange.com/questions/130174/calculating-matrix-rank-with-gaussian-elimination 
# https://socratic.org/questions/how-do-i-find-the-rank-of-a-matrix-using-gaussian-elimination 
# TODO file reading [X]

#flatten_matrix(reshape_matrix(r2_mat, c),1)

def determinant(matrix,n):
    #TODO [X]
    #https://math.stackexchange.com/questions/2269267/determinant-of-large-matrices-theres-gotta-be-a-faster-way
    #https://matrix-calculators.com/lu-decomposition-calculator
    """Use Gauss Jordan elimination with option 'calc_determinant'=True"""
    """Product of diagonals after Gauss-Jordan elimination"""
    pass

def inverse(matrix):
    """Input: "matrix": Full augmented matrix result from Gauss-elimination.
    Out: Identity matrix, and solution to the augmented part."""
    r, c = shape(matrix)
    identity = identity_matrix(c)
    augmented_matrix = []
    for rx, ry in zip(matrix, identity):
        augmented_matrix.append(rx+ry)
    aug_res, determinant = Gauss_Jordan_elimination(augmented_matrix, calc_determinant=False)
    print("AUG_RES", aug_res)
    final_aug_res = backward_substitution(aug_res, c)
    temp = reshape_matrix(flatten_matrix(final_aug_res,1), c)
    clean_res = []
    for i, val in enumerate(temp):
        if i % 2 == 1:
            clean_res.append(val)
    inv_res = {"full_matrix_result": final_aug_res, "clean_inverted_solution": clean_res}
    print("Inverted result", clean_res)
    return inv_res

def rank(matrix):
    """Calculates non-zero rows after Gauss-Jordan elimination. """
    out_matrix, _ = Gauss_Jordan_elimination(matrix)
    ranks = 0
    for row in out_matrix:
        non_zero_values = [x for x in row if x!=0]
        if len(non_zero_values)>0:
            ranks+=1
    print("rank of matrix", ranks)
    return ranks

def main():
    #test = [[1,2,3], [4,5,6], [7,8,9]]
    #test2 = [[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]]
    test5 = [[1,0,1],[0,2,1],[1,1,1]]
    #print(diagonal(test5))
    """print(test)
    print(shape(test))
    print(shape(test2))
    test3 = add_dimension(test2)
    print(test3)
    print(shape(test3))
    test4 = flatten_matrix(test3, 1)
    print(test4)
    print(shape(test4))
    print("hello")
    print(transpose_matrix(test))"""
    #out, determinant = Gauss_Jordan_elimination(test5, calc_determinant=False)
    #print("determinant", determinant)
    #print(out[0][2])
    #out2 = backward_substitution(out, 1) # last row is augmented solution
    #print("OUT after substitution", out2)
    #rank(test5)
    inverse(test5)
    """https://www.matrixcalc.org/#%7B%7B1,2%7D,%7B4,5%7D%7D*%7B%7B1%7D,%7B2%7D%7D""" #check
    #v1 = [2,4, 8]
    #v2 = [-2, 0, 1] 
    """v1 = [1,2,3]
    v2 = [4,5,6]
    res_vec = vector_cross_product(v1,v2)
    print("Result of vector_cross_product", res_vec)"""

#main()