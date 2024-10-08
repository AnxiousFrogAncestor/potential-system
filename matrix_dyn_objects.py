import matrix_operations as mp

class Array:
    def __new__(self, data):
        """Infer vector shape at runtime and create a same sized Vector or Matrix."""
        assert type(data) == list
        shape = mp.shape(data)
        if type(shape) != tuple:
            raise Exception("Shape of Array must be a tuple")
        for i in shape:
            assert type(i) == int
        self.shape = shape
        print(f"Array.shape is {self.shape}")
        self.data = data
        if len(shape)==1:
            return Vector(self.shape, self.data)
        else:
            return Matrix(self.shape, self.data)

class Matrix:
    def __init__(self, shape, data):
        if type(shape) != tuple:
            raise Exception("Shape of Matrix must be a tuple")
        self.shape = shape
        self.data = data
    def __repr__(self):
        return str(self.data)
    def __getitem__(self, ix):
        print(type(ix))
        """Returns row at given index OR slice."""
        if type(ix) == slice:
            return self.data[ix]
        elif type(ix) == tuple:
            x, y = ix
            return self.data[x][y]
    def __setitem__(self, ix, iy, values):
        """Vectorized assignment of values at index OR slice of ROW."""
        new_val = 0
        if type(values)==int and type(ix)==slice:
            #can only assign iterables
            new_val = [values]
        else:
            new_val = values
        print("NV" , new_val)
        self.data[ix][iy] = new_val
    def shape(self):
        self.shape = mp.shape(self.data)
        return self.shape
    def reshape_matrix(self, shape):
        """Returns a new matrix with new shape."""
        res = mp.matrix(shape, reshape=True,in_matrix=self.data)
        return Array(res)
    def flatten(self):
        new_mat = mp.flatten_matrix(self.data)
        return Array(new_mat)
    def add_dimension(self):
        new_mat = mp.add_dimension(self.data)
        return Array(new_mat)
    def __add__(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="+")
            return Array(res)
        elif type(other)==Vector:
            """Uses overloaded operator of Vector class to perform operations."""
            res = self+other
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self, "-")
    def __sub__(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="-")
            return Array(res)
        elif type(other)==Vector:
            """Uses overloaded operator of Vector class to perform operations."""
            res = self-other
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self, "-")
    def __mul__(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="*")
            return Array(res)
        elif type(other)==Vector:
            res = self*other
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self, "*")
    def __truediv__(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="/")
            return Array(res)
        elif type(other)==Vector:
            res =  self/other
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self, "/")
    def __matmul__(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="+")
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(other.data, self.data, operation="@", in_data=True)
            print("Matrix, vector broadcast.")
            return Array(res)
    def dot(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="@")
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(other.data, self.data, operation="@", in_data=True)
            return Array(res)
    def outer(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="(x)")
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(other.data, self.data, operation="(x)", in_data=True)
            return Array(res)
    def cross(self, other):
        if type(other)==Matrix:
            res = mp.matrix_broadcast(self.data, other.data, operation="x")
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(other.data, self.data, operation="x", in_data=True)
            return Array(res)
    def __eq__(self, other):
        assert type(other) == Matrix
        return mp.matrix_equality(self.data, other.data)


class Vector:
    def __init__(self, shape, data):
        if type(shape) != tuple:
            raise Exception("Shape of Vector must be a tuple")
        self.shape = shape
        self.data = data
    def __repr__(self):
        return str(self.data)
    def __getitem__(self, ix):
        """Returns item at given index OR slice."""
        return self.data[ix]
    def __setitem__(self, ix, values):
        """Vectorized assignment of values at index OR slice."""
        new_val = 0
        if type(values)==int and type(ix)==slice:
            #can only assign iterables
            new_val = [values]
        else:
            new_val = values
        print("NV" , new_val)
        self.data[ix] = new_val
    def shape(self):
        self.shape = mp.shape(self.data)
        return self.shape
    def reshape_vector(self, num_column):
        """Returns new 2d-matrix of shape (row x num_column) from a flattened vector."""
        new_mat = mp.reshape_matrix(self.data, num_column)
        new_shape = mp.shape(new_mat)
        return Matrix(new_shape, new_mat)
    def flatten(self):
        new_mat = mp.flatten_matrix(self.data)
        return Array(new_mat)
    def add_dimension(self):
        new_mat = mp.add_dimension(self.data)
        return Array(new_mat)
    def __add__(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="+",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(self.data, other.data, operation="+", in_data=False)
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self.data, "+")
    def __sub__(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="-",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(self.data, other.data, operation="-", in_data=False)
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self.data, "-")
    def __mul__(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="*",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_broadcast(self.data, other.data, operation="*", in_data=False)
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self.data, "-")
    def __truediv__(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="/",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res =  mp.vector_broadcast(self.data, other.data, operation="/", in_data=False)
            return Array(res)
        elif type(other)==int or type(other)==float:
            return mp.scalar_broadcast(other, self.data, "/")
    def __matmul__(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="@",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.dot_product(self.data, other.data)
            return res
    def dot(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="@",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.dot_product(self.data, other.data)
            return res
    def outer(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="(x)",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.outer_product(self.data, other.data)
            return Array(res)
    def cross(self, other):
        if type(other)==Matrix:
            res = mp.vector_broadcast(self.data, other.data, operation="x",in_data=True)
            return Array(res)
        elif type(other)==Vector:
            res = mp.vector_cross_product(self.data, other.data)
            return Array(res)
    def __eq__(self, other):
        assert type(other) == Vector
        return mp.vector_equality(self.data, other.data)

def main():
    x = Array([1,2])
    y = Array(data=[[1,2], [4,5]])
    z = Array(data=[[1,2], [3,4], [5,6]])
    r = Array(data=[[1,2, 3], [4,5,6], [7,8,9]])
    print(type(x), type(y))
    print("IX-ing", x[0])
    print("Going across rows", z[0:2])
    #ROW, COLUMN Indexing
    print("Going across rows and columns", z[0:2, 0])
    print("Going across rows and columns", r[0, 0:2])
    print(x, y)
    print(x+x)
    print(x*x)
    x[0:1] = 2
    print("Slice assignment after implicit conversion", x)
    x[0] = 1
    print("IX assignment", x)
    print(type(x))
    print("Dot_product", x@x)
    print("Matrix_product", x@y)
    # test https://www.matrixcalc.org/#%7B%7B1,2%7D,%7B4,5%7D%7D*%7B%7B1%7D,%7B2%7D%7D 
    
#main()