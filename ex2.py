import numpy as np


def inner_prod(v1, v2):
    """
    Calculates the inner product of two given vectors
    :param v1: vector
    :param v2: vector
    :return: inner product of two vectors
    """
    return sum([x * y for x, y in zip(v1, v2)])


def vec_subtract(v1, v2):
    """
    Subtract vector v2 from vector v1
    :param v1: vector
    :param v2: vector
    :return: vector v1 after subtracting vector v2
    """
    return [x - y for x, y in zip(v1, v2)]


def vec_multiply_by_scalar(vec, scalar):
    """
    Multiply a vector by scalar
    :param vec: vector
    :param scalar: number
    :return: vector multiplied by scalar
    """
    return [x * scalar for x in vec]


def norm(vec, n=2):
    """
    Calculates the norm induced by vector
    :param vec: vector
    :param n: norm, default is 2
    :return: scalar
    """
    return inner_prod(vec, vec) ** (1 / n)


def QR_decomposition(A):
    """
    QR decomposition of matrix A
    :param A: transposed matrix
    :return: Q and R decomposition of matrix A
    """
    Q, R, z_list, w_list = [], [], [], []
    for i, u_i in enumerate(A):
        z_i = u_i
        for index in range(i):
            scalar = inner_prod(u_i, z_list[index]) / inner_prod(z_list[index], z_list[index])
            z_i = vec_subtract(z_i, vec_multiply_by_scalar(z_list[index], scalar))
        z_list.append(z_i)
        w_i = vec_multiply_by_scalar(z_i, 1 / norm(z_i))
        w_list.append(w_i)
        Q.append(w_i)
    # Q Matrix
    Q = np.transpose(Q)
    # R Matrix
    for row in range(len(A)):
        new_row = []
        for col in range(len(A)):
            if row > col:
                new_row.append(0)
            elif row == col:
                new_row.append(norm(z_list[row]))
            else:
                new_row.append(inner_prod(A[col], w_list[row]))
        R.append(new_row)
    return Q, R


def proj(x, mat):
    """
    Projects vector x on subspace created by matrix columns
    :param x: vector to project
    :param mat: matrix
    :return: the projection vector
    """
    return np.matmul(mat, np.matmul(np.transpose(mat), x))


def proj_complement_span(x, mat):
    """
    Projects vector x on the orthogonal complement subspace created by matrix columns
    :param x: vector to project
    :param mat: matrix
    :return: the projection vector
    """
    return np.matmul(np.identity(len(mat)) - np.matmul(mat, np.transpose(mat)), x)


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    A = [[3, 6, 8, 0, 4, 3, 1, 5, 4, 4],
         [4, 0, 6, 5, 1, 9, 3, 3, 3, 3],
         [5, 0, 9, 8, 0, 4, 9, 6, 6, 4],
         [0, 7, 6, 9, 2, 5, 5, 5, 3, 4],
         [2, 3, 8, 1, 2, 2, 6, 6, 6, 4],
         [5, 4, 1, 8, 1, 5, 8, 9, 5, 3],
         [0, 1, 7, 5, 3, 7, 9, 4, 0, 7],
         [2, 9, 2, 8, 3, 4, 8, 2, 2, 5],
         [6, 6, 0, 0, 4, 6, 8, 2, 7, 1],
         [4, 7, 8, 6, 4, 8, 7, 8, 2, 7],
         [7, 5, 9, 9, 5, 1, 8, 4, 3, 8],
         [2, 4, 9, 2, 9, 4, 0, 7, 0, 8],
         [2, 8, 2, 4, 2, 4, 6, 3, 5, 1],
         [2, 9, 6, 8, 2, 5, 9, 0, 0, 9],
         [1, 4, 5, 2, 2, 2, 2, 6, 9, 5]]
    Q, R = QR_decomposition(np.transpose(A))
    print('1.')
    print(f'Q:\n{np.asmatrix(Q)}\n')
    print(f'R:\n{np.asmatrix(R)}\n')
    x = [11, 9, 6, 5, 4, 2, 1, 94, 91, 89, 85, 84, 16, 98, 95]
    print(f'2. x projected on S: {proj(x, Q)}\n')
    print(f'3. x projected on S complement: {proj_complement_span(x, Q)}')
