"""
This module defines the sparse matrices and operations on them.
We use tf.SparseMat to compress memory usage.
"""
import numpy as np
import tensorflow as tf


class SparseMat:
    """
    Define ths sparse matrix and operations on it.
    """
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.row, self.col, self.data = [], [], []

    def add(self, row, col, data):
        self.row.append(row)
        self.col.append(col)
        self.data.append(data)

    def get_row(self):
        return np.array(self.row)

    def get_col(self):
        return np.array(self.col)

    def get_data(self):
        return np.array(self.data)


def merge_sparse_mats(sparse_mats, depth):
    """
    Merge multiple sparse matrices (which have the same shape) to a global sparse matrix on its diagonal.
    The merge operation is taken on each depth separately.
    e.g., for the first depth of three matrices, from
    [0, 1, 0]    [0, 1, 0]    [0, 0, 1]
    [1, 0, 0]    [0, 0, 1]    [0, 1, 0]
    [0, 0, 1]    [1, 0, 0]    [0, 1, 0]

    we have

    [0, 1, 0]
    [1, 0, 0]   ..  ..    ..  ..
    [0, 0, 1]
              [0, 1, 0]
     ..  ..   [0, 0, 1]   ..  ..
              [1, 0, 0]
                        [0, 0, 1]
     ..  ..    ..  ..   [0, 1, 0]
                        [0, 1, 0]
    :param sparse_mats: a list of sparse matrices. Each matrix has the 3rd dim, depth
    :param depth: the depth of each sparse matrix, which is orthogonal to the planar operation above
    :return: a list of merged global sparse matrix. The list length is the depth
    """
    global_sp_mat = []
    for d in range(depth):
        row_idx, col_idx, data = [], [], []
        shape = 0
        base = 0
        for mat in sparse_mats:
            row_idx.append(mat[d].get_row() + base)
            col_idx.append(mat[d].get_col() + base)
            data.append(mat[d].get_data())
            shape += mat[d].shape[0]
            base += mat[d].shape[0]
        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)

        indices = np.mat([row_idx, col_idx]).transpose()
        global_sp_mat.append(tf.SparseTensorValue(indices, data, (shape, shape)))
    return global_sp_mat


def expand_sparse_mats(sparse_mats, exp_step):
    """
    Make a stack of the same sparse matrix to a global one on its diagonal. The expand operation is taken on each depth separately.
    Here the depth is the length of sparse_mats.
    e.g.,
    On one depth of sparse_mats, we have
    [0, 1, 0]    [0, 1, 0]
    [1, 0, 0]    [1, 0, 0]  ..  ..   ..  ..
    [0, 0, 1]    [0, 0, 1]
                          [0, 1, 0]
              to  ..  ..  [1, 0, 0]  ..  ..
                          [0, 0, 1]
                                   [0, 1, 0]
                  ..  ..   ..  ..  [1, 0, 0]
                                   [0, 0, 1]

    where exp_step is 3.
    :param sparse_mats: of type tf.SparseTensorValue
    :param exp_step: expand times
    :return: a list of expanded global sparse matrix. The list length is the same as sparse_mats
    """
    global_sp_mat = []
    depth = len(sparse_mats)
    for d in range(depth):
        row_idx, col_idx, data = [], [], []
        shape = 0
        base = 0
        for i in range(exp_step):
            indices = sparse_mats[d].indices.transpose()
            row_idx.append(np.squeeze(np.asarray(indices[0, :]) + base))
            col_idx.append(np.squeeze(np.asarray(indices[1, :]) + base))
            data.append(sparse_mats[d].values)
            shape += sparse_mats[d].dense_shape[0]
            base += sparse_mats[d].dense_shape[0]

        row_idx = np.hstack(row_idx)
        col_idx = np.hstack(col_idx)
        data = np.hstack(data)
        indices = np.mat([row_idx, col_idx]).transpose()
        global_sp_mat.append(tf.SparseTensorValue(indices, data, (shape, shape)))
    return global_sp_mat


def merge_and_extend_sparse_mats(sparse_mats):
    """
    Transform multiple sparse matrices (which have the same shape) to a global sparse matrix on its diagonal.
    e.g.,
    list of
    [1, 0, 1, 1] [0, 0, 0, 1]
    [1, 1, 1, 1] [0, 1, 1, 1]
    [0, 0, 1, 1] [1, 1, 1, 1]

    to

    [1, 0, 1, 1]
    [1, 1, 1, 1]    ..  ..
    [0, 0, 1, 1]
                 [0, 0, 0, 1]
       ..  ..    [0, 1, 1, 1]
                 [1, 1, 1, 1]
    Compared with merge_sparse_mats(), the input sparse_mats here do not have the 3rd dim.
    :param sparse_mats: a batch of sparse matrices. Each matrix has only 2 dims (not has depth)
    :return: a transformed global matrix
    """
    batch_size = len(sparse_mats)
    row_idx, col_idx, data = [], [], []
    shape = (sparse_mats[0].dense_shape[0] * batch_size, sparse_mats[0].dense_shape[1] * batch_size)

    row_base, col_base = 0, 0
    for b in range(batch_size):
        indices = sparse_mats[b].indices.transpose()
        row_idx.append(np.squeeze(np.asarray(indices[0, :]) + row_base))
        col_idx.append(np.squeeze(np.asarray(indices[1, :]) + col_base))
        data.append(sparse_mats[b].values)
        row_base += sparse_mats[b].dense_shape[0]
        col_base += sparse_mats[b].dense_shape[1]

    row_idx = np.hstack(row_idx)
    col_idx = np.hstack(col_idx)
    data = np.hstack(data)
    indices = np.mat([row_idx, col_idx]).transpose()
    return tf.SparseTensorValue(indices, data, shape)
