import tensorflow as tf
import numpy as np


def laplacian_filter_tf(img_batch):
    """
    Laplacian filter. Also considers diagonals.
    """
    laplacian_filter = tf.constant([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=tf.float32)
    laplacian_filter = tf.reshape(laplacian_filter, [3, 3, 1, 1])
    laplacian_filter = tf.cast(laplacian_filter, tf.float32)
    filter_input = tf.cast(img_batch, tf.float32)
    filtered_batch = tf.nn.convolution(input=filter_input, filters=laplacian_filter, padding="SAME")
    return filtered_batch


def total_variance_1d(vec):
    """
    Total variance for 1-D vector.
    """
    pixel_diff = vec[1:] - vec[:-1]
    tv_id = tf.reduce_sum(tf.abs(pixel_diff))
    return tv_id


def laplace_l1_regularizer(scale=100.0):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l1(a_tensor):
        # with tf.compat.v1.name_scope('laplace_l1_regularizer'):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        # 此处若不将其转为 scalar 将导致 model.test_step 中累加 正则 loss 和 基本 loss 操作 (Add_N) 时 shape 不匹配
        return tf.reshape(
            tf.multiply(scale, tf.reduce_mean(input_tensor=tf.abs(laplace_filtered)),
                        name="laplace_l1_regularized"), [])

    return laplace_l1


def laplace_l2_regularizer(scale):
    if np.allclose(scale, 0.):
        print("Scale of zero disables the laplace_l1_regularizer.")

    def laplace_l2(a_tensor):
        laplace_filtered = laplacian_filter_tf(a_tensor)
        laplace_filtered = laplace_filtered[:, 1:-1, 1:-1, :]
        return scale * tf.reduce_mean(input_tensor=tf.square(laplace_filtered))

    return laplace_l2


REGULARIZER_MAP = {
    "laplace_l1": laplace_l1_regularizer(scale=0.01),
    "total_variance_1d": total_variance_1d,
}
