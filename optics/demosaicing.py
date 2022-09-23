import numpy as np
import tensorflow as tf

from optics.sensor_bayer_pattern import RGGB_BAYER_PATTERN_MASK, RGGB_BAYER_PATTERN_BOOLEAN_MASK


def mask_generator(pattern='RGGB', shape=(2, 2)):
    pattern = pattern.upper()

    channels = dict((channel, np.zeros(shape)) for channel in 'RGB')
    for channel, (y, x) in zip(pattern, [(0, 0), (0, 1), (1, 0), (1, 1)]):
        channels[channel][y::2, x::2] = 1

    return tuple(channels[c].astype(bool) for c in 'RGB')


def conv_nchw(_inp, _filter):
    _filter = tf.reshape(_filter, [5, 5, 1, 1])
    return tf.nn.conv2d(input=_inp, filters=_filter, strides=(1, 1), padding="SAME", data_format="NCHW")


def broadcast_2d_height_width_tensor_to_4d_with_single_batch_and_channel_nchw(tensor_2d):
    return tf.expand_dims(tf.expand_dims(tensor_2d, axis=0), axis=0)


@tf.function
def demosaicing_bayer_malvar_2004_rggb(raw_image):
    """
    去马赛克 *Malvar (2004)* demosaicing algorithm

    TF BUG: 在使用 NHWC 时 TF计算图优化过程会报错
    layout failed: Invalid argument: Size of values 0 does not match size of permutation 4 @ fanin shape
    ingradient_tape/learned_diffractive_optics_model/sensor/SelectV2-2-TransposeNHWCToNCHW-LayoutOptimizer
    故内部实现转为 NCHW 格式绕开TF的通道顺序优化

    Args:
        raw_image: RGGB RAW图像

    Returns: 去马赛克后的RGB图像

    """
    _, height, width, _ = raw_image.shape
    assert height == width, "方法demosaicing_bayer_malvar_2004_rggb()只支持正方型RAW图。"
    raw_image = tf.transpose(raw_image, perm=[0, 3, 1, 2])
    f0 = tf.constant(
        [[0, 0, -1, 0, 0],
         [0, 0, 2, 0, 0],
         [-1, 2, 4, 2, -1],
         [0, 0, 2, 0, 0],
         [0, 0, -1, 0, 0]], dtype=tf.float32)

    f1 = tf.constant(
        [[0, 0, 0.5, 0, 0],
         [0, -1, 0, -1, 0],
         [-1, 4, 5, 4, -1],
         [0, -1, 0, -1, 0],
         [0, 0, 0.5, 0, 0]], dtype=tf.float32)

    f2 = tf.transpose(f1)

    f3 = tf.constant(
        [[0, 0, -1.5, 0, 0],
         [0, 2, 0, 2, 0],
         [-1.5, 0, 6, 0, -1.5],
         [0, 2, 0, 2, 0],
         [0, 0, -1.5, 0, 0]], dtype=tf.float32)

    d0 = conv_nchw(raw_image, f0 / 8.0)
    d1 = conv_nchw(raw_image, f1 / 8.0)
    d2 = conv_nchw(raw_image, f2 / 8.0)
    d3 = conv_nchw(raw_image, f3 / 8.0)

    bc = broadcast_2d_height_width_tensor_to_4d_with_single_batch_and_channel_nchw

    r_mask = bc(RGGB_BAYER_PATTERN_MASK[height].r)
    g_mask = bc(RGGB_BAYER_PATTERN_MASK[height].g1 + RGGB_BAYER_PATTERN_MASK[height].g2)
    b_mask = bc(RGGB_BAYER_PATTERN_MASK[height].b)

    r_boolean_mask = bc(RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].r)
    g1_boolean_mask = bc(RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].g1)
    g2_boolean_mask = bc(RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].g2)
    b_boolean_mask = bc(RGGB_BAYER_PATTERN_BOOLEAN_MASK[height].b)

    r = tf.multiply(raw_image, r_mask)
    g = tf.multiply(raw_image, g_mask)
    b = tf.multiply(raw_image, b_mask)

    r = tf.where(g1_boolean_mask, d1, r)
    r = tf.where(g2_boolean_mask, d2, r)
    r = tf.where(b_boolean_mask, d3, r)

    g = tf.where(tf.logical_or(r_boolean_mask, b_boolean_mask), d0, g)

    b = tf.where(g2_boolean_mask, d1, b)
    b = tf.where(g1_boolean_mask, d2, b)
    b = tf.where(r_boolean_mask, d3, b)

    rgb_nchw = tf.concat([r, g, b], axis=1)
    rgb_nhwc = tf.transpose(rgb_nchw, perm=[0, 2, 3, 1])  # NCHW->NHWC
    return rgb_nhwc
