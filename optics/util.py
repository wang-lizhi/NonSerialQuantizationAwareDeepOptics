import math

import numpy as np
import tensorflow as tf


##############################
# 工具函数
##############################
# def get_zernike_volume(resolution, n_terms, scale_factor=1e-6):
#     import poppy
#     zernike_volume = poppy.zernike.zernike_basis(nterms=n_terms, npix=resolution, outside=0.0)
#     return zernike_volume * scale_factor


def fspecial(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def zoom(image_batch, zoom_fraction):
    """
    Get central crop of batch
    """
    images = tf.unstack(image_batch, axis=0)
    crops = []
    for image in images:
        crop = tf.image.central_crop(image, zoom_fraction)
        crops.append(crop)
    return tf.stack(crops, axis=0)


def transpose_2d_fft(a_tensor, dtype=tf.complex64):
    """
    Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflow fft2d to work.
    """
    # Tensorflow FFT 仅支持64复数形式
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow 的二维 FFT 操作 two innermost (最后2个) 维度，将宽高两维度变换到最末以供 FFT 操作
    a_tensor_transp = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])
    a_fft2d = tf.signal.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a=a_fft2d, perm=[0, 2, 3, 1])
    return a_fft2d


def transpose_2d_ifft(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a=a_tensor, perm=[0, 3, 1, 2])  # b, h, w, c => b, c, h, w
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.signal.ifft2d(a_tensor)  # inner-most做ifft
    # 重新变换回 [batch_size, x, y, channels] 的形式
    a_ifft2d = tf.transpose(a=a_ifft2d_transp, perm=[0, 2, 3, 1])  # b, c, h, w => b, h, w, c
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d


def complex_exponent_tf(phase, dtype=tf.complex64, name='complex_exp'):
    """
    Complex exponent via euler's formula, since Cuda doesn't have a GPU kernel for that.
    Casts to *dtype*.
    Returns: cos(phi) + sin(phi)j
    """
    phase = tf.cast(phase, tf.float64)
    return tf.add(tf.cast(tf.cos(phase), dtype=dtype),
                  1.j * tf.cast(tf.sin(phase), dtype=dtype),
                  name=name)


def get_one_phase_shift_thickness(wave_lengths, refractive_index):
    """
    Calculate the thickness (in meter) of a phase-shift of 2pi.
    """
    # refractive index difference
    delta_n = refractive_index - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths

    two_pi_thickness = (2. * np.pi) / (wave_nos * delta_n)
    return two_pi_thickness


def fft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        split = (input_shape[axis] + 1) // 2
        temp_list = np.concatenate((np.arange(split, input_shape[axis]), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def ifft_shift_2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()
    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        temp_list = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, temp_list, axis=axis)
    return new_tensor


def psf2otf(input_filter, output_size):
    """
    Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(tensor=input_filter, paddings=[[pad_top, pad_bottom],
                                                       [pad_left, pad_right], [0, 0], [0, 0]], mode="CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(a=padded, perm=[2, 0, 1, 3])
    padded = ifft_shift_2d_tf(padded)
    padded = tf.transpose(a=padded, perm=[1, 2, 0, 3])

    # Take FFT
    tmp = tf.transpose(a=padded, perm=[2, 3, 0, 1])
    tmp = tf.signal.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(a=tmp, perm=[2, 3, 0, 1])


def next_power_of_two(number):
    closest_pow = np.power(2, np.ceil(np.math.log(number, 2)))
    return closest_pow


def image_patches_serial_conv_with_psfs(image_patches, psfs, patch_num):
    image_patches_split = tf.split(image_patches, axis=0, num_or_size_splits=patch_num)
    psfs_split = tf.split(psfs, axis=0, num_or_size_splits=patch_num)
    first_image = tf.squeeze(image_patches_split[0], axis=0)
    result = tf.expand_dims(
        image_convolve_with_psf(img=first_image, psf=tf.squeeze(psfs_split[0], axis=0), img_shape=first_image.shape),
        axis=0)
    for i in range(1, patch_num):
        result = tf.concat([result, tf.expand_dims(
            image_convolve_with_psf(img=tf.squeeze(image_patches_split[i], axis=0),
                                    psf=tf.squeeze(psfs_split[i], axis=0),
                                    img_shape=first_image.shape), axis=0)], axis=0)
    return result


def image_convolve_with_psf(img, psf, otf=None, adjoint=False, img_shape=None):
    """
    图像与PSF进行卷积
    Args:
        img: Image tensor.
        psf: PSF tensor.
        otf: If OTF is already computed, the otf.
        adjoint: Whether to perform an adjoint convolution or not.
        circular: Whether to perform a circular convolution or not.
        img_shape: Image shape

    Returns: The image convolved with PSF.

    """
    img = tf.convert_to_tensor(value=img, dtype=tf.float32)
    psf = tf.convert_to_tensor(value=psf, dtype=tf.float32)
    if img_shape is None:
        img_shape = img.shape.as_list()
    psf_shape = psf.shape.as_list()

    # _, psf_h, _, _ = psf.shape
    print(" DEBUG [f] image_convolve_with_psf: img_shape=", img_shape)  # b, h, w, c
    print(" DEBUG [f] image_convolve_with_psf： psf_shape=", psf_shape)  # h, w, b, c

    if psf_shape[0] != img_shape[1] or psf_shape[1] != img_shape[2]:
        # 若不一致,则 padding 到和 PSF 一致的大小
        target_side_length = psf_shape[0]  # 2 * img_shape[1]
        height_pad = tf.abs(target_side_length - img_shape[1]) / 2
        width_pad = tf.abs(target_side_length - img_shape[1]) / 2

        pad_top, pad_bottom = int(tf.math.ceil(height_pad)), int(tf.math.floor(height_pad))
        pad_left, pad_right = int(tf.math.ceil(width_pad)), int(tf.math.floor(width_pad))
        if psf_shape[0] > img_shape[1]:
            print("[Image Conv.] Images will be padded from image height", img_shape[1], "to PSF height", psf_shape[0],
                  ".")
            img = tf.pad(tensor=img, paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                         mode="SYMMETRIC")  # CONSTANT tf.reduce_mean(img)
            img_shape = img.shape.as_list()
        else:
            print("[Image Conv.] PSF will be padded from PSF height", psf_shape[0], "to image height", img_shape[1],
                  ".")
            psf = tf.pad(tensor=psf, paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0], [0, 0]],
                         mode="CONSTANT")  # CONSTANT tf.reduce_mean(img)
            psf_shape = psf.shape.as_list()

    img_fft = transpose_2d_fft(img)
    print(" DEBUG [f] image_convolve_with_psf: img_fft shape=", img_fft.shape)

    if otf is None:
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(a=otf, perm=[2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)

    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transpose_2d_ifft(img_fft * tf.math.conj(otf))
    else:
        result = transpose_2d_ifft(img_fft * otf)
    result = tf.cast(tf.math.real(result), tf.float32)

    # if not circular:
    #     result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]
    return result


def depth_dep_convolution(img, psfs, disc_depth_map):
    """
    Convolve an image with different psfs at different depths as determined by a discretized depth map.

    Args:
        img: image with shape (batch_size, height, width, num_img_channels)
        psfs: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        disc_depth_map: Discretized depth map.
        use_fft: Use img_psf_conv or normal conv2d
    """
    # TODO: only convolve with PSFS that are necessary.
    img = tf.cast(img, dtype=tf.float32)
    input_shape = img.shape.as_list()

    zeros_tensor = tf.zeros_like(img, dtype=tf.float32)
    disc_depth_map = tf.tile(tf.cast(disc_depth_map, tf.int16),
                             multiples=[1, 1, 1, input_shape[3]])
    blurred_imgs = []
    for depth_idx, psf in enumerate(psfs):
        psf = tf.cast(psf, dtype=tf.float32)
        condition = tf.equal(disc_depth_map, tf.convert_to_tensor(value=depth_idx, dtype=tf.int16))
        blurred_img = image_convolve_with_psf(img, psf)
        # 根据条件添加
        blurred_imgs.append(tf.where(condition, blurred_img, zeros_tensor))

    result = tf.reduce_sum(input_tensor=blurred_imgs, axis=0)
    return result


# def get_spherical_wavefront_phase(resolution,
#                                   physical_size,
#                                   wave_lengths,
#                                   source_distance):
#     source_distance = tf.cast(source_distance, tf.float64)
#     physical_size = tf.cast(physical_size, tf.float64)
#     wave_lengths = tf.cast(wave_lengths, tf.float64)
#
#     N, M = resolution
#     [x, y] = np.mgrid[-N // 2:N // 2, -M // 2:M // 2].astype(np.float64)
#     x = x / N * physical_size
#     y = y / M * physical_size
#
#     # Assume distance to source is approx. constant over wave
#     curvature = tf.sqrt(x ** 2 + y ** 2 + source_distance ** 2)
#     wave_nos = 2. * np.pi / wave_lengths
#
#     phase_shifts = complex_exponent_tf(wave_nos * curvature)
#     phase_shifts = tf.expand_dims(tf.expand_dims(phase_shifts, 0), -1)
#     return phase_shifts


def least_common_multiple(a, b):
    return abs(a * b) / math.gcd(a, b) if a and b else 0


def area_downsampling_tf(input_image, target_side_length):
    input_shape = input_image.shape.as_list()
    input_image = tf.cast(input_image, tf.float32)

    if not input_shape[1] % target_side_length:
        factor = int(input_shape[1] / target_side_length)
        output_img = tf.nn.avg_pool2d(input=input_image,
                                      ksize=[1, factor, factor, 1],
                                      strides=[1, factor, factor, 1],
                                      padding="VALID")
    else:
        # 上采样图像并进行平均池化
        lcm_factor = least_common_multiple(target_side_length, input_shape[1]) / target_side_length

        if lcm_factor > 10:
            print("【警告】源宽度和目标宽度最小公倍数过大，下采样（area downsampling）将不精确且非常耗费性能。")
            upsample_factor = 10
        else:
            upsample_factor = int(lcm_factor)

        img_upsampled = tf.image.resize(input_image,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                        size=2 * [upsample_factor * target_side_length])
        output_img = tf.nn.avg_pool2d(input=img_upsampled,
                                      ksize=[1, upsample_factor, upsample_factor, 1],
                                      strides=[1, upsample_factor, upsample_factor, 1],
                                      padding="VALID")
    return output_img
