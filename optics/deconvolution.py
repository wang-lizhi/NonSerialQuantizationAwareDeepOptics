import tensorflow as tf

from optics.util import psf2otf


def inverse_filter(blurred_image, estimation, psf, gamma):
    """
     Inverse filtering in the frequency domain.

     Args:
         blurred_image: image with shape (batch_size, height, width, num_img_channels)
         estimation: image with shape (batch_size, height, width, num_img_channels)
         psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
         gamma: Optional. Scalar that determines regularization (higher --> more regularization, output is closer to
                "estimate", lower --> less regularization, output is closer to straight inverse filtered-result). If
                not passed, a trainable variable will be created.
         init_gamma: Optional. Scalar that determines the square root of the initial value of gamma.
     """
    img_shape = blurred_image.shape.as_list()

    # if gamma is None:  # Gamma (the regularization parameter) is also a trainable parameter.
    #     gamma = tf.Variable(name="inverse_filter_gamma",
    #                         shape=(),
    #                         dtype=tf.float32,
    #                         trainable=True,
    #                         initial_value=init_gamma)
    #     gamma = tf.square(gamma)  # Enforces positivity of gamma.
    #     tf.summary.scalar('inverse_filter_gamma', gamma)

    blurred_transposed = tf.transpose(blurred_image, [0, 3, 1, 2])
    estimation_transposed = tf.transpose(estimation, [0, 3, 1, 2])

    # Everything has shape (batch_size, num_channels, height, width)
    img_fft = tf.signal.fft2d(tf.complex(blurred_transposed, 0.))   # F(yc
    otf = psf2otf(psf, output_size=img_shape[1:3])
    otf = tf.transpose(otf, [2, 3, 0, 1])  # p_c_bar

    adj_conv = img_fft * tf.math.conj(otf)

    # This is a slight modification to standard inverse filtering - gamma not only regularizes the inverse filtering,
    # but also trades off between the regularized inverse filter and the unfiltered estimation_transposed.
    numerator = adj_conv + tf.signal.fft2d(tf.complex(gamma * estimation_transposed, 0.))

    kernel_mags = tf.square(tf.abs(otf))  # Magnitudes of the blur kernel.

    denominator = tf.complex(kernel_mags + gamma, 0.0)
    filtered = tf.math.divide(numerator, denominator)
    cplx_result = tf.signal.ifft2d(filtered)
    real_result = tf.math.real(cplx_result)  # Discard complex parts.

    # Get back to (batch_size, num_channels, height, width)
    result = tf.transpose(real_result, [0, 2, 3, 1])
    return result
