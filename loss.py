import tensorflow as tf

from metrics import sam_metric, psnr_hyper_metric, ergas_metric
from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function


def ssim_loss(ground_truth, prediction):
    return 1 - tf.reduce_mean(tf.image.ssim(ground_truth, prediction, max_val=1))


def log_loss(ground_truth, prediction):
    loss = tf.math.square(tf.math.log(ground_truth + 1) - tf.math.log(prediction + 1))
    return loss


def mrae_loss(ground_truth, prediction):
    loss = tf.reduce_mean(tf.math.divide(tf.abs(ground_truth - prediction), ground_truth + 1e-5))
    return loss


def post_recheck_mae(ground_truth, prediction):
    reconstruction, pre_sensor, post_recheck = tf.split(prediction, num_or_size_splits=3, axis=-1)
    pre_sensor = tf.squeeze(pre_sensor, axis=-1)
    post_recheck = tf.squeeze(post_recheck, axis=-1)
    mae_recheck = tf.reduce_mean(tf.abs(pre_sensor - post_recheck))
    return mae_recheck


def mae_cyc(gt, pred):
    _reconst, _cyc_loss = tf.split(pred, num_or_size_splits=2, axis=-1)
    return tf.reduce_mean(tf.abs(gt - _reconst)) + _cyc_loss


def ssim_l1_mix_loss(ground_truth, prediction):
    alpha = 0.84
    _l1_loss = tf.reduce_mean(tf.abs(ground_truth - prediction))
    _ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(ground_truth, prediction, max_val=1))
    return alpha * _ssim_loss + (1 - alpha) * _l1_loss


def sam_ssim_huber_mix_scaled_loss(ground_truth, prediction):
    # alpha = 0.84
    _huber_loss = tf.reduce_mean(tf.losses.huber(ground_truth, prediction, delta=0.05))
    _ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(ground_truth, prediction, max_val=1.0))
    _rgb_loss = 1 - tf.reduce_mean(tf.image.ssim(simulated_rgb_camera_spectral_response_function(ground_truth),
                                                 simulated_rgb_camera_spectral_response_function(prediction),
                                                 max_val=1.0))
    _sam_loss = sam_metric(ground_truth, prediction)
    _loss = _ssim_loss + _huber_loss + _rgb_loss + _sam_loss
    return _loss


def mse_sam_mix_loss(ground_truth, prediction):
    _sam_loss = sam_metric(ground_truth, prediction)
    _mse_loss = 1 - tf.reduce_mean(tf.losses.mse(ground_truth, prediction))
    return _sam_loss + _mse_loss


def mae_sam_mix_loss(ground_truth, prediction):
    _sam_loss = 0.2 * sam_metric(ground_truth, prediction)
    _mae_loss = 0.8 * tf.reduce_mean(tf.keras.losses.mae(ground_truth, prediction))
    return _sam_loss + _mae_loss


def psnr_hyper_loss(ground_truth, prediction):
    return -psnr_hyper_metric(ground_truth, prediction)


LOSS_FUNCTION_FILTER = {
    "mse": "mse",
    "mae": "mae",
    "mae+sam": mae_sam_mix_loss,
    "ergas": ergas_metric,
    "sam": sam_metric,
    "mae_cyc": mae_cyc,
    "mrae": mrae_loss
}
