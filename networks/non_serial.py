import enum

import tensorflow as tf

from log import Logger
from metrics import ergas_metric
from networks.res_block_u_net import get_res_block_u_net
from optics.sensor_srfs import inverse_simulated_rgb_channel_camera_spectral_response_function, \
    simulated_rgb_camera_spectral_response_function
from optics.util import transpose_2d_fft, transpose_2d_ifft, ifft_shift_2d_tf
from summary import summary_hyper_spec_image

_IFT_SHIFT = ifft_shift_2d_tf
_FFT = transpose_2d_fft
_IFT = transpose_2d_ifft
_CONJ = tf.math.conj
_SRF = simulated_rgb_camera_spectral_response_function
_SRF_INV = inverse_simulated_rgb_channel_camera_spectral_response_function


def _img_conv_psf(_img, _psf, conj=False, transpose=False):
    _otf = _FFT(_IFT_SHIFT(_psf))
    if transpose:
        _otf = tf.transpose(_otf, perm=[0, 2, 1, 3])
    if conj:
        _otf = tf.math.conj(_otf)
    result = _IFT(_FFT(_img) * _otf)
    return tf.cast(tf.math.real(result), tf.float32)


class BHCType(enum.Enum):
    NORMAL = 0
    GATED = 1
    DISABLED = -1


BHC_TYPE_MAP = {
    "normal": BHCType.NORMAL,
    "gated": BHCType.GATED,
    "disabled": BHCType.DISABLED
}


class NonSerialDecoder(tf.keras.layers.Layer):
    def __init__(self, iteration=3, regularization_module_args=None,
                 is_eta_trainable=True, is_eps_trainable=True, eta_init=0.1, eps_init=0.8, input_size=None,
                 shared_hyper_parameters=False, bhc_type="normal", bhc_threshold=50, bhc_weight_init=0, **kwargs):
        super().__init__(**kwargs)
        self.shared_hyper_parameters = shared_hyper_parameters
        self._regularization_term_module = []
        if input_size is not None:
            regularization_module_args["input_size"] = input_size

        Logger.d("regularization_module_args", regularization_module_args)

        for i in range(iteration):
            self._regularization_term_module.append(get_res_block_u_net(**regularization_module_args))

        self.bhc_type = BHC_TYPE_MAP[bhc_type]
        Logger.i("BHC type = %s", bhc_type)
        if self.bhc_type == BHCType.GATED:
            self.bhc_gate_weights = []
            self.bhc_gate_threshold_func = lambda _x: _x < bhc_threshold

            Logger.w("Gated BHC: threshold=", bhc_threshold, "; gate weight init=", bhc_weight_init)
            for i in range(iteration):
                self.bhc_gate_weights.append(self.add_weight(name="BHCGate%d" % iteration, dtype=tf.float32,
                                                             initializer=tf.keras.initializers.constant(
                                                                 bhc_weight_init),
                                                             trainable=False))

        Logger.w("Trainable hyper parameters is enabled in `NonSerialDecoder`.")
        self._eta = []
        self._eps = []
        for i in range(iteration):
            self._eta.append(self.add_weight(name="UnfoldingEta%d" % iteration, dtype=tf.float32,
                                             initializer=tf.keras.initializers.constant(eta_init),
                                             trainable=is_eta_trainable,
                                             constraint=lambda x: tf.where(tf.less(x, 0.0), 0.0, x)))

            self._eps.append(self.add_weight(name="UnfoldingEps%d" % iteration, dtype=tf.float32,
                                             initializer=tf.keras.initializers.constant(eps_init),
                                             trainable=is_eps_trainable,
                                             constraint=lambda x: tf.where(tf.less(x, 0.0), 0.0, x)))

        self.iteration_num = iteration

    def srf(self, _img, iteration):
        return _SRF(_img)

    def inv_srf(self, _img, iteration):
        return _SRF_INV(_img)

    def data_term_conjugate_gradient(self, _z_k, _x_k, _z_0, _psf, _eps, _eta, _iter):
        _term_z_k = _z_k - _eps * _eta * _z_k - _eps * \
                    _img_conv_psf(
                        self.inv_srf(self.srf(_img_conv_psf(_z_k, _psf), iteration=_iter), iteration=_iter),
                        _psf, conj=False, transpose=False)
        _term_z_0 = _eps * _z_0
        _term_x_k = _eps * _eta * _x_k
        return _term_z_k + _term_z_0 + _term_x_k

    def regularization_term(self, _z_cur, _cur_iter):
        return self._regularization_term_module[_cur_iter](_z_cur)

    def call(self, inputs, training=None, testing=None, *args, **kwargs):
        _y, _psf, training_gt = inputs
        # Logger.d("PSF=", _psf)
        # Logger.d("training_gt=", training_gt)
        # training_gt is optional. Gated BHC requires it for training. It's fine to pass a None value when testing.

        Logger.d("NonSerialDecoder input image (_y) shape=", _y.shape)
        Logger.d("NonSerialDecoder input PSF shape=", _psf.shape)
        # PSF Padding
        if _psf.shape[1] != _y.shape[1] or _psf.shape[2] != _y.shape[2]:
            height_pad = (_y.shape[1] - _psf.shape[1]) / 2
            width_pad = (_y.shape[2] - _psf.shape[2]) / 2
            pad_top, pad_bottom = int(tf.math.ceil(height_pad)), int(tf.math.floor(height_pad))
            pad_left, pad_right = int(tf.math.ceil(width_pad)), int(tf.math.floor(width_pad))
            _psf = tf.pad(tensor=_psf,
                          paddings=[[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
                          mode="CONSTANT")
            assert _y.shape[1] == _psf.shape[1] and _y.shape[2] == _psf.shape[2], \
                "Error occurred when padding the PSF in unrolling decoder."

        if self.bhc_type == BHCType.DISABLED:
            _psf = tf.stop_gradient(_psf)
            Logger.w("BHC Type: Disabled. The non-serial model will degrade to a serial model.")
        elif self.bhc_type == BHCType.NORMAL:
            Logger.w("BHC Type: Normal.")

        if self.shared_hyper_parameters:
            _eps = self._eps
            _eta = self._eta
            tf.summary.scalar(name="NSUnrollingSharedEps", data=self._eps)
            tf.summary.scalar(name="NSUnrollingSharedEta", data=self._eta)

        for _iter in range(self.iteration_num):
            if not self.shared_hyper_parameters:
                Logger.w("Using trainable hyper parameters in `NonSerial` iteration.")
                _eps = self._eps[_iter]
                _eta = self._eta[_iter]
                tf.summary.scalar(name="NSUnrollingEps%d" % _iter, data=self._eps[_iter])
                tf.summary.scalar(name="NSUnrollingEta%d" % _iter, data=self._eta[_iter])

            # Data Term "z"
            if training and self.bhc_type == BHCType.GATED:
                Logger.w("Gated BHC is enabled.")

                # calculate BHC gradient for each iteration (BHC)
                @tf.custom_gradient
                def bhc_gate(_x):
                    def grad(_y):
                        if self.bhc_gate_threshold_func(self.bhc_gate_weights[_iter]):
                            return _y
                        else:
                            return 0 * _y

                    return _x, grad

                _psf = bhc_gate(_psf)

            if _iter == 0:
                _z0 = _img_conv_psf(self.inv_srf(_y, iteration=0), _psf, conj=False, transpose=False)
                _z = _z0
            else:
                _z = self.data_term_conjugate_gradient(_z, _x, _z0, _psf, _eps, _eta, _iter=_iter)

            if training and self.bhc_type == BHCType.GATED:
                # Update BHC gate weight
                self.bhc_gate_weights[_iter].assign(ergas_metric(training_gt, _z))
                # if _iter == 0:
                #     tf.print("\n[GateLog]iter0/weight=", (self.bhc_gate_weights[_iter]))
                # if _iter == 1:
                #     tf.print("[GateLog]iter1/weight=", (self.bhc_gate_weights[_iter]))
                # if _iter == 2:
                #     tf.print("[GateLog]iter2/weight", (self.bhc_gate_weights[_iter]))
                # if _iter == 3:
                #     tf.print("[GateLog]iter3/weight=", (self.bhc_gate_weights[_iter]))

                tf.summary.scalar(name="NSBHCGateWeight%d" % _iter, data=self.bhc_gate_weights[_iter])
            summary_hyper_spec_image(name="UnrollingVarZ%d" % _iter, image=_z, norm_all=True)

            # Regularization Term "x"
            _x = self.regularization_term(_z, _iter)
            summary_hyper_spec_image(name="UnrollingVarX%d" % _iter, image=_x, norm_all=True)
        assert _x is not None, "The iteration number must be larger than 0."
        return _x
