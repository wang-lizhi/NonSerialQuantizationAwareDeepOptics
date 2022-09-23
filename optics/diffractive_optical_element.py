import math

import numpy as np
import tensorflow as tf

from abc import ABC

from log import Logger
from summary import image_normalization
from .util import complex_exponent_tf, transpose_2d_ifft, ifft_shift_2d_tf

BASE_PLANE_THICKNESS = 2 * 1e-3


def summary_height_map(height_map):
    # Normalization is necessary otherwise the etching pattern is too small is be visible
    tf.summary.image(name="HeightMapNormalized", data=image_normalization(
        tf.keras.activations.relu(height_map - BASE_PLANE_THICKNESS + 1.5e-6)))


def _phase_to_height_with_material_refractive_idx_func(_phase, _wavelength, _refractive_index_function):
    return _phase / (2 * math.pi / _wavelength) / (_refractive_index_function(_wavelength) - 1)


# NOA61_HEIGHT_AT_PI_PHASE_DELAY = {
#     700: 1.264587254186165e-6 / 2,
#     550: 0.9776038248514279e-6 / 2,
#     400: 0.6883836469624465e-6 / 2
# }
#
# NOA61_HEIGHT_AT_2_PI_PHASE_DELAY = {
#     700: 1.264587254186165e-6,
#     550: 0.9776038248514279e-6,
#     400: 0.6883836469624465e-6
# }

def _copy_quad_to_full(quad_map):
    height_map_half_left = tf.concat([tf.reverse(quad_map, axis=[0]), quad_map], axis=0)
    height_map_full = tf.concat([tf.reverse(height_map_half_left, axis=[1]), height_map_half_left], axis=1)
    return height_map_full


class DOELayer(ABC, tf.keras.layers.Layer):
    @staticmethod
    def shift_phase_according_to_height_map(input_field, height_map, wave_lengths, wavelength_to_refractive_index_func):
        """
        Calculates the phase shifts created by a height map with certain
        refractive index for light with specific wave length.
        """
        summary_height_map(height_map)
        # delta_n = refractive_indexes.reshape([1, 1, 1, -1]) - 1.  # refractive index difference
        delta_n = wavelength_to_refractive_index_func(wave_lengths) - 1
        wave_numbers = 2. * np.pi / wave_lengths
        wave_numbers = wave_numbers.reshape([1, 1, 1, -1])
        # phase delay indiced by height field
        phi = wave_numbers * delta_n * height_map
        phase_shifts = complex_exponent_tf(phi)
        input_field = tf.cast(input_field, tf.complex64)
        shifted_field = tf.multiply(input_field, phase_shifts, name='phase_plate_shift')
        return shifted_field

    @staticmethod
    def add_height_map_noise(height_map, tolerance=None):
        if tolerance is not None:
            height_map = height_map + tf.random.uniform(shape=height_map.shape,
                                                        minval=-tolerance,
                                                        maxval=tolerance,
                                                        dtype=height_map.dtype)
            Logger.i("DOE tolerance (additive noise): %0.2e" % tolerance)
        else:
            Logger.i("No DOE additive noise will be added.")
        return height_map

    def preprocess_height_map(self, training=None, testing=None):
        return NotImplemented

    def modulate(self, input_field, preprocessed_height_map, height_map_regularizer, height_tolerance, wave_length_list,
                 wavelength_to_refractive_index_func, with_clean_modulation=False):
        if height_map_regularizer is not None:
            self.add_loss(height_map_regularizer(preprocessed_height_map))

        noised_height_map = self.add_height_map_noise(preprocessed_height_map, tolerance=height_tolerance)
        normal_modulation = self.shift_phase_according_to_height_map(
            input_field=input_field,
            height_map=noised_height_map,
            wave_lengths=wave_length_list,
            wavelength_to_refractive_index_func=wavelength_to_refractive_index_func)

        if with_clean_modulation:
            assert height_tolerance is not None and height_tolerance != 0, \
                "It's redundant to enable `with_clean_modulation while `height_tolerance` is None or zero."
            clean_modulation = self.shift_phase_according_to_height_map(
                input_field=input_field,
                height_map=preprocessed_height_map,
                wave_lengths=wave_length_list,
                wavelength_to_refractive_index_func=wavelength_to_refractive_index_func)
            return normal_modulation, clean_modulation
        else:
            return normal_modulation


def read_pretrained_height_map(check_point_path):
    check_point = tf.train.latest_checkpoint(check_point_path)
    # checkpoint_variables = tf.train.list_variables(check_point)
    doe_var = tf.train.load_variable(check_point,
                                     "doe_layer/weight_height_map_radius_1d/.ATTRIBUTES/VARIABLE_VALUE")
    return doe_var


class FixedDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_map_initializer=None,
                 height_tolerance=None, quantization_levels=None, name="FixedDOELayer"):
        super(FixedDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func

        self.height_map_initializer = height_map_initializer
        self.height_tolerance = height_tolerance
        self.fixed_weight_height_map = None
        self.height_map_shape = None
        self.quantization_levels = quantization_levels

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height, width, 1]

        if isinstance(self.height_map_initializer, str):
            import scipy.io as sio
            mat_value = sio.loadmat(self.height_map_initializer)["height_map"]
            mat_value = tf.expand_dims(tf.expand_dims(mat_value, axis=0), axis=-1)
            if mat_value.shape[1] != height or mat_value.shape[2] != width:
                Logger.w("The initializer MAT file has different shape [%d, %d] from the inner weight. "
                         "It will be resized to [%d, %d]." % (mat_value.shape[1], mat_value.shape[2], height, width))
                mat_value = tf.image.resize(mat_value, size=[height, width], method=tf.image.ResizeMethod.BILINEAR)
            self.height_map_initializer = tf.keras.initializers.constant(value=mat_value)

        assert self.height_map_initializer is not None, \
            "There must be a given initializer for height map of `FixedDOELayer`."
        self.fixed_weight_height_map = self.add_weight(name="fixed_weight_height_map",
                                                       shape=self.height_map_shape,
                                                       dtype=tf.float32,
                                                       trainable=False,
                                                       initializer=self.height_map_initializer)

    def preprocess_height_map(self, training=None, testing=None):
        if self.quantization_levels is None:
            # Return full-precision Fresnel
            return self.fixed_weight_height_map
        else:
            # Quantize the map and return
            Logger.w("The Fresnel lens will be quantized to %d levels." % self.quantization_levels)
            full_precision_value = self.fixed_weight_height_map
            _max_val = tf.reduce_max(full_precision_value)
            Logger.i("Fresnel Max Val {}".format(_max_val))
            quantized_value = tf.cast(tf.round((full_precision_value / _max_val) * (self.quantization_levels - 1)),
                                      dtype=tf.float32)
            quantized_value = (quantized_value / (self.quantization_levels - 1)) * _max_val
        return quantized_value

    def call(self, inputs, training=None, testing=None, with_clean_modulation=False, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=None,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func,
                             with_clean_modulation=with_clean_modulation)


class HeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, block_size=1, height_map_initializer=None,
                 height_map_regularizer=None, height_tolerance=None, name="HeightMapDOELayer"):
        super(HeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.block_size = block_size
        self.height_map_initializer = height_map_initializer
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.weight_height_map = None
        self.height_map_shape = None
        self.height_constraint_max = _phase_to_height_with_material_refractive_idx_func(
            math.pi, 700 * 1e-9, wavelength_to_refractive_index_func)  # height_constraint_max

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height // self.block_size, width // self.block_size, 1]
        if self.height_map_initializer is None:
            init_height_map_value = np.ones(shape=self.height_map_shape, dtype=np.float32) * 1e-4
            # init_height_map_value = read_pretrained_height_map()
            self.height_map_initializer = tf.keras.initializers.constant(value=init_height_map_value)
        self.weight_height_map = self.add_weight(name="weight_height_map_sqrt",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -10, 10))

    def preprocess_height_map(self, training=None, testing=None):
        height_map = self.height_constraint_max * tf.sigmoid(self.weight_height_map)
        return height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class QuantizedHeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_map_initializer=None,
                 height_map_regularizer=None, height_tolerance=None,
                 quantization_height_base_wavelength=700 * 1e-9, quantization_height_base_phase=2 * math.pi,
                 quantization_level_cnt=4, quantize_at_test_only=False,
                 adaptive_quantization=True, alpha_blending=True, step_per_epoch=960,
                 alpha_blending_start_epoch=5, alpha_blending_end_epoch=25, shuffle_doe_height=False,
                 name="QuantizedHeightMapDOELayer"):
        super(QuantizedHeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.height_map_initializer = height_map_initializer
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.step_counter = None
        self.weight_height_map = None
        self.height_map_shape = None
        self.alpha_blending = alpha_blending
        self.base_plane_thickness = BASE_PLANE_THICKNESS
        self.quantization_height_base_wavelength = quantization_height_base_wavelength
        self.quantization_base_height = _phase_to_height_with_material_refractive_idx_func(
            _phase=quantization_height_base_phase,
            _wavelength=quantization_height_base_wavelength,
            _refractive_index_function=self.wavelength_to_refractive_index_func)
        Logger.i("[DOE] Quantization base height: %.12e" % self.quantization_base_height)
        self.quantization_level_adaptive_weight = None
        self.quantization_level_cnt = quantization_level_cnt
        self.quantize_at_test_only = quantize_at_test_only
        self.adaptive_quantization = adaptive_quantization

        self.ALPHA_BLENDING_START_STEP = step_per_epoch * alpha_blending_start_epoch
        self.ALPHA_BLENDING_END_STEP = step_per_epoch * alpha_blending_end_epoch

        self.shuffle_doe_height = shuffle_doe_height

        Logger.i("[DOE] AB Start Step: %d" % self.ALPHA_BLENDING_START_STEP)
        Logger.i("[DOE] AB End Step: %d" % self.ALPHA_BLENDING_END_STEP)
        Logger.i("[DOE] AB Start Epoch: %d" % alpha_blending_start_epoch)
        Logger.i("[DOE] AB End Epoch: %d" % alpha_blending_end_epoch)

    # ALPHA_BLENDING_START_STEP = 960
    # ALPHA_BLENDING_END_STEP = 1920

    # ALPHA_BLENDING_START_STEP = 1920
    # ALPHA_BLENDING_END_STEP = 19200

    def generalizable_pre_build(self):
        # 根据需求初始化 alpha_blending 和 adaptive quantization 的权重
        self.step_counter = self.add_weight(name="step_counter", shape=None, dtype=tf.int32, trainable=False,
                                            initializer=tf.keras.initializers.constant(value=0),
                                            # For distributive training. Only the first replica can update the value.
                                            aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        if self.adaptive_quantization:
            self.quantization_level_adaptive_weight = self.add_weight(name="quantization_level_adaptive_weight",
                                                                      shape=self.quantization_level_cnt,
                                                                      trainable=True,
                                                                      initializer=tf.keras.initializers.constant(
                                                                          tf.zeros(shape=self.quantization_level_cnt)),
                                                                      constraint=(lambda x: tf.clip_by_value(x, -1, 1))
                                                                      if self.quantization_level_cnt > 2
                                                                      else (lambda x: tf.clip_by_value(x, -0.5, 0.5)))
            # 2-level 自适应量化感知时缩小自适应权重取值范围（通过 constraint 设置 clip 表达式）

        if self.height_map_initializer is None:
            # 随机初始化为极小值 -0.0001
            init_height_map_value = np.ones(shape=self.height_map_shape, dtype=np.float32) * 1e-4
            # 加载预训练和height map
            # init_height_map_value = read_pretrained_height_map(
            #     "/data/lilingen/lilingen/projects/learned-optics-general/init-ckpt/Init-DOE-3.45e-06With3mmBase700nm2Pi-SsDst0.025-Sc1stDst1-WvRes1024-MAE/")
            self.height_map_initializer = tf.keras.initializers.constant(value=init_height_map_value)
            # 高斯初始化
            # self.height_map_initializer = tf.random_normal_initializer(mean=0.0, stddev=1)

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [1, height, width, 1]
        self.generalizable_pre_build()
        self.weight_height_map = self.add_weight(name="weight_height_map",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 # 用于使 sigmoid 落在合适的范围
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1)
                                                 )

    def preprocess_height_map(self, training=None, testing=None):

        def _norm_to_0_and_1(_x):
            return (_x + 1) * 0.5  # tf.sigmoid(_x)

        def _full_precise_path(_weight):
            return _norm_to_0_and_1(_weight)

        def _quantized_path(_weight, _round_op, _quantization_level_cnt=4, _adaptive=False,
                            _cur_step=None, _start_step=None, _end_step=None, _tune_first_level=True):
            _normed_height_map = _norm_to_0_and_1(_weight)
            quantized_levels = tf.cast(_round_op(_normed_height_map * (_quantization_level_cnt - 1)),
                                       dtype=tf.float32)  # 0, 1, 2, 3
            Logger.i("[DOE] 量化级别数量：", _quantization_level_cnt)
            # ===== Fine-tune
            if _adaptive:
                Logger.i("[DOE] 将使用附带自适应微调权重的 <alpha_blending> 方法进行量化感知训练。")
                _start_level = 1

                if _tune_first_level:
                    Logger.w("Tuning the 1st level using adaptive mechanism.")
                    _start_level = 0
                else:
                    Logger.i("The 1st level will not be tuned using adaptive mechanism.")

                for level in range(_start_level, _quantization_level_cnt):  # 自适应机制第0个level不能微调
                    level_fine_tune_weight = self.quantization_level_adaptive_weight[level]
                    quantized_levels = tf.where(tf.equal(quantized_levels, level),
                                                quantized_levels + 0.5 * level_fine_tune_weight,
                                                quantized_levels)
                    tf.summary.scalar(name="level_fine_tune_weight%d" % level, data=level_fine_tune_weight)
            # ===== !Fine-tune
            _quantized_height_map = quantized_levels / (_quantization_level_cnt - 1)  # 0-1
            # ===== Fine-tune
            if _adaptive and (testing is not True):
                assert _cur_step is not None, "`_cur_step` must not be none when adaptive mechanism is enabled."
                assert _start_step is not None, "`_start_step` must not be none when adaptive mechanism is enabled."
                assert _end_step is not None, "`_end_step` must not be none when adaptive mechanism is enabled."
                # if _cur_step > _end_step:
                #     adaptive_decay_weight = 1.0
                # else:
                #     adaptive_decay_weight = 0.0
                # if _cur_step < _start_step:
                #     adaptive_decay_weight = 0.0
                # elif _cur_step > _end_step:
                #     adaptive_decay_weight = 1.0
                # else:
                #     adaptive_decay_weight = tf.cast(1.0 - ((_end_step - _cur_step) /
                #                                            (_end_step - _start_step)) ** 3, dtype=tf.float32)
                # (1 - adaptive_decay_weight) *

                # quantization_fine_tune_loss = tf.reduce_mean(tf.square(_quantized_height_map - _normed_height_map))
                quantization_fine_tune_loss = tf.reduce_mean(tf.abs(_quantized_height_map - _normed_height_map))
                tf.summary.scalar(name="quantization_fine_tune_loss", data=quantization_fine_tune_loss)
                self.add_loss(0.01 * quantization_fine_tune_loss)  # fine-tune loss
            # ===== !Fine-tune
            return _quantized_height_map

        def _alpha_blending(_path1, _path2, _cur_step, _start_step, _end_step):
            if _cur_step < _start_step:
                # 全精度 path
                quantization_blending_alpha = 0.0
            elif _cur_step > _end_step:
                # 全量化 path
                quantization_blending_alpha = 1.0
            else:
                quantization_blending_alpha = tf.cast(
                    1.0 - ((_end_step - _cur_step) /
                           (_end_step - _start_step)) ** 3, dtype=tf.float32)

            tf.summary.scalar(name="quantization_blending_alpha", data=quantization_blending_alpha)

            return quantization_blending_alpha * _path1 + (1.0 - quantization_blending_alpha) * _path2

        def _base_plane_wrapper(_etching_height_map_weight):
            return self.base_plane_thickness - (self.quantization_base_height * _etching_height_map_weight)

        if self.height_map_regularizer is not None and (testing is not True):
            Logger.i("[DOE] 作用于高度图底层权重的正则化惩罚项将被应用。")
            self.add_loss(self.height_map_regularizer(self.weight_height_map))

        if self.quantize_at_test_only:
            Logger.i("[DOE] <quantize_at_test_only> 模式已启用。"
                     "此模式可模拟实际制造时不考虑量化带来的误差情况，即仅测试、评估阶段量化，训练阶段使用全精度。")
            assert not self.alpha_blending, "<quantize_at_test_only> 模式下，用于量化感知训练的 ”alpha_blending“ 不可为True."
            assert not self.adaptive_quantization, "<quantize_at_test_only> 模式下，" \
                                                   "用于自适应量化感知训练的 ”adaptive_quantization“ 不可为True."
            # assert testing is not None, "testing 参数不可为None，<quantize_at_test_only>模式未按预期工作。"
            if testing is None:
                Logger.w("testing 参数为 None 的情况出现在 <quantize_at_test_only> 模式中。")
            if training is None:
                Logger.w("training 参数为 None 的情况出现在 <quantize_at_test_only> 模式中。")
            # 模拟实际制造时不考虑量化带来的误差情况，即仅测试量化，训练阶段使用全精度
            if not training and testing:
                # 测试阶段量化
                Logger.i("[DOE] <quantize_at_test_only> 模式：测试阶段，将输出量化的DOE。")

                final_processed_height_map = _base_plane_wrapper(
                    _quantized_path(self.weight_height_map,
                                    _round_op=tf.round,
                                    _quantization_level_cnt=self.quantization_level_cnt,
                                    _adaptive=False))
            else:
                # 训练/验证阶段使用全精度
                Logger.i("[DOE] <quantize_at_test_only> 模式：非测试阶段，将输出全精度 (32-bit) 的DOE。")
                final_processed_height_map = _base_plane_wrapper(_full_precise_path(self.weight_height_map))
        else:
            Logger.i("[DOE] <quantization-aware> 模式已启用。此模式可进行量化感知训练。")
            # 量化感知训练，测试和训练均量化
            if self.alpha_blending:
                Logger.i("[DOE] 将使用 <alpha_blending> 方法进行量化感知训练。")
                # --- With alpha-blending

                final_processed_height_map = _base_plane_wrapper(_alpha_blending(
                    _path1=_quantized_path(
                        self.weight_height_map, _round_op=tf.round, _quantization_level_cnt=self.quantization_level_cnt,
                        _adaptive=self.adaptive_quantization, _cur_step=self.step_counter,
                        _start_step=self.ALPHA_BLENDING_START_STEP,
                        _end_step=self.ALPHA_BLENDING_END_STEP),
                    _path2=_full_precise_path(self.weight_height_map),
                    _cur_step=self.step_counter,
                    _start_step=self.ALPHA_BLENDING_START_STEP,
                    _end_step=self.ALPHA_BLENDING_END_STEP))

            else:
                Logger.i("[DOE] 将使用 <STE> 方法进行量化感知训练。")
                # --- Without alpha-blending

                @tf.custom_gradient
                def _round_keep_gradients(_var_x):
                    def _ste_grad(_dy):
                        return _dy

                    return tf.round(_var_x), _ste_grad

                final_processed_height_map = _base_plane_wrapper(_quantized_path(
                    _weight=self.weight_height_map, _round_op=_round_keep_gradients,
                    _quantization_level_cnt=self.quantization_level_cnt, _adaptive=False))

        # === QE Record
        ideal_cond = _base_plane_wrapper(_full_precise_path(self.weight_height_map))
        quantization_error_mae = tf.reduce_mean(tf.abs(ideal_cond - final_processed_height_map))
        quantization_error_mse = tf.reduce_mean(tf.square(ideal_cond - final_processed_height_map))
        tf.summary.scalar(name="quantization_error_mae", data=quantization_error_mae)
        tf.summary.scalar(name="quantization_error_mse", data=quantization_error_mse)
        # !=== QE Record

        tf.summary.scalar(name="step_counter", data=self.step_counter)
        if training:
            self.step_counter.assign_add(1)  # increase step

        return final_processed_height_map

    def call(self, inputs, training=None, testing=None, with_clean_modulation=False, **kwargs):
        height_map = self.preprocess_height_map(training=training, testing=testing)
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=height_map,
                             height_map_regularizer=None,  # 量化的regularizer需要在量化前执行
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func,
                             with_clean_modulation=with_clean_modulation)


class QuadSymmetricQuantizedHeightMapDoeLayer(QuantizedHeightMapDOELayer):

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = [int(height / 2), int(width / 2)]
        self.generalizable_pre_build()
        self.weight_height_map = self.add_weight(name="weight_height_map_quad",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1))

    def preprocess_height_map(self, training=None, testing=None):
        # 调用父方法
        height_map_quad = super().preprocess_height_map(training=training, testing=testing)
        height_map_full = _copy_quad_to_full(height_map_quad)
        height_map_full = tf.expand_dims(height_map_full, axis=0)
        height_map_full = tf.expand_dims(height_map_full, axis=-1)  # reshape => [1, h, w, 1]
        return height_map_full


class RotationallySymmetricQuantizedHeightMapDOELayer(QuantizedHeightMapDOELayer):

    def build(self, input_shape):
        _, height, width, _ = input_shape
        self.height_map_shape = int(height / 2)
        self.generalizable_pre_build()
        Logger.d("!!! Creating weight weight_height_map_radius_1d !!!")
        self.weight_height_map = self.add_weight(name="weight_height_map_radius_1d",
                                                 shape=self.height_map_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=self.height_map_initializer,
                                                 # 用于使 sigmoid 落在合适的范围
                                                 constraint=lambda x: tf.clip_by_value(x, -1, 1))

        Logger.d("!!! Created weight weight_height_map_radius_1d !!!")

    def preprocess_height_map(self, training=None, testing=None):
        # 调用父方法
        height_map_1d = super().preprocess_height_map(training=training, testing=testing)
        if self.shuffle_doe_height is not None and self.shuffle_doe_height:
            Logger.w("Shuffling on DOE 1D height map will be applied.")
            # height_map_1d = tf.random.shuffle(height_map_1d)
            # 100nm noise
            height_map_1d = height_map_1d + tf.random.normal(shape=[self.height_map_shape], mean=0, stddev=600e-9)
            # height_map_1d = tf.expand_dims(tf.slice(height_map_1d, begin=[0], size=[int(self.height_map_shape*0.9)]), axis=-1)
            # height_map_1d = tf.expand_dims(height_map_1d, axis=0)
            # height_map_1d = tf.expand_dims(height_map_1d, axis=-1)
            # height_map_1d = tf.image.resize(height_map_1d, size=[self.height_map_shape, 1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            # height_map_1d = tf.squeeze(height_map_1d)
        radius = self.height_map_shape
        diameter = 2 * radius
        [x, y] = np.mgrid[0:diameter // 2, 0:diameter // 2].astype(np.float32)
        radius_distance = tf.sqrt(x ** 2 + y ** 2)
        height_map_quad = tf.where(tf.logical_and(tf.less(radius_distance, 1.0),
                                                  tf.less_equal(0.0, radius_distance)),
                                   height_map_1d[0], 0.0)
        for r in range(1, radius - 1):
            height_map_quad += tf.where(tf.logical_and(tf.less(radius_distance, tf.cast(r + 1, dtype=tf.float32)),
                                                       tf.less_equal(tf.cast(r, dtype=tf.float32), radius_distance)),
                                        height_map_1d[r], 0.0)

        height_map_full = _copy_quad_to_full(height_map_quad)
        height_map = tf.reshape(height_map_full, shape=[1, diameter, diameter, 1])
        return height_map


class FourierDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, frequency_range,
                 height_map_regularizer=None,
                 height_tolerance=None, name="FourierDOELayer"):
        super(FourierDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.frequency_range = frequency_range
        self.height_map_initializer = tf.compat.v1.zeros_initializer()
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.height_map_var = None
        self.height_map_full = None
        self.padding_width = None

        self.weight_fourier_real = None
        self.weight_fourier_imaginary = None

    def build(self, input_shape):
        assert self.frequency_range is not None, "Invalid args"
        _, height, width, _ = input_shape.as_list()
        frequency_range = self.frequency_range
        self.weight_fourier_real = self.add_weight('weight_fourier_coefficients_real',
                                                   shape=[1, int(height * frequency_range),
                                                          int(width * frequency_range), 1],
                                                   dtype=tf.float32, trainable=True,
                                                   initializer=self.height_map_initializer)
        self.weight_fourier_imaginary = self.add_weight('weight_fourier_coefficients_imaginary',
                                                        shape=[1, int(height * frequency_range),
                                                               int(width * frequency_range), 1],
                                                        dtype=tf.float32, trainable=True,
                                                        initializer=self.height_map_initializer)
        self.padding_width = int((1 - self.frequency_range) * height) // 2

    def preprocess_height_map(self, training=None, testing=None):
        fourier_coefficients = tf.complex(self.weight_fourier_real, self.weight_fourier_imaginary)
        fourier_coefficients_padded = tf.pad(tensor=fourier_coefficients,
                                             paddings=[[0, 0], [self.padding_width, self.padding_width],
                                                       [self.padding_width, self.padding_width], [0, 0]])
        height_map = tf.math.real(transpose_2d_ifft(ifft_shift_2d_tf(fourier_coefficients_padded)))
        return height_map

    def call(self, inputs, training=None, testing=None, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func)


class Rank1HeightMapDOELayer(DOELayer):
    def __init__(self, wave_length_list, wavelength_to_refractive_index_func, height_constraint_max,
                 height_map_regularizer=None, height_tolerance=None, name="Rank1ParameterizedHeightMapDOELayer"):
        super(Rank1HeightMapDOELayer, self).__init__(name=name)
        self.wave_length_list = wave_length_list
        self.wavelength_to_refractive_index_func = wavelength_to_refractive_index_func
        self.height_map_regularizer = height_map_regularizer
        self.height_tolerance = height_tolerance
        self.height_constraint_max = height_constraint_max

        self.weight_map_column = None
        self.weight_map_row = None

    def build(self, input_shape):
        _, height, width, _ = input_shape.as_list()
        column_shape = [1, width]
        row_shape = [height, 1]
        column_init_value = np.ones(shape=column_shape, dtype=np.float32) * 1e-2
        row_init_value = np.ones(shape=row_shape, dtype=np.float32) * 1e-2
        column_initializer = tf.keras.initializers.constant(value=column_init_value)
        row_initializer = tf.keras.initializers.constant(value=row_init_value)
        # 分解为行列两个向量
        self.weight_map_column = self.add_weight(name="weight_height_map_column",
                                                 shape=column_shape,
                                                 dtype=tf.float32,
                                                 trainable=True,
                                                 initializer=column_initializer)
        self.weight_map_row = self.add_weight(name="weight_height_map_row",
                                              shape=row_shape,
                                              dtype=tf.float32,
                                              trainable=True,
                                              initializer=row_initializer)

    def preprocess_height_map(self, training=None, testing=None):
        height_map_mul = tf.matmul(self.weight_map_row, self.weight_map_column)  # (h, w)
        height_map = 1.125 * 1e-6 * tf.sigmoid(height_map_mul)  # clip to [0, 1.125μm]
        height_map = tf.expand_dims(height_map, 0)  # (1, h, w)
        height_map = tf.expand_dims(height_map, -1)  # (1, h, w, 1)
        return height_map

    def call(self, inputs, training=None, testing=None, with_clean_modulation=False, **kwargs):
        return self.modulate(input_field=inputs,
                             preprocessed_height_map=self.preprocess_height_map(training=training, testing=testing),
                             height_map_regularizer=self.height_map_regularizer,
                             height_tolerance=self.height_tolerance,
                             wave_length_list=self.wave_length_list,
                             wavelength_to_refractive_index_func=self.wavelength_to_refractive_index_func,
                             with_clean_modulation=with_clean_modulation)
