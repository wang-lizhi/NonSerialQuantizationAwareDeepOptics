import enum

import numpy as np
import tensorflow as tf

from log import Logger
from summary import summary_hyper_spec_image, image_normalization
from .noise import gaussian_noise
from .diffractive_optical_element import DOELayer
from .propagation import FresnelPropagation
from .sensor import Sensor
from .util import depth_dep_convolution, image_convolve_with_psf, complex_exponent_tf


class PSFMode(enum.Enum):
    FULL_PAD_IMAGE = 0
    FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG = 1
    FULL_RESIZE = 2
    CROP_VALID = 3


class Camera(tf.keras.layers.Layer):
    """
    模拟光学系统模型类
    """

    def __init__(self, wave_resolution, wave_length_list, sensor_distance, sensor_resolution, input_sample_interval,
                 sensor: Sensor = None, doe_layer: DOELayer = None, target_distance=None, input_channel_num=31,
                 noise_model=gaussian_noise, depth_map=None, depth_list=None, otfs=None,
                 should_use_planar_incidence=False, should_depth_dependent=False,
                 psf_mode=PSFMode.FULL_PAD_IMAGE, noise_sigma=0.001, freeze_doe_layer_weight=False,
                 name="camera"):
        """
        构造 Camera 对象

        Args:
            wave_resolution: 波前分辨率，和DOE、PSF分辨率一致
            wave_length_list: 波长表
            sensor_distance: 传感器和DOE之间的距离
            sensor_resolution: 传感器分辨率
            input_sample_interval: 采样率
            doe_layer: 指定DOE Layer对象，即衍射光学元件层，用于在训练时生成PSF
            target_distance: 目标距离（带训练距离，仅用于景深扩展成像任务）
            input_channel_num: 输入通道，默认为31
            noise_model: 噪声模型
            depth_map: 深度图
            depth_list: 深度分段索引，需要和深度图对应深度的维度
            otfs: 预先输入的OTF
            should_use_planar_incidence: 是否使用平面光（使用平面光则默认深度无关，覆盖深度有关的设定）
            should_depth_dependent: 是否深度相关（若should_use_planar_incidence为True则默认深度无关）
            psf_mode: PSF Mode
            noise_sigma: standard deviation of the Gaussian noise
            name: 模型名称
        """
        super(Camera, self).__init__(name=name)
        self.noise_model = noise_model
        self.wave_resolution = wave_resolution
        self.wave_length_list = wave_length_list
        self.sensor_resolution = sensor_resolution

        self.input_sample_interval = input_sample_interval
        self.sensor_distance = sensor_distance
        self.sensor = sensor
        self.otfs = otfs
        self.psfs = None
        self.clean_psfs = None
        self.pre_sensor_image = None

        self.target_distance = target_distance

        self.target_psf = None
        self.input_channel_num = input_channel_num

        # 上采样亚像素编码设置
        self.psf_mode = psf_mode
        should_resize_image_to_psf_size = True

        if psf_mode == PSFMode.FULL_RESIZE:
            should_resize_image_to_psf_size = True
            Logger.w("PSF Resize模式已启用。")

        if should_resize_image_to_psf_size \
                and self.wave_resolution[0] == self.sensor_resolution[0] \
                and self.wave_resolution[1] == self.sensor_resolution[1]:
            should_resize_image_to_psf_size = False
            Logger.w("因 wave_resolution 和 sensor_resolution 大小一致，上采样设定已禁用。")
        if should_resize_image_to_psf_size and self.psf_mode == PSFMode.CROP_VALID:
            should_resize_image_to_psf_size = False
            Logger.w("因 PSFMode 设置为 CROP_VALID，上采样设定已禁用。")
        if should_resize_image_to_psf_size and \
                (self.psf_mode == PSFMode.FULL_PAD_IMAGE or
                 self.psf_mode == PSFMode.FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG):
            should_resize_image_to_psf_size = False
            Logger.w("因 PSFMode 设置为 FULL_PAD_IMAGE，上采样设定已禁用。")

        self.flag_resize_image_to_psf_size = should_resize_image_to_psf_size

        # 平面光/球面光生成准备
        self.simulated_incidence = None
        self.flag_use_planar_incidence = should_use_planar_incidence  # 默认球面光
        self.physical_size = float(self.wave_resolution[0] * self.input_sample_interval)
        Logger.i("Physical size is %0.2e m.\n Wave resolution is %d." % (self.physical_size, self.wave_resolution[0]))
        self.pixel_size = self.input_sample_interval * np.array(wave_resolution) / np.array(sensor_resolution)

        # 模拟生成PSF相关：提前构造传播模拟器(Fresnel)
        self.propagation = FresnelPropagation(distance=self.sensor_distance,
                                              discretization_size=self.input_sample_interval,
                                              wave_lengths=self.wave_length_list)

        self.invalid_energy_mask = None
        # PSF masks
        if psf_mode == PSFMode.CROP_VALID or psf_mode == PSFMode.FULL_PAD_IMAGE_WITH_SENSOR_LEAK_REG:
            from util.mask_generator import circle_mask
            self.invalid_energy_mask = circle_mask(full_square_length=wave_resolution[0],
                                                   inner_circle_radius=sensor_resolution[0] // 2)

        # DOE
        self.doe_layer = doe_layer
        self.freeze_doe_layer_weight = freeze_doe_layer_weight

        # 深度相关设置
        if should_depth_dependent and should_use_planar_incidence:
            should_depth_dependent = False
            Logger.w("深度相关仅适用于球面波，但当前相机的模拟入射光已被设置为平面光，将保持深度无关的设定。")
        elif should_depth_dependent and depth_list is None:
            Logger.w("使用球面波时必须指定depth bin，但现在您并未设置。")
        self.flag_depth_dependent = should_depth_dependent

        self.depth_map = depth_map
        self.depth_list = depth_list

        self.noise_sigma = noise_sigma
        self._input_shape = None

    def build(self, input_shape):
        self._input_shape = input_shape

    def done(self):
        """
        完成相机设置，并进行准备工作。
        相机（Camera）对象只有在done方法 被调用后，才可被作为 keras.layers.Layer 被调用

        Returns: 设置后的相机对象

        """
        # 提前存储入射场
        if self.flag_use_planar_incidence:
            # 平面光深度无关
            if self.depth_list is not None:
                Logger.w("指定使用平面光生成 PSF 时 depth_list 参数是无效的。此时的 PSF 是深度无关的。")
            self.simulated_incidence = planar_light(self.wave_resolution)
        else:
            # 点光源球面波深度相关
            assert self.depth_list is not None, "指定使用點光源生成 PSF 时 depth_list 参数是必須的。"
            self.simulated_incidence = point_source_of_light_spherical_wave_field(depth_list=self.depth_list,
                                                                                  target_distance=self.target_distance,
                                                                                  physical_size=self.physical_size,
                                                                                  wave_resolution=self.wave_resolution,
                                                                                  wave_lengths=self.wave_length_list)
        return self

    def psf_from_incidence(self, incidence, training=True, testing=None, with_clean_psf=False):
        # 模拟光线通过DOE层产生的相位差效果
        if with_clean_psf:
            field_after_height_map, clean_field_after_height_map = self.doe_layer(incidence, training=training,
                                                                                  testing=testing,
                                                                                  with_clean_modulation=True)
            field = circular_aperture(field_after_height_map)
            # 透过模拟传播和计算冲激响应获得点扩散函数
            sensor_incident_field = self.propagation(field)
            psf = get_intensities(sensor_incident_field)
            clean_field = circular_aperture(clean_field_after_height_map)
            # 透过模拟传播和计算冲激响应获得点扩散函数
            clean_sensor_incident_field = self.propagation(clean_field)
            clean_psf = get_intensities(clean_sensor_incident_field)
            return psf, clean_psf
        else:
            field_after_height_map = self.doe_layer(incidence, training=training, testing=testing,
                                                    with_clean_modulation=with_clean_psf)
            field = circular_aperture(field_after_height_map)
            # 透过模拟传播和计算冲激响应获得点扩散函数
            sensor_incident_field = self.propagation(field)
            psf = get_intensities(sensor_incident_field)
            return psf

    def _psf_post_process(self, psf, depth_index=0):
        if self.psf_mode == PSFMode.CROP_VALID:
            crop_length = (self.wave_resolution[0] - self.sensor_resolution[0]) // 2
            psf = psf[:, crop_length:-crop_length, crop_length:-crop_length, :]
            # 断言crop后的PSF有效区域和sensor区域等大
            tf.debugging.assert_equal(psf.shape[1], self.sensor_resolution[0],
                                      message="CROP_VALID模式crop后的PSF的高应与目标图像相同.")
            tf.debugging.assert_equal(psf.shape[2], self.sensor_resolution[1],
                                      message="CROP_VALID模式下crop后的PSF的宽应与目标图像相同.")

        # ============================================= PSF能量正则
        # 若配置，应用 PSF mask 添加能量约束正则化 loss (例如 crop_valid 模式)
        # if self.invalid_energy_mask is not None:
        #     psf = tf.math.divide(psf, tf.reduce_sum(input_tensor=psf, axis=[1, 2], keepdims=True),
        #                          name='psf_before_cropping_depth_idx_%d' % depth_idx)
        #     psf_invalid_energy = psf * self.invalid_energy_mask
        #     summary_hyper_spec_image(image_normalization(psf_invalid_energy),
        #                              name='PSF%d-InvalidEnergy' % depth_idx,
        #                              with_single_channel=False, norm_channel=True)
        #     psf_invalid_energy = tf.reduce_sum(psf_invalid_energy)
        #     tf.summary.scalar(name="psf_invalid_energy_mean", data=psf_invalid_energy)
        #     self.add_loss(psf_invalid_energy)
        # !============================================= PSF能量正则
        # if self.target_distance is not None:
        #     self.target_psf = psfs.pop()

        # 修改：tf.compat.v1.div => tf.math.divide => divide_no_nan
        # 保持每个通道能量总和为1
        psf = tf.math.divide(psf, tf.reduce_sum(input_tensor=psf, axis=[1, 2], keepdims=True),
                             name='psf_depth_idx_%d' % depth_index)

        # 记录 PSF
        summary_hyper_spec_image(image_normalization(psf), name='PSFNormed%d' % depth_index,
                                 with_single_channel=True, norm_channel=True)

        transposed_psf = tf.transpose(a=psf, perm=[1, 2, 0, 3], name="transposed_psf_%d" % depth_index)
        # 转置为 (height, width, batch=1, channels) 便于后续使用 tf-fft
        return transposed_psf

    def generate_psf_from(self, input_fields, training=True, testing=None, with_clean_psf=False):
        assert self.doe_layer is not None, "Lens未被指定DOE，请调用attach()方法为其添加一个DOE。"
        psfs = []
        clean_psfs = []
        for depth_idx, input_field in enumerate(input_fields):
            if with_clean_psf:
                psf, clean_psf = self.psf_from_incidence(input_field, training=training, testing=testing,
                                                         with_clean_psf=True)
                clean_psfs.append(self._psf_post_process(clean_psf))
            else:
                psf = self.psf_from_incidence(input_field, training=training, testing=testing)
            psfs.append(self._psf_post_process(psf))

        if with_clean_psf:
            return psfs, clean_psfs
        else:
            return psfs

    def get_pre_sensor_image(self, input_img, psfs):
        depth_map = self.depth_map
        if self.flag_depth_dependent:
            # if self.flag_do_up_sampling:
            #     depth_map = tf.image.resize(depth_map, self.wave_resolution,
            #                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            sensor_img = depth_dep_convolution(input_img, psfs, disc_depth_map=depth_map)
        else:
            sensor_img = image_convolve_with_psf(input_img, psfs[0], otf=self.otfs, img_shape=self._input_shape)
        # if self.flag_do_up_sampling:  # Down sample measured image to match sensor resolution.
        #     sensor_img = area_downsampling_tf(sensor_img, self.sensor_resolution[0])
        sensor_img = tf.cast(sensor_img, tf.float32)
        return sensor_img

    def prepare_psfs(self, with_ideal_clean_encoded_image, training=None, testing=None):
        if with_ideal_clean_encoded_image:
            assert not self.flag_resize_image_to_psf_size, "PSF sizing is not supported when using dual PSFs."
            self.psfs, self.clean_psfs = self.generate_psf_from(input_fields=self.simulated_incidence,
                                                                training=training,
                                                                testing=testing, with_clean_psf=True)
        else:
            self.psfs = self.generate_psf_from(input_fields=self.simulated_incidence, training=training,
                                               testing=testing)

    def call(self, inputs, training=None, testing=None, with_ideal_clean_encoded_image=False, **kwargs):
        """
        Args:
            with_ideal_clean_encoded_image:
            inputs: 输入图像tensor，若大小和wave resolution不一致，将被resize到wave resolution进行处理
            training: 是否训练阶段
            testing: 是否测试阶段

        Returns: 透过此镜头后并模拟指定的传感器所成的像，大小和sensor_resolution一致
        """
        assert self.simulated_incidence is not None, "[Camera] 光源输入场为空。请在Camera模型参数配置完成后调用done()方法。"
        if self.sensor is None:
            Logger.e("[Warning]: in Camera; 未指定相机传感器类型。")

        # if not self.freeze_doe_layer_weight or (self.freeze_doe_layer_weight and self.psfs is None):
        self.prepare_psfs(with_ideal_clean_encoded_image=with_ideal_clean_encoded_image,
                              training=training, testing=testing)

        if self.flag_resize_image_to_psf_size:
            # 断言此时的输入大小不等,若不然则其他代码逻辑恐未按预期工作
            tf.debugging.assert_none_equal(inputs.shape[1], self.wave_resolution[0],
                                           message="此时的输入大小应不等,否则flag_do_up_sampling不应为True."
                                                   "其他部分代码逻辑恐未按预期工作,请检查.")
            # 上采样数据集以贴合wave resolution(亦是psf大小).
            inputs = tf.image.resize(inputs, self.wave_resolution,
                                     method=tf.image.ResizeMethod.BILINEAR)
            tf.debugging.assert_equal(inputs.shape[1], self.wave_resolution[0],
                                      message="此时的输入大小应相等。"
                                              "其他部分代码逻辑恐未按预期工作,请检查.")

        if with_ideal_clean_encoded_image:
            self.pre_sensor_image = self.get_pre_sensor_image(input_img=inputs, psfs=self.psfs)
            clean_sensor_image = self.get_pre_sensor_image(input_img=inputs, psfs=self.clean_psfs)
        else:
            self.pre_sensor_image = self.get_pre_sensor_image(input_img=inputs, psfs=self.psfs)

        if self.flag_resize_image_to_psf_size:
            # 下采样样数据集以贴合wave resolution(亦是psf大小).
            # sensor_image = area_downsampling_tf(sensor_image, self.sensor_resolution[0])
            self.pre_sensor_image = tf.image.resize(self.pre_sensor_image, self.sensor_resolution,
                                                    method=tf.image.ResizeMethod.BILINEAR)

        if training and self.noise_sigma is not None:
            pre_sensor_image = self.noise_model(image=self.pre_sensor_image, std_dev=self.noise_sigma)
            Logger.i("[Camera] Addictive noise: sigma=%0.2e" % self.noise_sigma)
        else:
            pre_sensor_image = self.pre_sensor_image
            Logger.i("[Camera] No addictive sensor noise will be added.")

        if self.sensor is not None:
            sensor_image = self.sensor(pre_sensor_image)
            if with_ideal_clean_encoded_image:
                clean_sensor_image = self.sensor(clean_sensor_image)
        else:
            sensor_image = pre_sensor_image

        if self.sensor is not None:
            tf.summary.image(name="SensorImage", data=sensor_image, max_outputs=1)

        if with_ideal_clean_encoded_image:
            tf.summary.image(name="CleanPreSensorImage", data=sensor_image, max_outputs=1)
            return sensor_image, clean_sensor_image
        else:
            return sensor_image

    def get_config(self):
        config = super(Camera, self).get_config()
        config.update({
            "wave_resolution": self.wave_resolution,
            "wave_length_list": self.wave_length_list,
            "sensor_distance": self.sensor_distance,
            "sensor_resolution": self.sensor_resolution,
            "input_sample_interval": self.input_sample_interval,
            "doe_layer": self.doe_layer,
            "target_distance": self.target_distance,
            "input_channel_num": self.input_channel_num,
            "noise_model": self.noise_model,
            "depth_map": self.depth_map,
            "depth_bin": self.depth_list,
            "otfs": self.otfs,
            "should_use_planar_incidence": self.flag_use_planar_incidence,
            "should_do_up_sampling": self.flag_resize_image_to_psf_size,
            "should_depth_dependent": self.flag_depth_dependent,
            "name": "camera"})
        return config


class PSFFixedCamera(tf.keras.layers.Layer):
    def __init__(self, wave_resolution=(1024, 1024), sensor_resolution=(512, 512), psf_mat_file_path_and_key=None,
                 noise_sigma=0.001, height_map_noise_sigma=None):
        super(PSFFixedCamera, self).__init__()
        self._input_shape = None
        self.psf = None
        self.flag_resize_image_to_psf_size = False
        self.sensor_resolution = sensor_resolution
        self.wave_resolution = wave_resolution
        self.psf_mat_file_path_and_key = psf_mat_file_path_and_key
        self.response_curve = None
        self.noise_sigma = noise_sigma
        import scipy.io as sio
        psf_mat = sio.loadmat(self.psf_mat_file_path_and_key[0])[self.psf_mat_file_path_and_key[1]]
        Logger.i("[PSFFixedCamera] Loading PSF from MAT file {} -> [{}]".format(self.psf_mat_file_path_and_key[0],
                                                                                self.psf_mat_file_path_and_key[1]))
        self.psf = self.add_weight(name="PSF", shape=(1, 1024, 1024, 31),
                                   dtype=tf.float32, initializer=tf.constant_initializer(value=psf_mat),
                                   trainable=False)
        self.psf = tf.image.resize(self.psf, (512, 512), method=tf.image.ResizeMethod.BILINEAR)
        self.psf = tf.math.divide(self.psf, tf.reduce_sum(self.psf, axis=[1, 2], keepdims=True))
        # pad_length = (2048 - 512) // 2
        # self.psf = tf.pad(self.psf, paddings=[[0, 0], [pad_length, pad_length], [pad_length, pad_length], [0, 0]])
        self.psf = tf.transpose(self.psf, perm=[1, 2, 0, 3])  # h, w, 1, 31

    def build(self, input_shape):
        self._input_shape = input_shape

    def call(self, inputs, training=None, testing=None, with_ideal_clean_encoded_image=False, **kwargs):

        if self.flag_resize_image_to_psf_size:
            # 上采样数据集以贴合wave resolution(亦是psf大小).
            inputs = tf.image.resize(inputs, self.wave_resolution,
                                     method=tf.image.ResizeMethod.BILINEAR)
            tf.debugging.assert_equal(inputs.shape[1], self.wave_resolution[0],
                                      message="此时的输入大小应相等。"
                                              "其他部分代码逻辑恐未按预期工作,请检查.")
        hyper_sensor_img = image_convolve_with_psf(inputs, self.psf, otf=None, img_shape=self._input_shape)

        from optics.sensor_srfs import simulated_rgb_camera_spectral_response_function
        sensor_img = simulated_rgb_camera_spectral_response_function(hyper_sensor_img)
        if self.flag_resize_image_to_psf_size:
            sensor_img = tf.image.resize(sensor_img, self.sensor_resolution, method=tf.image.ResizeMethod.BILINEAR)

        if training:
            sensor_img = gaussian_noise(image=sensor_img, std_dev=self.noise_sigma)

        summary_hyper_spec_image(image=tf.transpose(self.psf, perm=[2, 0, 1, 3]), name="PSF",
                                 with_single_channel=True, norm_channel=True)
        return sensor_img


def circular_aperture(input_field):
    input_shape = input_field.shape.as_list()
    [x, y] = np.mgrid[-input_shape[1] // 2: input_shape[1] // 2, -input_shape[2] // 2: input_shape[2] // 2].astype(
        np.float64)

    max_val = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None, :, :, None]
    aperture = (r < max_val).astype(np.float64)
    return aperture * input_field


# 入射光点光源场模拟函数
# 平面波
def planar_light(wave_resolution):
    return [tf.ones(wave_resolution, dtype=tf.float32)[None, :, :, None]]


# 点光源
def point_source_of_light_spherical_wave_field(depth_list, target_distance, physical_size, wave_resolution,
                                               wave_lengths, point_location=(0, 0)):
    distances = depth_list
    if distances is None:
        distances = []
    if target_distance is not None:
        distances += [target_distance]
    wave_res_n, wave_res_m = wave_resolution
    [x, y] = np.mgrid[-wave_res_n // 2:wave_res_n // 2, -wave_res_m // 2:wave_res_m // 2].astype(np.float64)
    x = x / wave_res_n * physical_size
    y = y / wave_res_m * physical_size

    x0, y0 = point_location
    squared_sum = (x - x0) ** 2 + (y - y0) ** 2
    # squared_sum = x ** 2 + y ** 2
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1, 1, 1, -1])

    input_fields = []
    # edit: depth-invariant psf
    for distance in distances:
        # Assume distance to source is approx. constant over wave
        curvature = tf.sqrt(squared_sum + tf.cast(distance, tf.float64) ** 2)
        curvature = tf.expand_dims(tf.expand_dims(curvature, 0), -1)
        spherical_wavefront = complex_exponent_tf(wave_nos * curvature, dtype=tf.complex64)
        input_fields.append(spherical_wavefront)
    return input_fields


def get_intensities(input_field):
    return tf.square(tf.abs(input_field), name='intensities')
