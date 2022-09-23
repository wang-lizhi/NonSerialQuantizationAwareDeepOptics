import tensorflow as tf

import constants
from log import Logger
from optics.camera import PSFFixedCamera


def process_possible_wave_length_str(wave_length_list):
    if isinstance(wave_length_list, str):
        if wave_length_list == "rgb":
            wave_length_list = constants.wave_length_list_rgb
        elif wave_length_list == "400-700":
            wave_length_list = constants.wave_length_list_400_700nm
        elif wave_length_list == "420-720":
            wave_length_list = constants.wave_length_list_420_720nm
        else:
            assert False, "Invalid wave length list string."
    Logger.i("Wavelengths to be used：", wave_length_list)
    return wave_length_list


def build_doe_layer(doe_layer_type, doe_extra_args, wave_resolution, wave_length_list, sample_interval,
                    wavelength_to_refractive_index_func_name, height_map_noise):
        from constants import MATERIAL_REFRACTIVE_INDEX_FUNCS
        wavelength_to_refractive_index_func = \
            MATERIAL_REFRACTIVE_INDEX_FUNCS[wavelength_to_refractive_index_func_name]
        assert wavelength_to_refractive_index_func is not None, \
            "Unsupported doe_material argument. It should be in: " + str(
                MATERIAL_REFRACTIVE_INDEX_FUNCS.keys())

        doe_general_args = {
            "wave_length_list": wave_length_list,
            "wavelength_to_refractive_index_func": wavelength_to_refractive_index_func,
            "height_tolerance": height_map_noise,
        }

        print("\n\n==============>DOE Args<===============")
        print("General:")
        print(doe_general_args)
        print("Extra:")
        print(doe_extra_args)
        print("==============<DOE Args>===============\n\n")

        height_map_regularizer = None
        swapped_height_map_regularizer_string = None

        doe_layer = None

        if "height_map_regularizer" in doe_extra_args:
            from regularizers import REGULARIZER_MAP
            if doe_extra_args["height_map_regularizer"] in REGULARIZER_MAP:
                height_map_regularizer = REGULARIZER_MAP[doe_extra_args["height_map_regularizer"]]
                swapped_height_map_regularizer_string = doe_extra_args["height_map_regularizer"]
            del doe_extra_args["height_map_regularizer"]

        if doe_layer_type == "rank1":
            from optics.diffractive_optical_element import Rank1HeightMapDOELayer
            doe_layer = Rank1HeightMapDOELayer(height_map_regularizer=height_map_regularizer,
                                               **doe_general_args,
                                               **doe_extra_args)
        elif doe_layer_type == 'htmp':
            from optics.diffractive_optical_element import HeightMapDOELayer
            doe_layer = HeightMapDOELayer(height_map_regularizer=height_map_regularizer,
                                          **doe_general_args,
                                          **doe_extra_args)
        elif doe_layer_type == 'htmp-quant':
            from optics.diffractive_optical_element import QuantizedHeightMapDOELayer
            doe_layer = QuantizedHeightMapDOELayer(
                height_map_regularizer=height_map_regularizer,
                **doe_general_args,
                **doe_extra_args)
        elif doe_layer_type == 'htmp-quant-quad':
            from optics.diffractive_optical_element import QuadSymmetricQuantizedHeightMapDoeLayer
            doe_layer = QuadSymmetricQuantizedHeightMapDoeLayer(
                height_map_regularizer=height_map_regularizer,
                **doe_general_args,
                **doe_extra_args)
        elif doe_layer_type == 'htmp-quant-sym':
            from optics.diffractive_optical_element import RotationallySymmetricQuantizedHeightMapDOELayer
            doe_layer = RotationallySymmetricQuantizedHeightMapDOELayer(
                height_map_regularizer=height_map_regularizer,
                **doe_general_args,
                **doe_extra_args)
        elif doe_layer_type == 'htmp-fixed':
            from optics.diffractive_optical_element import FixedDOELayer
            doe_layer = FixedDOELayer(
                **doe_general_args,
                **doe_extra_args
            )

        if swapped_height_map_regularizer_string is not None:
            Logger.i("Storing `height_map_regularizer` from `swapped_height_map_regularizer_string`.")
            doe_extra_args["height_map_regularizer"] = swapped_height_map_regularizer_string

        return doe_layer


class NSQDOModel(tf.keras.Model):
    def get_config(self):
        config = super(NSQDOModel, self).get_config()
        return config

    def __init__(self, image_patch_size, sensor_distance, wavelength_to_refractive_index_func_name, wave_resolution,
                 sample_interval, input_channel_num, doe_layer_type, depth_bin,
                 wave_length_list=constants.wave_length_list_400_700nm,
                 reconstruction_network_type=None, reconstruction_network_args=None,
                 use_psf_fixed_camera=False, extra_mat_file_path_and_key=None, freeze_doe_layer_weight=False,
                 srf_type=None, doe_extra_args=None, height_map_noise=None, skip_optical_encoding=False,
                 noise_sigma=0.001, *args, **kwargs):
        super().__init__(*args, **kwargs)

        wave_length_list = process_possible_wave_length_str(wave_length_list)

        if doe_extra_args is None:
            doe_extra_args = {}

        self.image_patch_size = image_patch_size
        self.sensor_distance = sensor_distance
        self.wavelength_to_refractive_index_func_name = wavelength_to_refractive_index_func_name

        self.wave_length_list = wave_length_list
        self.sample_interval = sample_interval
        self.wave_resolution = wave_resolution
        self.input_channel_num = input_channel_num

        self.doe_layer_type = doe_layer_type
        self.doe_layer = None
        self.depth_bin = depth_bin
        self.freeze_doe_layer_weight = freeze_doe_layer_weight

        self.reconstruction_network_type = reconstruction_network_type

        if reconstruction_network_type == "res_block_u_net":
            from networks.res_block_u_net import get_res_block_u_net
            self.reconstruction_network = get_res_block_u_net(**reconstruction_network_args)
        elif reconstruction_network_type == "non_serial_decoder":
            from networks.non_serial import NonSerialDecoder
            self.reconstruction_network = NonSerialDecoder(**reconstruction_network_args)

        self.srf_type = srf_type
        self._input_shape = None
        self.height_map_noise = height_map_noise

        self.skip_optical_encoding = skip_optical_encoding
        self.use_psf_fixed_camera = use_psf_fixed_camera

        if not skip_optical_encoding \
                or "unfolding" in reconstruction_network_type:
            if not use_psf_fixed_camera:
                self.doe_layer = build_doe_layer(doe_layer_type, doe_extra_args, wave_resolution, wave_length_list,
                                                 sample_interval, wavelength_to_refractive_index_func_name,
                                                 height_map_noise)

                assert self.doe_layer is not None, "DOE layer is required."
                from optics.camera import Camera
                from optics.sensor import Sensor

                sensor = None
                if srf_type is not None:
                    sensor = Sensor(srf_type=srf_type)

                Logger.i("Using normal camera...")
                self.optical_system = Camera(wave_resolution=self.wave_resolution,
                                             wave_length_list=self.wave_length_list,
                                             sensor_distance=self.sensor_distance,
                                             sensor_resolution=(self.image_patch_size, self.image_patch_size),
                                             sensor=sensor,
                                             input_sample_interval=self.sample_interval,
                                             doe_layer=self.doe_layer,
                                             input_channel_num=self.input_channel_num,
                                             depth_list=depth_bin, should_use_planar_incidence=False,
                                             should_depth_dependent=False, noise_sigma=noise_sigma,
                                             freeze_doe_layer_weight=freeze_doe_layer_weight).done()
                Logger.i("The flag `freeze_doe_layer_weight`=", self.freeze_doe_layer_weight)
                if self.freeze_doe_layer_weight:
                    Logger.w("DOE layer weights are all frozen.")
                    self.optical_system.doe_layer.trainable = False
            else:
                Logger.w("Using `PSFFixedCamera`.")
                from optics.camera import PSFFixedCamera
                self.optical_system = PSFFixedCamera(wave_resolution=wave_resolution,
                                                     sensor_resolution=(self.image_patch_size, self.image_patch_size),
                                                     psf_mat_file_path_and_key=extra_mat_file_path_and_key,
                                                     noise_sigma=noise_sigma)
        else:
            Logger.w("Optical encoder is going to be disabled.")
            if self.reconstruction_network_type == "non_serial_decoder" \
                    or self.reconstruction_network_type == "unfolding_decoder":
                Logger.w("However, an optical encoder will still be initialized for unrolling decoder to use the PSF.")
                from optics.camera import Camera
                # self.optical_system = PSFFixedCamera(wave_resolution=wave_resolution,
                #                                      sensor_resolution=(self.image_patch_size, self.image_patch_size),
                #                                      psf_mat_file_path_and_key=extra_mat_file_path_and_key,
                #                                      noise_sigma=noise_sigma)
                self.doe_layer = build_doe_layer(doe_layer_type, doe_extra_args, wave_resolution, wave_length_list,
                                                 sample_interval, wavelength_to_refractive_index_func_name,
                                                 height_map_noise)
                self.optical_system = Camera(wave_resolution=self.wave_resolution,
                                             wave_length_list=self.wave_length_list,
                                             sensor_distance=self.sensor_distance,
                                             sensor_resolution=(self.image_patch_size, self.image_patch_size),
                                             sensor=None,
                                             input_sample_interval=self.sample_interval,
                                             doe_layer=self.doe_layer,
                                             input_channel_num=self.input_channel_num,
                                             depth_list=depth_bin, should_use_planar_incidence=False,
                                             should_depth_dependent=False, noise_sigma=noise_sigma,
                                             freeze_doe_layer_weight=freeze_doe_layer_weight).done()
                Logger.w("The optical_system object has been loaded because `non_serial_decoder` depends on it.")
            else:
                self.optical_system = None
                Logger.w("Optical encoder has been disabled.")

        self.model_description = "DOE{}_SpItv{}_SsDst{}_ScDst{}_WvRes{}_ImgSz{}_SRF{}" \
            .format(doe_layer_type, sample_interval, sensor_distance, depth_bin[0], wave_resolution[0],
                    image_patch_size, srf_type)

    def train_step(self, data):
        return super(NSQDOModel, self).train_step((data, data))

    def test_step(self, data):
        return super(NSQDOModel, self).test_step((data, data))

    def call(self, inputs, training=None, testing=None, **kwargs):
        if not self.skip_optical_encoding:
            x = self.optical_system(inputs, with_ideal_clean_encoded_image=False, training=training,
                                    testing=testing)
        elif self.reconstruction_network_type in {"non_serial_decoder", "unfolding_decoder"}:
            Logger.w("Mocked input tensor for non-serial decoder to test.")
            input_b, input_h, input_w, input_c = inputs.shape
            mock_input = tf.zeros(shape=(input_b, 512, 512, self.input_channel_num))
            self.optical_system(mock_input, with_ideal_clean_encoded_image=False, training=training,
                                testing=testing)

            Logger.w("Skipped optical encoder. The input tensor is directly used for decoding.")
            x = inputs
        else:
            assert False, "Unsupported reconstruction network type for mocking input tensor."

        if training:
            tf.summary.image(name="SensorImage", data=x, max_outputs=1)

        if self.reconstruction_network is not None:
            if self.reconstruction_network_type in ("unfolding_decoder", "non_serial_decoder"):
                if self.use_psf_fixed_camera:
                    assert isinstance(self.optical_system, PSFFixedCamera)
                    _psf = self.optical_system.psf
                else:
                    assert not isinstance(self.optical_system, PSFFixedCamera)
                    _psf = self.optical_system.psfs[0]
                # Double inputs for unfolding_decoder
                Logger.d("Shape of PSF for unfolding decoder: ", _psf.shape)
                if self.reconstruction_network_type == "non_serial_decoder":
                    x = x, tf.transpose(a=_psf, perm=[2, 0, 1, 3]), inputs
                else:
                    x = x, tf.transpose(a=_psf, perm=[2, 0, 1, 3])
            reconstructed = self.reconstruction_network(x)

            if training:
                from summary import summary_hyper_spec_image
                summary_hyper_spec_image(reconstructed, name="ReconstructedImage")

            return reconstructed
        else:
            return x
