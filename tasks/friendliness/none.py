from metrics import ssim_metric, psnr_metric, sam_metric, psnr_hyper_metric, ergas_metric
from optics.sensor_srfs import SRF_OUTPUT_SIZE_LAMBDA
from tasks.util import process_controlled_args

import trainer

image_patch_size = 512
doe_resolution = 512
srf_type = "rgb"

network_input_size = SRF_OUTPUT_SIZE_LAMBDA[srf_type](image_patch_size)

batch_size = 4
step_per_epoch = 1952 // batch_size  # * ICVL 512 MAT 1670 -> 1672

controlled_training_args = {
    "task_name": "HS", "dataset_name": "ICVL512-MAT", "loss": 'mae',
    "metrics": [ssim_metric, psnr_metric, psnr_hyper_metric, sam_metric, ergas_metric],
    "checkpoint_monitor": "psnr_hyper_metric", "checkpoint_save_mode": "max",
    "training_batch_size": batch_size,
    "step_per_epoch": step_per_epoch, "total_epoch": 50,
    "summary_update_freq": step_per_epoch // 3,
    "dataset_loader_func_name": "load_icvl_full_mat_512",
    "save_freq_epoch": 5,
    "global_random_seed": 259387618,
    "validation_steps": 170 // batch_size
}

controlled_model_args = {
    "model_class_name": "NSQDOModel",
    "image_patch_size": image_patch_size, "sensor_distance": 50e-3,
    "wavelength_to_refractive_index_func_name": "SK1300",
    "sample_interval": 8e-6,
    "wave_resolution": (doe_resolution, doe_resolution),
    "input_channel_num": 31, "depth_bin": [1],
    "doe_layer_type": "htmp-quant-sym",
    "srf_type": srf_type,

    "doe_extra_args": {
        "alpha_blending_start_epoch": 5,
        "alpha_blending_end_epoch": 40,
    },

    "default_optimizer_learning_rate_args": {
        "initial_learning_rate": 0.005, "decay_steps": 500, "decay_rate": 0.8, "name": "default_opt_lr"},

    "reconstruction_network_type": "non_serial_decoder",
    "reconstruction_network_args": {
        "iteration": 4, "is_eta_trainable": True, "is_eps_trainable": True,
        "eta_init": 0.1, "eps_init": 0.8, "bhc_type": "disabled",
        "regularization_module_args": {
            "filter_root": 32, "depth": 4, "output_channel": 31, "input_size": (image_patch_size, image_patch_size, 31),
            "activation": 'elu', "batch_norm": True, "batch_norm_after_activation": False,
            "final_activation": 'sigmoid', "net_num": 1, "extra_upsampling": False,
            "remove_first_long_connection": False, "channel_attention": False
        }
    },
    "height_map_noise": None,
    "noise_sigma": None
}


def train(doe_material=None, with_doe_noise=False, quantization_level=None, quantize_at_test_only=False,
          alpha_blending=False, adaptive_quantization=False, checkpoint=None, continue_training=False,
          tag=None, sensor_distance_mm=None, scene_depth_m=None, init_learning_rate=0.002):

    # OVERRIDE ARGS
    # None (Conventional DO)
    quantize_at_test_only = True
    alpha_blending = False
    adaptive_quantization = False

    assert not with_doe_noise, "This task only accept options that disable noise"
    assert controlled_model_args["height_map_noise"] is None, "This task only accept options that disable noise"
    assert controlled_model_args["noise_sigma"] is None, "This task only accept options that disable noise"

    processed_training_args, processed_model_args = process_controlled_args(controlled_training_args,
                                                                            controlled_model_args,
                                                                            doe_material, with_doe_noise,
                                                                            quantization_level,
                                                                            quantize_at_test_only,
                                                                            alpha_blending,
                                                                            adaptive_quantization,
                                                                            sensor_distance_mm,
                                                                            scene_depth_m,
                                                                            init_learning_rate)

    trainer.train(
        exp_group_tag=tag,
        controlled_model_args=processed_model_args,
        controlled_training_args=processed_training_args,
        pretrained_checkpoint_path_to_load=checkpoint,
        continue_training=continue_training,
        shuffle_training_set=True
        # xla=True,
    )
