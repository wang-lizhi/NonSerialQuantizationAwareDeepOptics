def process_controlled_args(controlled_training_args, controlled_model_args, doe_material, with_doe_noise,
                            quantization_level, quantize_at_test_only, alpha_blending, adaptive_quantization,
                            sensor_distance_mm, scene_depth_m, init_learning_rate):
    note = "-%s-%sLR-SamInt%s" % (doe_material, str(init_learning_rate), str(controlled_model_args["sample_interval"]))

    if quantize_at_test_only:
        assert not alpha_blending, \
            "When `quantize_at_test_only` is enabled, the QDO option `alpha_blending` must not be True."
        assert not adaptive_quantization, \
            "When `quantize_at_test_only` is enabled, the QDO+A option `alpha_blending` must not be True."
        note += "-FullTrain"
    else:
        assert not (adaptive_quantization and not alpha_blending), \
            "Enabling `adaptive_quantization` while disabling `alpha_blending` is invalid."
        note += ("-AdaAB" if adaptive_quantization else "-NoAda-AB") if alpha_blending else "-STE"
        note += "-" + str(quantization_level) + "Lv"



    controlled_model_args["default_optimizer_learning_rate_args"]["initial_learning_rate"] = init_learning_rate
    controlled_model_args["wavelength_to_refractive_index_func_name"] = doe_material
    if not with_doe_noise:
        controlled_model_args["height_map_noise"] = None
    else:
        assert controlled_model_args["height_map_noise"] is not None

    controlled_model_args["sensor_distance"] = sensor_distance_mm * 1e-3
    note += "-Sd%dmm" % sensor_distance_mm
    controlled_model_args["depth_bin"] = [scene_depth_m]
    note += "-Sc%dm" % scene_depth_m

    if controlled_model_args["reconstruction_network_type"] == "unfolding_decoder":
        note += "-UnfCG0ResUNetDp%dItr%d" % \
                (controlled_model_args["reconstruction_network_args"]["regularization_module_args"]["depth"],
                 controlled_model_args["reconstruction_network_args"]["iteration"])
        if not controlled_model_args["reconstruction_network_args"]["is_eta_trainable"]:
            note += "-FixedEta"
        if not controlled_model_args["reconstruction_network_args"]["is_eps_trainable"]:
            note += "-FixedEps"
        if controlled_model_args["reconstruction_network_args"]["shared_regularization_module"]:
            note += "-SharedRegModWeight"
        if controlled_model_args["reconstruction_network_args"]["use_flexible_srf"]:
            note += "FlexSRF"
        if controlled_model_args["reconstruction_network_args"]["cut_off_psf_grad"]:
            note += "-Serial"
        else:
            note += "-NonSerial"

    if "use_psf_fixed_camera" in controlled_model_args \
            and controlled_model_args["use_psf_fixed_camera"] is True:
        note += "-PSFFixedCam"
    elif controlled_model_args["doe_layer_type"] != "htmp-fixed":
        controlled_model_args["doe_extra_args"].update({
            "quantization_level_cnt": quantization_level,
            "quantize_at_test_only": quantize_at_test_only,
            "adaptive_quantization": adaptive_quantization,
            "alpha_blending": alpha_blending,
            "step_per_epoch": controlled_training_args["step_per_epoch"]
        })

    controlled_training_args["note"] = note

    return controlled_training_args, controlled_model_args
