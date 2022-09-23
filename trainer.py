import json
import os
from datetime import datetime

import tensorflow as tf

from log import Logger


def prepare_image_dataset(task_name, dataset_name, dataset_loader_func_name, training_batch_size, repeat_val_set=True,
                          prefetch_training_set=True, shuffle_training_set=False):
    from util.data.dataset_loader import DATASET_PATH
    root_dir = DATASET_PATH[dataset_name]

    from importlib import import_module
    dataset_loader_func = getattr(import_module("util.data.dataset_loader"), dataset_loader_func_name)
    assert dataset_loader_func is not None, "Invalid dataset_loader_func_name."

    if "PatchDiv" in dataset_name:
        full_dataset_dir = root_dir
        if not os.path.exists(full_dataset_dir):
            full_dataset_dir = full_dataset_dir.replace("/lilingen/lilingen/", "/lilingen/")
        # In the PatchDiv mode, training set and validation set are from the same loader function
        training_pairs, validation_pairs = dataset_loader_func(full_dataset_dir,
                                                               cache_name="./cache/%s-%s" % (task_name, dataset_name))
        training_pairs = training_pairs.batch(training_batch_size).repeat()
        validation_pairs = validation_pairs.batch(training_batch_size)
    else:
        train_dir = root_dir + "/train"
        val_dir = root_dir + "/validation"
        if not os.path.exists(train_dir):
            train_dir = train_dir.replace("/lilingen/lilingen/", "/lilingen/")
            val_dir = val_dir.replace("/lilingen/lilingen/", "/lilingen/")
        Logger.i("Preparing training datasets from {}...".format(train_dir))
        training_pairs = dataset_loader_func(train_dir,
                                             cache_name="./cache/%s-%s-train" % (task_name, dataset_name)) \
            .batch(training_batch_size).repeat()

        Logger.i("Preparing validation datasets from {}...".format(val_dir))
        validation_pairs = dataset_loader_func(val_dir,
                                               cache_name="./cache/%s-%s-validation" % (task_name, dataset_name)) \
            .batch(training_batch_size)

    if repeat_val_set:
        validation_pairs = validation_pairs.repeat()

    if prefetch_training_set:
        training_pairs = training_pairs.prefetch(tf.data.AUTOTUNE)

    if shuffle_training_set:
        training_pairs = training_pairs.shuffle(buffer_size=training_batch_size)

    return training_pairs, validation_pairs


def print_args_summary(model_args, training_args):
    Logger.i("\n\n==============>Controlled Args<==============="
             "\n>>Model: \n", model_args, "\n >>Training:", training_args, )


def model_factory(controlled_model_args, loss, metrics, training_batch_size):
    model = None
    model_class_name = controlled_model_args["model_class_name"]
    del controlled_model_args["model_class_name"]
    Logger.i("The `model_class_name` = %s" % model_class_name)

    if model_class_name is None or model_class_name == "LearnedDiffractiveOpticsModel":
        from model import LearnedDiffractiveOpticsModel
        model = LearnedDiffractiveOpticsModel(**controlled_model_args)
        model.build(input_shape=(training_batch_size,
                                 controlled_model_args["image_patch_size"],
                                 controlled_model_args["image_patch_size"],
                                 controlled_model_args["input_channel_num"]))
        model.compile(loss=loss, metrics=metrics)

    elif model_class_name == "NSQDOModel":
        from model import NSQDOModel
        optimizer_args = controlled_model_args["default_optimizer_learning_rate_args"]
        del controlled_model_args["default_optimizer_learning_rate_args"]
        model = NSQDOModel(**controlled_model_args)
        controlled_model_args["default_optimizer_learning_rate_args"] = optimizer_args
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        model.build(input_shape=(training_batch_size,
                                 controlled_model_args["image_patch_size"],
                                 controlled_model_args["image_patch_size"],
                                 controlled_model_args["input_channel_num"]))
        model.compile(loss=loss, metrics=metrics, optimizer=Adam(learning_rate=ExponentialDecay(**optimizer_args)))

    return model


def train(model=None, exp_group_tag=None, controlled_model_args=None, controlled_training_args=None,
          pretrained_checkpoint_path_to_load=None, continue_training=False, xla=False, shuffle_training_set=False,
          recompile=False):
    task_name = controlled_training_args["task_name"]
    dataset_name = controlled_training_args["dataset_name"]
    dataset_loader_func_name = controlled_training_args["dataset_loader_func_name"]
    loss = controlled_training_args["loss"]
    metrics = controlled_training_args["metrics"]
    note = controlled_training_args["note"]
    training_batch_size = controlled_training_args["training_batch_size"]
    step_per_epoch = controlled_training_args["step_per_epoch"]
    total_epoch = controlled_training_args["total_epoch"]
    checkpoint_monitor = controlled_training_args["checkpoint_monitor"]
    checkpoint_save_mode = controlled_training_args["checkpoint_save_mode"]

    summary_update_freq = controlled_training_args["summary_update_freq"]
    validation_steps = None if "validation_steps" not in controlled_training_args.keys() \
        else controlled_training_args["validation_steps"]

    assert controlled_model_args is not None, "The `controlled_model_args` must not be None."
    assert controlled_training_args is not None, "The `controlled_training_args` must not be None."

    global_random_seed = None if "global_random_seed" not in controlled_training_args.keys() \
        else controlled_training_args["global_random_seed"]

    if global_random_seed is not None:
        Logger.w("A random seed will be set globally. Seed value=%d." % global_random_seed)
        tf.random.set_seed(global_random_seed)
    else:
        Logger.i("No random seed is given. Current training process might not be reproducible due to the randomness.")

    if xla:
        # Enable TF XLA
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    # Mixed Precision
    # from tensorflow.keras.mixed_precision import experimental as mixed_precision
    # policy = mixed_precision.Policy('mixed_float16')
    # mixed_precision.set_policy(policy)
    if continue_training:
        assert pretrained_checkpoint_path_to_load is not None, "Checkpoint file must be given to continue training."
        split_path = pretrained_checkpoint_path_to_load.split("/")
        train_log_dir_name = split_path[-3] + "/" + split_path[-2]
    else:
        train_log_dir_name = datetime.now().strftime("%Y%m%d-%H%M%S") + note
        if exp_group_tag is not None and exp_group_tag != "":
            Logger.i("Using group tag: ", exp_group_tag)
            train_log_dir_name = exp_group_tag + "/" + train_log_dir_name

    Logger.i("[DIR] training log directory name = ", train_log_dir_name)
    log_dir = './logs/' + train_log_dir_name

    # Dataset
    train_pairs, validation_pairs = prepare_image_dataset(task_name=task_name,
                                                          dataset_name=dataset_name,
                                                          training_batch_size=training_batch_size,
                                                          dataset_loader_func_name=dataset_loader_func_name,
                                                          repeat_val_set=validation_steps is not None,
                                                          shuffle_training_set=shuffle_training_set)

    # <--Callbacks-->
    from summary import TensorBoardFix
    tensorboard_callback = TensorBoardFix(log_dir=log_dir, write_graph=False, write_images=False,
                                          update_freq=summary_update_freq, profile_batch=3)
    # Enable numerics check & performance profiler
    # tf.debugging.enable_check_numerics(True)
    # tf.summary.trace_on(graph=True, profiler=True)
    # tf.summary.trace_export(name="learned_optics_trace", step=0, profiler_outdir=log_dir)

    Logger.i("[ModelCheckPoint] save_freq epoch = ", step_per_epoch)

    checkpoint_dir_path = "./checkpoint/" + train_log_dir_name
    checkpoint_file_path = "./checkpoint/" + train_log_dir_name + "/cp-{epoch:03d}.ckpt"  # latest {epoch:03d}
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_file_path, verbose=1, save_best_only=False,
                                                     save_weights_only=True,
                                                     save_freq=step_per_epoch * controlled_training_args[
                                                         "save_freq_epoch"],
                                                     mode=checkpoint_save_mode, monitor=checkpoint_monitor)
    if not os.path.exists(checkpoint_dir_path):
        os.makedirs(checkpoint_dir_path)
        Logger.i("Creating dir:", checkpoint_dir_path)
    if controlled_model_args is not None and not continue_training:
        with open(checkpoint_dir_path + "/controlled_model_args.json", 'a+') as model_args_json:
            json.dump(controlled_model_args, model_args_json)
    if controlled_training_args is not None and not continue_training:
        del controlled_training_args["metrics"]
        with open(checkpoint_dir_path + "/controlled_training_args.json", 'a+') as training_args_json:
            json.dump(controlled_training_args, training_args_json)

    # early_stop_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor=checkpoint_monitor, min_delta=0, patience=5, verbose=0,
    #     mode=checkpoint_save_mode, baseline=None, restore_best_weights=True)
    # <!--Callbacks-->

    # Loss & metrics
    from loss import LOSS_FUNCTION_FILTER
    loss = LOSS_FUNCTION_FILTER[loss]

    # Create model from the arguments by default `LearnedDiffractiveOpticsModel` class.
    # If the model instance is already given, the given instance will be used.

    if model is None:
        model = model_factory(controlled_model_args, loss, metrics, training_batch_size)

    model.summary()
    print_args_summary(controlled_model_args, controlled_training_args)

    checkpoint_epoch = 0
    if pretrained_checkpoint_path_to_load is not None:
        Logger.i("Loading checkpoint file from: ", pretrained_checkpoint_path_to_load)
        checkpoint_epoch = int(pretrained_checkpoint_path_to_load[-8:-5])
        pretrained_checkpoint_to_load = tf.train.Checkpoint(model)
        pretrained_checkpoint_to_load.restore(pretrained_checkpoint_path_to_load)
        Logger.i("Restored checkpoint: ", pretrained_checkpoint_to_load)
        Logger.i("Start fitting from epoch ", checkpoint_epoch)

    if recompile:
        optimizer_args = controlled_model_args["default_optimizer_learning_rate_args"]
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.optimizers.schedules import ExponentialDecay
        model.compile(loss=loss, metrics=metrics, optimizer=Adam(learning_rate=ExponentialDecay(**optimizer_args)))

    model.fit(train_pairs, initial_epoch=checkpoint_epoch, epochs=total_epoch, validation_data=validation_pairs,
              validation_steps=validation_steps, verbose=1, steps_per_epoch=step_per_epoch,
              callbacks=[cp_callback, tensorboard_callback])
