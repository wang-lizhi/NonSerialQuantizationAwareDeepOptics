import argparse

from importlib import import_module

import tensorflow as tf

from log import Logger

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('--task', type=str, required=True,
                             help='Task name')

argument_parser.add_argument('--doe_material', type=str, required=False,
                             help='DOE material, support BK7 or NOA61, type in uppercase.')

argument_parser.add_argument('--quantization_level', type=int, required=False,
                             help="Quantization levels' count used in simulation.")

argument_parser.add_argument('--quantize_at_test_only', required=False, default=False, action="store_true",
                             help='If enabled, the full precision is used in training stage, '
                                  'and quantization will be applied in testing stage only.')

argument_parser.add_argument('--alpha_blending', required=False, default=False, action="store_true",
                             help='Whether to use alpha-blending for quantization-aware training. '
                                  'If not, the STE will be used for quantization-aware training.')

argument_parser.add_argument('--adaptive_quantization', required=False, default=False, action="store_true",
                             help='Whether to use adaptive quantization in alpha-blending quantization-aware training')

argument_parser.add_argument('--checkpoint', type=str, required=False,
                             help='Checkpoint file path.')

argument_parser.add_argument('--continue_training', required=False, default=False, action="store_true",
                             help='Whether to continue training and save checkpoints and log into the same directory.')

argument_parser.add_argument('--tag', type=str, required=False,
                             help='Tag name of task.')

argument_parser.add_argument('--sensor_distance_mm', type=int, required=False,
                             help='sensor_distance_mm.')

argument_parser.add_argument('--scene_depth_m', type=int, required=False,
                             help='scene_depth_m.')

argument_parser.add_argument('--init_learning_rate', type=float, required=False,
                             help='optimizer learning rate.')

arguments = argument_parser.parse_args()

if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices("GPU")
    logical_devices = tf.config.list_logical_devices("GPU")
    Logger.i("\n \n Physical GPU Devices(s): \n", physical_devices)
    Logger.i("\n \n Logical GPU Devices(s): \n", logical_devices)
    task_package_name = "tasks." + str(arguments.task)
    Logger.i("Task: ", task_package_name)
    train_func = getattr(import_module(task_package_name), "train")
    args_dict = vars(arguments)
    del args_dict["task"]
    Logger.i("Start TASK %s" % task_package_name)
    Logger.i("Extra Arguments: ", args_dict)
    train_func(**args_dict)
    Logger.i("Finished TASK %s" % task_package_name)
