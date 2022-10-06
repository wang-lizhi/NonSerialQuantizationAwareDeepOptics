# NSQDO

This repository provides the code for the non-serial quantization-aware deep optics model for snapshot hyperspectral imaging.

## Environment

Use Anaconda to create a virtual Python 3.8 environment with necessary dependencies from the **environment.yaml** file in the code.

```
conda env create -f ./environment.yaml
```

Then, activate the created environment and continue to train or test.

## Train

### Dataset Preparation

To train the model for hyperspectral imaging, at least one hyperspectral dataset should be downloaded to your computer in advance.
(e.g., [CAVE](https://www.cs.columbia.edu/CAVE/databases/multispectral/), [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/), or [Harvard](http://vision.seas.harvard.edu/hyperspec/index.html).)

Then, edit the ```DATASET_PATH``` dictionary in **util/data/dataset_loader.py** to indicate the name and path to your downloaded dataset. This is an example:
```
DATASET_PATH = {
  "dataset_name1": "/PATH1/TO/YOUR/DATSET1",
  "dataset_name2": "/PATH2/TO/YOUR/DATSET2",
  "dataset_name3": "/PATH2/TO/YOUR/DATSET3"
}
```
There should be three directories [train, validation, test] in your dataset directory to indicate which part should be used for training, validation, and testing, respectively.

### Configuration

After the dataset is prepared, configure the ```dataset_name``` and ```dataset_loader_func_name``` of ```controlled_training_args``` dictionary in **tasks/hyperspectral.py**.

The ```dataset_loader_func_name``` can be any function provided in **util/data/dataset_loader.py** or any function you implement using TensorFlow Dataset. 
(You should also put your customized dataset loader function in *util/data/dataset_loader.py* and set the ```dataset_loader_func_name``` to your customized function name. So that the trainer can automatically import and use it.)

The python file in ```tasks package``` could be duplicated and renamed to store different task configurations, including dataset, training options, loss, network arguments, etc.

Current **tasks/friendliness** has already provided configurations for training conventional, physics-friendly, decoder-friendly, and physics-and-decoder-friendly models using ICVL dataset.


### Start Training

After the configuration, the training can be started with the following commands:
```bash
python main.py --task=friendliness.physics-and-decoder-friendly --doe_material=SK1300 --quantization_level=4 --sensor_distance_mm 50 --scene_depth_m=1  --init_learning_rate=0.005 --tag=PDDO-4Lv-Training
```
Above example shows a 4-level physics-and-decoder-friendly model using SK1300 as the DOE material.

#### Arguments

 ```--task``` the name of the python file in the **tasks** package (without ".py").

 ```--doe_material``` the material refractive index used for DOE simulation. Supported options: SK1300, BK7, NOA61.

 ```--quantization_level``` the level count for the DOE quantization.

 ```--sensor_distance_mm``` the distance between the DOE to the sensor plane in millimeter.

 ```--scene_depth_m``` the distance between the scene to the DOE in meter.
 
 ```--init_learning_rate``` the initial learning rate for the optimizer.

 ```--tag``` a label that makes it easier to manage checkpoints and log files.

When the training starts, the trainer will save checkpoints and current task arguments into **./checkpoint/** as 2 JSON files named **controlled_model_args.json** and **controlled_training_args.json**. These files are important for the trainer to continue training and necessary for the evaluator to test the model. Visualization summary results, including DOE height maps, PSFs, and encoded images, will also be saved to the *./logs/* directory. Tensorboard can be used for viewing these results.

## Evaluation

After training, evaluation can be performed using the following commands:
```bash
# For physics-friendly models, using the test set
python evaluator.py --checkpoint_dir CHEKPOINT_DIR  \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS
                    
# For non-physics-friendly model, using the test set
python evaluator.py --checkpoint_dir CHEKPOINT_DIR \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS \
                    --test_q TEST_QUANTIZATION_LEVEL
                    
# For physics-friendly models, using the real RGB capture
python evaluator.py --checkpoint_dir CHEKPOINT_DIR \
                    --tag_name TAG_NAME \
                    --tag_vars TAG_VARS \
                    --real_data_dir REAL_DATA_DIR
```

#### Arguments

```--checkpoint_dir``` argument is the name of the sub directory in **./checkpoint/**.

```--tag_name``` argument is the inner tag name indicating the sub-directory in ```-checkpoint_dir``` given above.

```--tag_vars``` argument (optional) is the string value to insert in the %s placeholder of tag_name. Do not set this argument if there is no "%s" in your ```--tag_name``` argument.

```--test_q``` (optional) indicates the quantization level used during the test. Only the non-physics-friendly models needs this argument for test.

```--real_data_dir``` argument (optional) is the path to the directory storing real captured RGB images (PNG files).

 ```--real_data_height``` the height of the real capture.

 ```--real_data_width``` the width of the real capture.

 ```--with_mat``` save the hyperspectral reconstruction as a MAT file for each test case.

 ```--with_image``` save the RGB visualization as a image for each test case.

The evaluator will output test results into **./eval-res/**, including .csv files with test metrics, visualized RGB images, and corresponding hyperspectral .mat files.
