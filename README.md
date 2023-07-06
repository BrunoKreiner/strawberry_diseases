# DLBS mini-challenge in FS23

## Abstract

This mini-challenge focuses on the task of detecting and segmenting seven different kinds of
strawberry diseases found on leaves, fruit, and flowers using instance segmentation. A Mask R-
CNN model, as referenced in a [previous paper](https://www.mdpi.com/1424-8220/21/19/6565), served as a benchmark but could not
be reproduced due to package dependency problems. Without any modifications, a YOLOv8
segmentation model immediately surpassed the baseline Mask R-CNN model by around 10% in
terms of mAP. Marginal improvements were seen in the YOLOv8 model through fine-tuning,
changing the learning rate, adjusting the optimizer, and selecting the largest segmentation
model available on ultralytics.com. However, challenges persist, such as ambiguity in anno-
tating overlapping instances of disease clusters and potential overfitting. Despite these, the
evaluation in the notebooks showcases opportunities for further optimization of the YOLOv8
model.

## Data
The strawberry disease dataset is a high quality dataset where most images are close ups of individual leaves and fruits. It is available at [Kaggle](https://www.kaggle.com/datasets/usmanafzaal/strawberry-disease-detection-dataset). It consists of 2500 images in total with corresponding segmentation annotation files (one per image) for seven types of diseases found in Strawberry plant. The diseases can be seen in figure below. The data was collected from multiple greenhouses under natural illumination conditions in South Korea and the diseases were verified by experts. The images were processed to 419 x 419 resolution. The dataset is split into 1450, 307 and 743 images for training,  validation and test sets. For image there exists an individual json file with a list of ground-truth annotations. Each annotation has the label name as a string and a list of points with x and y coordinates that span the segmentation mask polygon. For model training, the data was uploaded to Roboflow: (LINK)[https://universe.roboflow.com/dlbs-zcxsj/strawberry-hxgaj/model/2]

![Strawberry-disease-image](https://github.com/BrunoKreiner/strawberry_diseases/assets/19567972/d0ddc2a2-37ae-43fb-81e0-14a9f6a5526e)


## Results
In this project, various configurations of the YOLOv8 model were evaluated for an instance segmentation task. For chapter 3, a Mask R-CNN skeleton was trained. As already discussed, the Mask R-CNN model didn't learn and the results of the baseline Mask R-CNN model from existing research couldn't be reproduced. Using YOLOv8 as a backup, a 10% mAP improvement could be achieved over the baseline from the existing paper. Different kinds of YOLOv8 models were trained over 100 epochs. These models show a steady decrease in training loss for box prediction, segmentation and class prediction. The base model already performs very well at around 92-93% mAP50. Additional data augmentation techniques can potentially decrease performance due to YOLOv8's inbuilt data augmentation. The "Base XL" performed the best on the validation data. It was trained using YOLOv8's XL model. All other models, which come very close to it, were trained using YOLOv8's small model. A test run with a smaller learning rate (factor of 10) and full augmentation strategy based on the augmentation steps from the existing research follows the "Base XL" closely while some other models actually worsen with data augmentation. This can be seen in the figure below. Using the same data augmentation, using the AdamW optimizer or setting the pretrained parameter in YOLOv8 to true also improved results. The existing paper uses "Edge Detect" and "Color Enhancement" for training their best Mask R-CNN model. The direct implementation of those augmentations were not found in common augmentation libraries such as PyTorch or Albumentations. Therefore, "Edge Detect" was left out and "Color Enhancement" was replaced with "RGB Shift". Interestingly, smaller learning rates led to faster learning, contradicting expectations. Just changing hyperparameters or using the base model with data augmentation didn't lead to improved training results. The use of dropout (regularization) improved the validation mAPs by almost 1% in comparison to its baseline. Despite validation loss not being tracked correctly by YOLOv8, results indicate possible overfitting since dropout clearly improved results. Furthermore, a spike in training loss at the 90th epoch was observed, without any discernible impact on the validation metrics. The confusion matrices revealed that most false positives or negatives were for the background class, which is also shown in the existing research for the Mask R-CNN. Manual evaluations showed accurate box and class predictions, with some limitatrions in detailed mask placement. It occasionally struggled to create detailed and accurate masks, especially for diseases appearing in clusters. This might be due to a misplacement of bounding boxes in the data set by researchers using the LabelMe website since for one cluster, multiple boxes can be placed on individual disease spots. In summary, this project demonstrated the robust capabilities of YOLOv8 in instance segmentation tasks, with potential further improvements through careful manipulation of learning rates, dropout and other hyperparameters. It highlights the importance of understanding in-built data augmentation functions and their impact on model performance. A more detailed description is in the notebook "yolov8\_evaluation.ipynb".

![barplot_yolov8_models](https://github.com/BrunoKreiner/strawberry_diseases/assets/19567972/d0845db9-d462-46c2-a82b-44d039e04749)

## Setup

To setup YOLOv8 training, follow the notebook [YOLOv8 Base Notebook](./notebooks/yolov8_base.ipynb) and install ultralytics, roboflow and pycocotools in a conda environment by first running:
```
conda create -n [env_name]
conda activate [env_name]
```
And then running whatever code IDE with the conda environment as the Python interpreter.
To download the dataset, you should use your own Roboflow API key by creating an account on their website.

## Project Structure

This project has the following key directories:

- `./media` - This directory contains the report and weekly reviews. It also contains saved plots from the analysis.

- `./models` - This is where the base models for the project are stored.

- `./notebooks` - This directory contains notebooks for training and evaluation. Detailed descriptions of the notebooks are as follows:
    - [EDA](./notebooks/EDA.ipynb) - This notebook contains an exploratory analysis of the data.
    - [EVA Model Testing](./notebooks/eva-1.ipynb) - This notebook demonstrates the error encountered when trying to install EVA.
    - [Mask R-CNN Evaluation](./notebooks/maskrcnn_evaluation.ipynb) - This notebook includes efforts to make the Mask R-CNN PyTorch model work with a small training set, and includes visualizations of input data and printouts of target and prediction dictionaries from the model's input/output.
    - [Mask R-CNN Training](./notebooks/maskrcnn_train.ipynb) - This is the notebook where the Mask R-CNN was trained with 50 epochs.
    - [Mask R-CNN TensorFlow Testing](./notebooks/maskrcnn_tensorflow_test.ipynb) - This notebook is used for testing a TensorFlow implementation of Mask R-CNN (this did not work because it is based on the older Mask R-CNN GitHub repository).
    - [YOLOv8 Evaluation](./notebooks/yolov8_evaluation.ipynb) - This notebook contains evaluations of all the YOLOv8 models trained.
    - [YOLOv8 Overfit](./notebooks/yolov8_overfit.ipynb) - This is the training notebook for all YOLOv8 models.
    - [YOLOv8 Base](./notebooks/yolov8_base.ipynb) - This basic notebook shows the training of YOLOv8 and possible evaluations.

- `./runs` - This directory contains models trained by YOLO.

- `./src` - This directory contains utility scripts used across the project.

- `./strawberry` - This directory contains the base strawberry dataset.

- `./strawberry1` - This directory contains the strawberry dataset by Roboflow and augmented copies.








