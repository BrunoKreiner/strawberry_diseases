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



## Additional Evaluation Criteria

**MC organisation in Sprint format**

Mini-challenge is organized and structured in 5 phases with weekly goals to be achieved, including weekly review & retrospective, next steps & risks/improvements.

**Scale**

Note 6. Scope defined for each week till mini-challenge submission with clear goals/targets to be achieved. All 5 weekly goals achieved on time. Weekly review & retrospective was conducted, documented and shared with Susanne. 

Note 5. Scope discussed and planned each week. 1 out of 5 weekly goals missed. Weekly review per week conducted, documented and shared every week.

Note 4. High level weekly planning in place. 60% of weekly goals were achieved on time i.e. 2 out of 5 weekly goals missed. Weekly review conducted, documented and shared with final submission.

Note 3. High level weekly planning in place. 4 out of 5 weekly goals missed. No weekly tracking & progress documented.

Note 2. High level weekly planning in place. All sprint goals missed. No weekly tracking

Note 1. No high level weekly planned.

## Proposed DLBS mini-challenge backlog

### Week 1 goals

Timeframe: 18.04. – 25.04.2023

- Deep-dive 3 participation
- Step 1: Data inspection
- Setup GitHub
- Milestone 1 preparation: Pitch on 25.04 
- Weekly review & retrospective conducted & documented

Definition of Done:
- First hand Data inspection completed and notebook uploaded in GitHub with Tag/comment in 'commit'
- Tag: 'Week 1 DoD'

### Week 2 goals

Timeframe: 25.04. – 02.05.2023

- Step 2: Training skeleton with Baseline model in place 
- Weekly review & retrospective

Definition of Done:
- Skeleton baseline model uploaded on GitHub with tag/comment in 'commit'
- Weekly retrospective available and uploaded on the GitHub with tag/commit in 'commit'
- Tag: 'Week 2 DoD'

### Week 3 goals

Timeframe: 02.05. – 09.05.2023

- Step 3: Overfit and regularize per person
- Weekly review & retrospective conducted & documented

Definition of Done:
- Each team member has model variation with overfit & regularization ready & uploaded on GitHub with tag/comment in 'commit'
- Weekly retrospective available and uploaded on the GitHub with tag/commit in 'commit'
- Tag: 'Week 3 DoD'

### Week 4 goals

Timeframe: 08.05 – 12.05.2023

- step 4: Model tuning 
- Weekly review & retrospective conducted & documented

Definition of Done:
- Finding / tuned model per team member is uploaded on GitHub with tag/comment in 'commit'
- Weekly retrospective available and uploaded on the GitHub with tag/commit in 'commit'
- Tag: 'Week 4 DoD'

### Week 5 goals

Timeframe: 09.05. – 19.05.2023

- Documentation completed (Document paper on OverLeaf)
- Milestone 2: Demo on 16.05.2023
- Weekly review & retrospective conducted & documented
- Bonus task per person completed
- Milestone 3: Final submission on 19.05.2023 in GitHub

Definition of Done:
- Final submission & Code including Documentation uploaded on GitHub Repo with tag/comment in 'commit'
- Tag: 'Week 5 DoD'








