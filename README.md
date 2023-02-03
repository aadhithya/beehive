# beehive
Counting bees using AI.

## 1. Task
We want to count the number of bees flying around in order to monitor the hive. You are given a sample set of images
of bees, along with ground truth labels containing dots at the centroids of each bee in the image. The goal of this
challenge is to automate the process of counting bees in a given image.

## 2. Approach
- I start by treating this as an object detection problem, where the task is to detect the centroid-keypoint of the detected obhject - in this case, bees.
- To this end, we can use a point-based object dcetection model for this task. I use the CenterNet proposed in the paper [Objects as Points](https://arxiv.org/pdf/1904.07850.pdf) by Zhou et. al. for the task.
- The CenterNet is designed to solve exactly the problem at hand - detect objects with centroid points.
- The CenterNet has a keypoint detection loss, an offset loss and a task specific loss (bounding box regression, etc.). In this implementation, I use only the keypoint detection and offset loss, as we don't care about the bounding boxes of the detection.

### 2.1. Network Architecture and Objective Functions
Here, I talk about the network archtecture and the reasons behind selecting the arch. I also discuss the objective functions and how this implementation differs from the original paper.

### 2.2 Results
results here
#### 2.2.1 Training Results
Training results and loss curves.
#### 2.2.2 Evaluation Results
Evaluation Results here.





## 3. Stack
- **Deep learning Framework:**  pyTorch
- **Training Framework:** pyTorch lightning
- **Experiment Logging:** tensorboard
- **CLI App Framework:** typer

## 4. Usage

### 4.1 Install locally
- The easiest way to get started is to install the project locally. This should take care of the dependencies as well: `pip install -e .`
- Once installed run the tool using: `beehive count /path/to/image.jpg`
- For more options run: `beehive count --help`

### 4.2 Install requirements and run the script:
- Install requirements: `pip install -r requirements.txt`.
- run script: `python count.py /path/to/image.jpg`.

### Paper Implementation details

#### Models
- Standard ResNet with 3 up-conv blocks.
- Use bilinear interpolation for upConv [256, 128, 64] channels.
- One 3x3 deformable convolution bvefore upscaling.

#### Training
- 512x512 images as input --> 128x128 downscaled output resolution.
- **Augmentations**: Random flip, crop, scaling (0.6 - 1.3), color jittering
- batch size: 128
- lr: 5e-4
- 140 epochs.
