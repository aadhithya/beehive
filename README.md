# beehive: counting bees using AI.

We want to count the number of bees flying around in order to monitor the hive. You are given a sample set of images
of bees, along with ground truth labels containing dots at the centroids of each bee in the image. The goal of this
challenge is to automate the process of counting bees in a given image.

## Solution Documentation
Check [report.md](./report.md)
## Get Started

- clone project: `git clone https://github.com/aadhithya/beehive.git`

### Inference
- `bee_counter.py` is a minimal inference script that runs inference using onnxruntime.
- Insatll requirements: `pip install -r inference_requirements.txt`
- run inference: `python bee_counter.py <image-path> --ckpt_path <ckpt-path> --show <bool>`
- *NOTE:* The checkpoint is downloaded from github if not available locally. Check [Releases](https://github.com/aadhithya/beehive/releases/tag/weights) for checkpoints.

### Training and Development
In case you want to develop or train/evaluate/infer model you need to do the following:
- install poetry: `pip install poetry`
- install requirements: `poetry install`

    **Note:** poetry install creates a new virtual env.
- now that the environment is created, you are ready to go.
- check installation: `python -m beehive --help`

**NOTE:** using this method needs **python >= 3.9**

#### Train Model
`python -m beehive train --help`

#### Evaluate Model
`python -m beehive eval --help`

#### pytorch Inference
`python -m beehive infer path/to/image --show`

#### ONNX Export
`python -m beehive export-onnx --help`
