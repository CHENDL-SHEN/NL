
# Nonlocal Loss for Weakly Supervised Semantic Segmentation 

This repository is the official implementation of "Nonlocal Loss for Weakly Supervised Semantic Segmentation". 

## Prerequisite
- Python 3.6, PyTorch 1.8.0, and more in requirements.txt
- CUDA 11.1
- 1 x  RTX 3090 GPUs

## Usage

### 1. Install python dependencies
```bash
python3 -m pip install -r requirements.txt
```
### 2. Train the Segmentation Network

- Train the segmentation model using pseudo labels and the proposed loss functions.
    ```python
    python NL_train_seg.py
### 3. Generate Segmentation Results
- After training, run inference to generate segmentation predictions.
    ```python
    python NL_infer_seg.py
## Acknowledgement
Our implementation is built upon the excellent work [PuzzleCAM](https://github.com/OFRIN/PuzzleCAM). If you find this repository useful, please also consider citing and visiting their project. We thank the authors for making their code publicly available.

