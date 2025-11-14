# ZeroSign_TITS_Revision

Reference implementation of the ZeroSign model (research prototype).

Contents:
- models/: model components (Swin backbone wrapper, detection head, density head, GCN fusion, ZeroSign model)
- data/: dataset stub for quick testing
- utils/: visualization and metrics helpers
- train.py: training loop skeleton
- eval.py: evaluation skeleton
- requirements.txt: Python dependencies
- scripts/run_example.sh: example run script

Notes:
- Detection heads and losses are simplified placeholders to create a runnable prototype.
- Replace dataset stub with your own data loader and annotation parser.
- For production use, integrate a proper detector head (YOLO/FCOS) and accurate detection/loss computation.


Datasets:
This project uses publicly available crowd counting datasets.
Due to copyright restrictions, datasets are NOT uploaded in this repository.
Please download them from the official or Kaggle sources below.

1. ShanghaiTech Part A & B
   https://www.kaggle.com/datasets/tthien/shanghaitech
2. JHU-CROWD++
   https://www.kaggle.com/datasets/hoangxuanviet/jhu-crowd
3. UCF-QNRF
   https://www.kaggle.com/datasets/kokunsyu/ucf-qnrf-eccv18

Citation:
If you use this repository, please cite our paper:
"Multi-Scale Crowd Detection and Density Estimation Using Graph Neural Networks" (submitted to The Visual Computer, 2025).
