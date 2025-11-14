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
