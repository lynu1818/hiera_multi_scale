# Detectron2 Hiera Multiscale
## Goal
To support multiscale of input for Hiera

## TODO
1. Build Environment
    - Install conda in the lab server, since there's no sudo
    - [detectron2](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)
    - Install pypi packages
        - requirments.txt
        - sam2 need to be installed
        - other dependecy that I forgot to include, please test it.

2. Setup COCO dataset. [Link](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)
3. Run evaluation (refer projects/MViTv2/README.md)
4. Modify configs (projects/Hiera/configs/mask_rcnn_hiera_abs_win_t_512.py)
4. Modify codes at detectron2/modeling/backbone/hiera_abs_win.py

## Final
After everything is working fine, run evaluation on COCO dataset. The mAP should be larger than 40. Can run configs of MViTv2 as reference.