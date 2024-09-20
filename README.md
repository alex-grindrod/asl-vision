# asl-vision

## Setup
1. Obtain [WLASL2000 dataset of videos](https://www.kaggle.com/datasets/utsavk02/wlasl-complete)
2. Obtain [WLASL2000 script files](https://github.com/dxli94/WLASL/tree/master)
3. Modify utilities.py to select model and other data/model preferences
4. Run `python3 src/annotation/process_and_split.py`
5. Run `python3 src/annotation/generate_annotations.py`
    - Use `-c` argument to clear annotations
6. Run `python3 training/train.py`
    - Use `-e` for max epochs
    - Use `-dw` to disable using `wandb`
    - Use `-lr` to set learning rate


## TODO

- [ ] Try splitting classes with multiple sign variants
- [ ] Hyperparameter sweeps for each architecture
- [ ] Incorporate top-k accuracy in training


## Works Cited
1. WLASL 2000 Dataset
```
@inproceedings{li2020transferring,
 title={Transferring cross-domain knowledge for video sign language recognition},
 author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
 pages={6205--6214},
 year={2020}
}
```

2. [STGCN Simple Implementation](https://github.com/FelixOpolka/STGCN-PyTorch)

3. STGCN Original Implementation
```
@misc{mmskeleton2019,
  author =       {Sijie Yan, Yuanjun Xiong, Jingbo Wang, Dahua Lin},
  title =        {MMSkeleton},
  howpublished = {\url{https://github.com/open-mmlab/mmskeleton}},
  year =         {2019}
}
```