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