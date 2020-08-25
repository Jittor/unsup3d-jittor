Jittor Implementation for the pepar Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild (CVPR 2020 oral).

## Datasets
1. [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) face dataset. Please download the original images (`img_celeba.7z`) from their [website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and run `celeba_crop.py` in `data/` to crop the images.
2. Synthetic face dataset generated using [Basel Face Model](https://faces.dmi.unibas.ch/bfm/). This can be downloaded using the script `download_synface.sh` provided in `data/`.
3. Cat face dataset composed of [Cat Head Dataset](http://academictorrents.com/details/c501571c29d16d7f41d159d699d0e7fb37092cbd) and [Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/) ([license](https://creativecommons.org/licenses/by-sa/4.0/)). This can be downloaded using the script `download_cat.sh` provided in `data/`.

Please remember to cite the corresponding papers if you use these datasets.

## Training
Check the configuration files in `experiments/` and run experiments, eg:

```python
git clone https://github.com/zhouwy19/unsup3d_jittor
cd unsup3d_jittor
bash install.sh
python3.7 run.py --config experiments/train_synface.yml
```

## Testing
Check the configuration files in `experiments/` and run experiments, eg:

```python
python3.7 run.py --config experiments/test_synface.yml
```

## Pretrained model

Here we provide our pretrained model trained using the default config experiments/train_synface.yml. You can run the following scripts to test Table 2 in the paper.

```python
bash pretrained/download_pretrained_synface.sh
python3.7 run.py --config experiments/test_synface.yml
```

The following is SIDE and MAD compared with original paper (Table 2).

|     | SIDE(×10−2) ↓  | MAD (deg.) ↓  |
|  ----  | ----  | ----  |
| Jittor  | 0.769±0.136 | 15.99±1.49 |
| Original paper  | 0.793±0.140 | 16.51±1.56 |

## Citation
```
@InProceedings{Wu_2020_CVPR,
  author = {Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title = {Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild},
  booktitle = {CVPR},
  year = {2020}
}
```