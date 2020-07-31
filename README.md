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
git clone https://github.com/zhouwy19/N3MR_jittor.git
cp -r N3MR_jittor/neural_renderer ./
mkdir init_models
wget https://cloud.tsinghua.edu.cn/seafhttp/files/80cb8a82-7062-438a-8b94-198bb78b342d/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl
python3.7 run.py --config experiments/train_cat.yml
```

## Citation
```
@InProceedings{Wu_2020_CVPR,
  author = {Shangzhe Wu and Christian Rupprecht and Andrea Vedaldi},
  title = {Unsupervised Learning of Probably Symmetric Deformable 3D Objects from Images in the Wild},
  booktitle = {CVPR},
  year = {2020}
}
```