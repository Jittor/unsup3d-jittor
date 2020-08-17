git clone https://github.com/zhouwy19/N3MR_jittor.git
cp -r N3MR_jittor/neural_renderer ./
mkdir init_models
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl