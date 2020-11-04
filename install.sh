git clone https://github.com/zhouwy19/jrender.git
cp -r jrender/jrender ./
mkdir init_models
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl