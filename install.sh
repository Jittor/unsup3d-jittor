git clone https://github.com/Jittor/jrender.git
mv jrender jrender-bak
mv jrender-bak/jrender .
mkdir init_models
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl