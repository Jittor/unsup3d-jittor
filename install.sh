git clone https://github.com/Jittor/n3mr-jittor
cp -r n3mr-jittor/neural_renderer ./
mkdir init_models
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl
