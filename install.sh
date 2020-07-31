git clone https://github.com/zhouwy19/N3MR_jittor.git
cp -r N3MR_jittor/neural_renderer ./
mkdir init_models
wget https://cloud.tsinghua.edu.cn/seafhttp/files/80cb8a82-7062-438a-8b94-198bb78b342d/vgg_pretrained_features.pkl
mv vgg_pretrained_features.pkl init_models/vgg_pretrained_features.pkl