echo "----------------------- downloading pretrained model on synthetic face dataset -----------------------"
wget https://cg.cs.tsinghua.edu.cn/jittor/assets/build/checkpoint030.pkl
mkdir -p results/synface
mv checkpoint030.pkl results/synface