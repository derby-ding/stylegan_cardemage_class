# stylegan_cardemage_class
环境：2个V100GPU，torch1.7以上，batchsize 128
训练时间：10000次，约30小时。

GAN训练及测试：
1. 数据准备：
[下载链接](https://drive.google.com/file/d/1pd4DhnCB659bsNsDcmMjCw18eYAajg3c/view?usp=sharing)，下载cls8gan128.zip

2. 8类stylegan训练：打开stylegan文件夹，运行下面命名
python -W ignore train.py --outdir=runs/car --data=../datasets/cls8gan128.zip --cond=1 --snap=500 --kimg=10000 --batch=128 --mirror=1 --augpipe=bgcfnc --cfg=cifar

3. 解耦disentangle stylegan：运行factor_gen.py，利用svd解耦stylegan模型，并沿解耦方向生成新样本。

4. 生成增强数据：
分别使用autogen.py，automixing.py生成随机增强和样式混合数据集

其他：
关于自建数据集：按照cls8gan128.zip的格式，数据文件夹包括图像文件夹和label的json文件，单图像文件夹包含不超过1万幅图片，超过1万则被差分为两个以上文件夹，文件名和样本标签信息存储在json文件中。注意不要使用其他格式，重写文件夹读取方式可能导致训练不收敛。

分类器训练及测试：
1. 新建datasets/cls8r128，将datasets中的train和valid文件拷入，然后将生成图像拷贝到datasets/cls8r128/train中，形成增强的图像集：

2. timm图像分类及预测：打开pytorch-image-models，运行训练命令：python train.py ../datasets/cls8r128 --model efficientnet_b2 -b 64 --epochs 200 --aa 'v0' --lr 0.005 --drop 0.5 --opt 'adam'
