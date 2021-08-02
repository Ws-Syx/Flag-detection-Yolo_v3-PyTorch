# Flag-Detection 国旗检测

该项目基于PyTorch框架，在一定程度上使用了开源的yolo v3部分代码。(本人并未对原有的yolo v3部分的代码做过多的修饰与更改，原作者的注释得到了充分的保留。)


本项目提供了数据集和本人在Titan XP上训练完成的检测模型。

(当我回忆起我采用了哪位dalao的代码以后，我会在此注明引用来源，并表示感谢)

## 1 简介

### 1.1 支持的国旗类别
本项目提供了数据集，并提供了基于本数据集训练完成的模型。
该项目可以完成对64种国旗的分类，涵盖了世界上主要国家，同时根据基于字典序的国家列表从头选择了一部分国家。

（该项目具有一定的可扩展性，欢迎更多的开发者完善数据集）

以下是本项目中所涵盖的国家列表：

{'Afghanistan': 0, 'Albania': 1, 'Andorra': 2, 'Angola': 3,

'Argentina': 4, 'Armenia': 5, 'Australia': 6, 'Austria': 7, 

'Azerbaijan': 8, 'Bahamas': 9, 'Bahrain': 10, 'Bangladesh': 11, 

'Barbados': 12, 'Belarus': 13, 'Bhutan': 14, 'Bolivia': 15, 

'Brazil': 16, 'Bulgaria': 17, 'Cambodia': 18, 'Canada': 19, 

'CapeVerde': 20, 'Chile': 21, 'China': 22, 'Christmas Island': 23, 

'Colombia': 24, 'Congo': 25, 'Cook Island': 26, 'Costa rica': 27, 

'Cuba': 28, 'Cyprus': 29, 'Czech': 30, 'Democratic Republic of the Congo': 31, 

'France': 32, 'German': 33, 'India': 34, 'Iran': 35, 

'Iraq': 36, 'Israel': 37, 'Japan': 38, 'Jordan': 39, 

'Laos': 40, 'Lebanon': 41, 'Malaysia': 42, 'Myanmar': 43, 

'NewZealand': 44, 'NorthKoera': 45, 'Palestine': 46, 'Philippines': 47, 

'Russia': 48, 'Saudi Arabia': 49, 'Singapore': 50, 'Slovakia': 51, 

'SouthKoera': 52, 'Spain': 53, 'Sverige': 54, 'Switzerland': 55, 

'Syrian': 56, 'Tailand': 57, 'The Central African Republic': 58, 'Turkey': 59, 

'UK': 60, 'US': 61, 'Ukraine': 62, 'Vietnam': 63}

### 1.2 数据集

本项目使用的数据集是VOC格式的。数据集分为两部分，分别由两部分构成

(1)手工标注的来自网络的照片：

共2400张左右。这些照片涵盖了上文提到的国家中的18种国家~~(似乎是的QAQ记忆不深)~~。
在数据集中以[XXX.jpg]的文件名形式存在。

(2)根据国旗图像与随机的底版批量生成的：
共1240张左右。在数据集中以[qwq_XXX.jpg]的文件名形式存在。

数据集的压缩包已经上传到百度网盘当中了：

链接：https://pan.baidu.com/s/1_xyZJAbi1sLhT1bzW97dPg 

提取码：6xqg

## 2 如何启动

### 2.1 数据集的准备

数据集应当放置在'./data/dataset'路径下，每一个图片都应该对应一个文件名相同的VOC格式的xml标签文件。

### 2.2 数据集的处理

首先运行“1.pre_process.py”，完成对数据集的预处理，这一步操作的意义在于产生两个文件：

(1)将VOC格式的数据集中的信息提取出来，存放到'./data/dataset.txt'文本文档中，方便统计所有的数据样本，方便进行数据集的统计、打乱、划分。

(2)同时程序会将VOC格式的数据集中的所有出现过的Label进行编号，形成映射关系{编号：国家名}，存放到'./model/dict.npy'文件中。这样做是为了方便模型进行多分类，方便根据分类结果映射到实际的国家名。

该操作会自动进行数据集的完整性检查，忽略掉缺失了图片和或缺失了标签的文件。

### 2.3 训练

运行“2.train.py”，开始模型的训练。受限于本人的代码开发水平，本项目并不支持在未安装Cuda框架的环境中运行。（在运算过程中，多如牛毛的向量会被发送到显存中，而同时设计CUDA和非CUDA的代码会显著的增加代码量）。

- 请在utils\train_details.py 更改训练过程中的batch_size, learning_rate, epoch, 训练集与验证集的比例, 模型保存的名称
- 请在utils\config.py 更改类别数量、输入大小等其他模型数据

训练完成的模型将会存储在＂model/[你制定的保存名称].pth＂位置

PS：本人训练完成的模型已经上传到了百度网盘，也可以直接下载下来放到model文件夹

链接：https://pan.baidu.com/s/1QO80UkUktFRfS1pTJnAeDQ 

提取码：n9rx

### 2.4 使用训练完成的模型

要使用训练完成的模型进行预测，可以参考本人写的一个小demo “3.predict.py”，这个demo提供了一种实例：如何载入模型，如何打开一个图片并使模型完成预测。

- 请在yolo.py里更改要加载的[模型的名称]、[预测的序号与国家名之间的映射]
