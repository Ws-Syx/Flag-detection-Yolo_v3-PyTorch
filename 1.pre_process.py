import xml.etree.ElementTree as ET
from os import getcwd
import os
import random
import numpy as np
from utils.dictionary import load_dict, save_dict


# 清楚无法正常配对的数据（jpg+xml必须一一配对）
def clear_data(path):
    filelist = os.listdir(path)
    for filename in filelist:
        if filename.endswith('.xml'):
            if not os.path.isfile(path + filename[:-4] + '.jpg'):
                os.remove(path + filename)
        elif filename.endswith('.jpg'):
            if not os.path.isfile(path + filename[:-4] + '.xml'):
                os.remove(path + filename)
        else:
            os.remove(path + filename)


def get_datalist(path):
    """
    :param path: 文件列表
    :return: 文件列表（不带后缀）
    """
    # 文件夹下的所有文件
    filelist = os.listdir(path)
    # 被选中的jpg+xml文件组合
    data_list = []

    for cur_file in filelist:
        if cur_file.endswith(".xml") and (cur_file[:-4] + '.jpg' in filelist):
            data_list.append(cur_file[:-4])

    return data_list


# 将文件夹中可用的训练数据进行划分
def split_data(path, train_rate=0.9):
    # 随机种子
    random.seed(0)

    # 文件夹下的所有文件
    datafilepath = path + '/dataset'
    datafilelist = os.listdir(datafilepath)

    # 被选中的jpg+xml文件组合
    data_list = []
    for cur_file in datafilelist:
        if cur_file.endswith(".xml") and (cur_file[:-4] + '.jpg' in datafilelist):
            data_list.append(cur_file[:-4])
    random.shuffle(data_list)

    # 在列表上划分训练集
    train_list = data_list[:int(len(data_list) * train_rate)]
    # np.savetxt(path + '/train_list.txt', train_list, encoding='utf-8', fmt='%s')

    # 在列表上划分测试集
    test_list = data_list[len(train_list):]
    # np.savetxt(path + '/test_list.txt', test_list, encoding='utf-8', fmt='%s')

    return train_list, test_list


# 生成多分类的字典，建立{label}->{index}的映射关系
def get_dictionary(path):
    # 储存所有出现过的标签（不重复）
    classes = []

    filelist = os.listdir(path)
    for filename in filelist:
        if filename.endswith('.xml') and os.path.isfile(path + filename[:-4] + '.jpg'):
            in_file = open(path + filename, encoding='utf-8')
            tree = ET.parse(in_file)
            root = tree.getroot()

            for obj in root.iter('object'):
                # 标签
                cur_class = obj.find('name').text
                if cur_class not in classes:
                    classes.append(cur_class)

    # 所有出现过的标签类别
    classes = sorted(classes)
    print('class_num = {}'.format(len(classes)))

    # 建立字典
    dictionary = {classes[i]: i for i in range(len(classes))}
    return dictionary


# 将jpg+xml转化为可供训练的格式
def convert_annotation(image_id, list_file, dictionary):
    # 打开当前图片所对应的xml标签文件
    in_file = open(dataset_path + '{}.xml'.format(image_id), encoding='utf-8')

    # 在xml标签中遍历所有object
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        # 标签
        class_name = obj.find('name').text
        class_index = dictionary[class_name]

        # bounding box的坐标
        xmlbox = obj.find('bndbox')
        xmin = int(xmlbox.find('xmin').text)
        ymin = int(xmlbox.find('ymin').text)
        xmax = int(xmlbox.find('xmax').text)
        ymax = int(xmlbox.find('ymax').text)
        b = (xmin, ymin, xmax, ymax)

        # 存在bounding box
        list_file.write(" {},{},{},{},{}".format(xmin, ymin, xmax, ymax, class_index))

    list_file.write('\n')

    in_file.close()


if __name__ == '__main__':
    dataset_path = './data/dataset/'

    # 1. 清楚缺失标签或缺失数据的文件
    clear_data(path='./data/dataset/')

    # 2. 获取xml+jpg对的列表
    image_list = get_datalist(path='./data/dataset/')
    print('size of dataset is ', len(image_list))

    # 3. 在类别上建立形如{class_name: class_index}的映射
    # 因为最后分类预测不能直接预测出[字符串]结果，只能预测出[数字]，因此要建立[标签]到[数字]的映射
    dictionary = get_dictionary(path='./data/dataset/')
    # 把映射关系以【字典】的方式存起来
    save_dict('./model/dict.npy', dictionary)
    print(dictionary)

    # 4. 整理成格式化的数据集
    list_file = open('./data/dataset.txt', 'w')
    for image_id in image_list:
        list_file.write(dataset_path + '{}.jpg'.format(image_id))
        convert_annotation(image_id, list_file, dictionary)
    list_file.close()
