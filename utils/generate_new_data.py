# 数据处理
import numpy as np
import random
# 图像处理
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import io
from skimage import transform, data
# xml文件制作
import xml.dom.minidom as Dom
import codecs

save_path = 'C:/Users/Ws_Syx/Desktop/new_dataset/new_'


def makeXML(count, label, box):
    # 制作xml
    doc = Dom.Document()
    node_annotation = doc.createElement('annotation')
    doc.appendChild(node_annotation)

    # filename
    node_filename = doc.createElement('filename')
    node_filename_value = doc.createTextNode('{}.jpg'.format(count))
    node_filename.appendChild(node_filename_value)
    node_annotation.appendChild(node_filename)

    # size
    node_size = doc.createElement('size')
    node_annotation.appendChild(node_size)
    # size.weight
    node_width = doc.createElement('width')
    node_width.appendChild(doc.createTextNode('{}'.format(back_w)))
    node_size.appendChild(node_width)
    # size.height
    node_height = doc.createElement('height')
    node_height.appendChild(doc.createTextNode('{}'.format(back_h)))
    node_size.appendChild(node_height)
    # size.depth
    node_depth = doc.createElement('depth')
    node_depth.appendChild(doc.createTextNode('3'))
    node_size.appendChild(node_depth)

    # object
    node_object = doc.createElement('object')
    node_annotation.appendChild(node_object)
    # object.name
    node_name = doc.createElement('name')
    node_name.appendChild(doc.createTextNode(label))
    node_object.appendChild(node_name)
    # object.pose
    node_pose = doc.createElement('pose')
    node_pose.appendChild(doc.createTextNode('Unspecified'))
    node_object.appendChild(node_pose)
    # object.bndbox
    node_bndbox = doc.createElement('bndbox')
    node_object.appendChild(node_bndbox)
    # object.bndbox.xmin
    node_bndbox_xmin = doc.createElement('xmin')
    node_bndbox_xmin.appendChild(doc.createTextNode('{}'.format(box[0])))
    node_bndbox.appendChild(node_bndbox_xmin)
    # object.bndbox.ymin
    node_bndbox_ymin = doc.createElement('ymin')
    node_bndbox_ymin.appendChild(doc.createTextNode('{}'.format(box[1])))
    node_bndbox.appendChild(node_bndbox_ymin)
    # object.bndbox.xmax
    node_bndbox_xmax = doc.createElement('xmax')
    node_bndbox_xmax.appendChild(doc.createTextNode('{}'.format(box[2])))
    node_bndbox.appendChild(node_bndbox_xmax)
    # object.bndbox.ymax
    node_bndbox_ymax = doc.createElement('ymax')
    node_bndbox_ymax.appendChild(doc.createTextNode('{}'.format(box[3])))
    node_bndbox.appendChild(node_bndbox_ymax)

    # save xml file
    f = codecs.open(save_path + '{}.xml'.format(count), 'w', 'utf-8')
    doc.writexml(f, indent='\t', addindent='\t', newl='\n', encoding='utf-8')


def get_flag_list(data_path):
    train_dataset = []
    test_dataset = []

    listdir = os.listdir(data_path)
    for class_name in listdir:
        class_path = data_path + '\\' + class_name
        if os.path.isdir(class_path):
            # 获取当前文件夹下所有的图片文件的绝对地址
            image_list = []
            for file_name in os.listdir(class_path):
                file_name = data_path + '\\' + class_name + '\\' + file_name
                if os.path.isfile(file_name):
                    if file_name[-4:] == '.jpg' or file_name[-4:] == '.JPG':
                        image_list.append(file_name)
                    elif file_name[-5:] == '.jpeg' or file_name[-5:] == '.JPEG':
                        image_list.append(file_name)
                    elif file_name[-4:] == '.png' or file_name[-4:] == '.PNG':
                        image_list.append(file_name)

            # 对当前目录下所有的图片文件切割成两部分
            length = len(image_list)
            index = np.arange(0, length)
            random.shuffle(index)

            train_index = index[:int(length * 0.9)]
            test_index = index[int(length * 0.9):]

            for i in train_index:
                train_dataset.append([image_list[i], class_name])
            for i in test_index:
                test_dataset.append([image_list[i], class_name])

    return train_dataset, test_dataset


def get_background_list(background_path):
    background_list = []
    listdir = os.listdir(background_path)
    for file_name in listdir:
        file_name = background_path + '\\' + file_name
        if os.path.isfile(file_name):
            if file_name[-4:] == '.jpg' or file_name[-4:] == '.JPG':
                background_list.append(file_name)
            elif file_name[-5:] == '.jpeg' or file_name[-5:] == '.JPEG':
                background_list.append(file_name)
            elif file_name[-4:] == '.png' or file_name[-4:] == '.PNG':
                background_list.append(file_name)

    return background_list


if __name__ == '__main__':
    data_path = 'C:/Users/Ws_Syx/Desktop/standard_flag'
    background_path = 'C:/Users/Ws_Syx/Desktop/背景图'

    train_dataset, test_dataset = get_flag_list(data_path)
    background_list = get_background_list(background_path)

    count = 0
    repeat = 10
    for qwq in range(repeat):

        for [flag_path, label] in train_dataset:
            # 计数
            count = count + 1
            # 随机数
            rand = random.randint(0, 100)

            # 随机确定背景图的尺寸
            # back_h, back_w = [random.randint(200, 400), random.randint(200, 400)]
            back_h, back_w = 300, 300
            # 随即确定前景图的尺寸
            flag_h = int(back_h * random.uniform(0.1, 0.99))
            flag_w = min(int(flag_h * random.uniform(0.8, 1.5)), back_w)

            if rand % 2 == 1:
                # 随机选取一张背景图
                index = random.randint(0, len(background_list) - 1)
                background_path = background_list[index]
                background = Image.open(background_path)
                background = background.resize((back_w, back_h))
            else:
                random_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                background = Image.new("RGB", (back_w, back_h), random_color)

            # 打开前景图
            flag = Image.open(flag_path)
            flag = flag.resize((flag_w, flag_h))
            # 粘贴
            x = random.randint(0, back_w - flag_w)
            y = random.randint(0, back_h - flag_h)
            box = (x, y, x + flag_w, y + flag_h)
            background.paste(flag, box)
            # 保存图片
            background.save(save_path + '{}.jpg'.format(count))

            # 制作xml
            makeXML(count, label, box)
