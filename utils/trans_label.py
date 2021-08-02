import xml.etree.ElementTree as ET
from os import getcwd
import os

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
#
# classes = ["flag"]
#
# classes = ["china", "ccp", 'pla', "us", "uk",
#            "france", "japan", "northkoera", "southkoera", "russia",
#            "spain", "olympic", "union", "europe", "philippine",
#            "india", "baxi", "vietnam", "liao", "jianpuzhai",
#            "miandian", "tai", "malaixiya", "xinjiapo", "afuhan",
#            "yilake", "yilang", "xuliya", "yuedan", "libanen",
#            "yiselie", "balesitan", "shatealabo", "ruidian", "aodaliya",
#            "jianada", "baieluosi", "beiyue", "dongnanya", "shimaozuzhi"]
#
# classes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
#            '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
#            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
#            '31', '32', '33', '34', '35', '36', '37', '38', '39', '40']

classes_index = ['1', '4', '5',
                 '6', '7', '8', '9', '10',
                 '11', '15',
                 '16', '17', '18', '19', '20',
                 '21', '22', '23', '24', '25',
                 '26', '27', '28', '29', '30',
                 '31', '32', '33', '34', '35',
                 '36', '37']

classes_name = {'China', 'US', 'UK',
                'France', 'Japan', 'NorthKoera', 'SouthKoera', 'Russia',
                'Spain', 'Philippines',
                'India', 'Brazil', 'Vietnam', 'Laos', 'Cambodia',
                'Myanmar', 'Tailand', 'Malaysia', 'Singapore', 'Afghanistan',
                'Iraq', 'Iran', 'Syrian', 'Jordan', 'Lebanon',
                'Israel', 'Palestine', 'Saudi Arabia', 'Sverige', 'Australia',
                'Canada', 'Belarus'}

index_to_name = {'1': 'China', '4': 'US', '5': 'UK',
                 '6': 'France', '7': 'Japan', '8': 'NorthKoera', '9': 'SouthKoera', '10': 'Russia',
                 '11': 'Spain', '15': 'Philippines',
                 '16': 'India', '17': 'Brazil', '18': 'Vietnam', '19': 'Laos', '20': 'Cambodia',
                 '21': 'Myanmar', '22': 'Tailand', '23': 'Malaysia', '24': 'Singapore', '25': 'Afghanistan',
                 '26': 'Iraq', '27': 'Iran', '28': 'Syrian', '29': 'Jordan', '30': 'Lebanon',
                 '31': 'Israel', '32': 'Palestine', '33': 'Saudi Arabia', '34': 'Sverige', '35': 'Australia',
                 '36': 'Canada', '37': 'Belarus'}

img_path = '../data/dataset/'


def trans_label(filename):
    # print('filename = {}'.format(filename))
    in_file = open(filename, encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        # 每一个object
        cur_class = obj.find('name').text
        if(cur_class == '13'):
            print('sb')
        if cur_class not in classes_index and cur_class not in classes_name:
            root.remove(obj)
            continue
        else:
            if cur_class in classes_index:
                obj.find('name').text = index_to_name[cur_class]

    in_file.close()
    tree = ET.ElementTree(root)  # root为修改后的root
    tree.write(filename)


if __name__ == '__main__':
    # split_data(path='./data', train_rate=0.9)
    wd = getcwd()

    path = '../data/dataset/'
    filelist = os.listdir(path)
    for filename in filelist:
        if filename.endswith('.xml') and os.path.isfile(path + filename[:-4] + '.jpg'):
            trans_label(path + filename)
