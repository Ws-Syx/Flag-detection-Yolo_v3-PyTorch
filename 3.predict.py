from PIL import Image
from yolo import YOLO
import os

if __name__ == '__main__':
    yolo = YOLO()

    if True:
        filelist = os.listdir('./img')

        for filename in filelist:
            if filename.endswith('.jpg') or filename.endswith('.JPG'):
                print('\n' + filename)

                try:
                    image = Image.open('./img/' + filename)
                except:
                    print('Open Error! Try again!')
                    continue
                else:
                    r_image = yolo.detect_image(image)
                    r_image.show()
