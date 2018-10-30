
import os
import shutil

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
LABEL_PATH = 'data/labels.txt'


def refine_data(dir, classes):
    print(f'start processing {dir}...')

    for c in classes:
        os.makedirs(os.path.join(dir, c), exist_ok=True)
    
    img_list = list(filter(lambda x: len(x.split('.')) > 1, os.listdir(dir)))

    for img in img_list:
        class_ = img.split('.')[0].split('_')[1]
        old_path = os.path.join(dir, img)
        new_path = os.path.join(dir, class_)
        shutil.move(old_path, new_path)

    print('OK')

if __name__ == '__main__':
    f = open(LABEL_PATH, 'r')
    classes_list = list(map(lambda x:x.strip(), f.readlines()))
    refine_data(TRAIN_DIR, classes_list)
    refine_data(TEST_DIR, classes_list)
    f.close()
