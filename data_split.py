import os 
import shutil

train_data_dir = 'ICME2022_Training_Dataset/images/'
train_gt_dir = 'ICME2022_Training_Dataset/labels/class_labels/'
test_data_dir = "ICME2022_Training_Dataset/test_images/"
test_gt_dir = 'ICME2022_Training_Dataset/labels/class_test_labels/'

if not os.path.isdir(test_data_dir):
    os.mkdir(test_data_dir)
if not os.path.isdir(test_gt_dir):
    os.mkdir(test_gt_dir)

datas = sorted(os.listdir(train_data_dir))
gts = sorted(os.listdir(train_gt_dir))
test_num = 2500
datas = datas[:test_num]
gts = gts[:test_num]
print(len(datas))
print(len(gts))
for data, gt in zip(datas,gts):
    shutil.move(train_data_dir+data,test_data_dir+data)
    shutil.move(train_gt_dir+gt, test_gt_dir+gt)


datas = sorted(os.listdir(train_data_dir))
gts = sorted(os.listdir(train_gt_dir))
print(len(datas))
print(len(gts))

test_datas = sorted(os.listdir(test_data_dir))
test_gts = sorted(os.listdir(test_gt_dir))
print(len(test_datas))
print(len(test_gts))