import os

train_data_path = 'ICME2022_Training_Dataset/images/'
train_gt_path = 'ICME2022_Training_Dataset/labels/class_labels/'
files = zip(sorted(os.listdir(train_data_path)),sorted(os.listdir(train_gt_path)))
idx = 1
# for fname in files:
#     os.rename(fname, "img_" + str(idx) + ".jpg")
for data, gt in files:
    os.rename(train_data_path + data, train_data_path + "img_" + str(idx) + ".jpg")
    os.rename(train_gt_path + gt, train_gt_path + "gt_" + str(idx) + ".jpg")
    idx += 1

files = sorted(os.listdir(train_data_path))
print(len(files))

files = sorted(os.listdir(train_gt_path))
print(len(files))
