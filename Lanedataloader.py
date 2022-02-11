from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import torch
import cv2
import os

output_h, output_w = 512,512
class LaneDataset(Dataset):
    def __init__(self, img_dir, gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.img_fnames = sorted(os.listdir(img_dir))
        self.gt_fnames = sorted(os.listdir(gt_dir))

    def __len__(self):
        return len(self.img_fnames)

    def __getitem__(self,idx):
        img = Image.open(os.path.join(self.img_dir, self.img_fnames[idx]))
        gt_src = cv2.imread(os.path.join(self.gt_dir,self.gt_fnames[idx]))
        h,w = gt_src.shape[:2]
        gt = np.zeros((6,output_h,output_w),np.float32)
        for i in range(output_h):
            for j in range(output_w):
                gt[gt_src[round(i / output_h * h),round(j / output_w * w),0],i,j] = 1

        img_transforms = get_transforms()
        img = img_transforms(img)
        gt = torch.from_numpy(gt)
        img,gt = get_data_augmentation(img,gt)
        return img, gt

def get_data_augmentation(img, gt):
    hflip_p = random.randint(0,1)
    if hflip_p:
        img = transforms.functional.hflip(img)
        gt = transforms.functional.hflip(gt)
    return img, gt

def get_transforms():
    return transforms.Compose([
        transforms.Resize((output_h,output_w)),
        transforms.ToTensor()
    ])

if "__main__" == __name__:
    train_data_dir = 'ICME2022_Training_Dataset/images/'
    train_gt_dir = 'ICME2022_Training_Dataset/labels/class_labels/'
    data_loader = LaneDataset(train_data_dir, train_gt_dir)
    img, gt = data_loader.__getitem__(0)
    toPIL = transforms.ToPILImage()
    img = toPIL(img)
    img.save("sample_img.jpg")
    for i in range(6):
        tmp = gt[i].numpy()*255
        cv2.imwrite("sample_gt_"+str(i)+".png",tmp)

