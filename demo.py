from Lanedataloader import get_transforms
from swinmodel import SwinNet
from PIL import Image
import torch
import os

if "__main__" == __name__:
    model = SwinNet()
    model.to(device)
    model.eval()

    img_transforms = get_transforms()
    toPIL = transforms.ToPILImage()
    output_dir = "output/"
    if not os.path.isdir(output_dir):
        os.mkdir(outpupt_dir)
    
    input_fname = "ICME2022_Training_Dataset/images_real_world/"
    if not os.path.exists(input_fname):
        print("path not exist! Please check your input file path")

    if os.path.isdir(input_fname):
        fnames = sorted(os.listdir(input_fname))
        for fname in fnames:
            img = Image.open(input_fname + fname)
            img = img_transforms(img)
            img = img.unsqueeze(0)
            output = model(img)
            output = toPIL(output[0])
            output.save(output_dir + "pred_"+fname)
    else:
        img = Image.open(input_fname)
        img = img_transforms(img)
        img = img.unsqueeze(0)
        output = model(img)
        output = toPIL(output[0])
        output.save(output_dir + "pred_"+fname)
