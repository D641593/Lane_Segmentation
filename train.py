import os
import math
import torch
# import logging
# from Unetmodel import Unet
from upernet import UperNet
from Lanedataloader import LaneDataset
from torch.utils.data import DataLoader

def validation(model, test_loader, loss_fn, min_loss, test_step_num_each_batch, save_flag):
    model.eval()
    with torch.no_grad():
        current_loss = 0
        for img,gt in test_loader:
            img = img.to(device)
            img = img.unsqueeze(0)
            gt = gt.to(device)
            gt = gt.unsqueeze(0)
            gt = gt.long()
            output = model(img)
            loss = loss_fn(output,gt)
            current_loss += loss.detach().cpu().item()
        current_loss /= test_step_num_each_batch
        print(current_loss)
        # logging.info("epoch: %d valid loss: %3.7f"%(epoch+1,current_loss))
        if current_loss < min_loss:
            save_flag = True
            # logging.info("save")
            min_loss = current_loss
    model.train()
    return save_flag, min_loss

if '__main__' == __name__:

    # logging_fname = "train_log.log"
    # log_format = '%(asctime)s %(levelname)s %(message)s'
    # logging.basicConfig(level=logging.INFO,filename=logging_fname,filemode='w',format=log_format,force = True)

    deviceType = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(deviceType)
    # logging.info('train using %s'%deviceType)
    print('train using ',deviceType)
    
    # parameter
    batch_size = 2
    epoches = 300
    learning_rate = 0.0001
    model_save_dir = "weights/"
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_name = "Unet_Lane"  # model parameter will save like (model_save_name)_(epoch).pth Ex. hello_50.pth
    
    save_each_epoch = 10
    val_each_epoch = 5

    # train_data_dir = 'ICME2022_Training_Dataset/test_few/'
    # train_gt_dir = 'ICME2022_Training_Dataset/test_few_gt/'
    # test_data_dir = "ICME2022_Training_Dataset/test_few/"
    # test_gt_dir = 'ICME2022_Training_Dataset/test_few_gt/'
    train_data_dir = 'ICME2022_Training_Dataset/images/'
    train_gt_dir = 'ICME2022_Training_Dataset/labels/class_labels/'
    test_data_dir = "ICME2022_Training_Dataset/test_images/"
    test_gt_dir = 'ICME2022_Training_Dataset/labels/class_test_labels/'

    dataset = LaneDataset(train_data_dir,train_gt_dir)
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)
    test_dataset = LaneDataset(test_data_dir,test_gt_dir)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 4, pin_memory = True)

    step_num_each_batch = math.ceil(dataset.__len__() / batch_size)
    test_step_num_each_batch = math.ceil(test_dataset.__len__() / batch_size)

    model = UperNet()
    model.to(device)
    model.train()

    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 0)

    save_flag = False
    min_loss = 10
    for epoch in range(epoches):
        iteration = 1
        for img, gt in loader:
            img = img.to(device)
            gt = gt.to(device)
            gt = gt.long()
            output = model(img)
            loss = loss_fn(output,gt)

            loss.backward()
            optimizer.step()
            current_loss = loss.detach().cpu().item() # only for show loss 
            print('epoch: {%d/%d}, step: {%d/%d}, loss : {%3.7f}, lr : {%3.7f}'%((epoch+1),epoches,iteration,step_num_each_batch, current_loss, optimizer.param_groups[0]['lr']))
            # logging.info('epoch: {%d/%d}, step: {%d/%d}, loss : {%3.7f}, lr : {%3.7f}'%((epoch+1),epoches,iteration,step_num_each_batch, current_loss, optimizer.param_groups[0]['lr']))
            iteration += 1
        lr_scheduler.step()

        if (epoch+1) % val_each_epoch == 0:
            save_flag, min_loss = validation(model,test_dataset,loss_fn, min_loss, test_step_num_each_batch,save_flag)
        # print(save_flag, min_loss)
        if save_flag or (epoch+1) % save_each_epoch == 0:
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_loss,
                'train_loss': save_flag
                }, model_save_dir + model_save_name + "_" + str(epoch+1) + '.pth')
            save_flag = False
            # logging.info("epoch %d save. save at %s"%(epoch+1, model_save_dir + model_save_name + "_" + str(epoch+1) + '.pth'))

    torch.save(model.state_dict(), model_save_dir + "final.pth")  
    print("That's it! training finish. ")