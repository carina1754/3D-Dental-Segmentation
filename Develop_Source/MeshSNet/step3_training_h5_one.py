import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
import torch.nn as nn
from Mesh_dataset_h5 import *
from meshsegnet import *
from losses_and_metrics_for_mesh import *
import utils
import pandas as pd
from tqdm import tqdm
if __name__ == '__main__':

    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    use_visdom = False # if you don't use visdom, please set to False

    train_list = 'tensor_list/train_list_1_h5_one.csv' # use 1-fold as example
    val_list = 'tensor_list/val_list_1_h5_one.csv' # use 1-fold as example

    model_path = './models/'
    model_name = 'MeshSNet_N' # need to define
    checkpoint_name = 'latest_checkpoint.tar'

    num_classes = 17
    num_channels = 15 #number of features
    num_epochs = 200
    num_workers = 0
    train_batch_size = 15
    val_batch_size = 15
    
    if use_visdom:
        # set plotter
        global plotter
        plotter = utils.VisdomLinePlotter(env_name=model_name)

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # set dataset
    training_dataset = Mesh_Dataset(data_list_path=train_list)
    val_dataset = Mesh_Dataset(data_list_path=val_list)

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("쿠다 가능 :{}".format(torch.cuda.is_available()))
    print("현재 디바이스 :{}".format(torch.cuda.current_device()))
    print("디바이스 갯수 :{}".format(torch.cuda.device_count()))
    
    for idx in range(0, torch.cuda.device_count()):
        print("디바이스 이름 :{}".format(torch.cuda.get_device_name(idx)))
    _net = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).cuda()

    if(device.type =="cuda")and(torch.cuda.device_count()>1):
        print('Multi GPU activate')
    model = nn.DataParallel(_net,device_ids=[0,1,2,3])
    opt = optim.Adam(model.parameters(), amsgrad=True)
    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []

    best_val_dsc = 0.0

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...')
    class_weights = torch.ones(17).to(device, dtype=torch.float)
    for epoch in tqdm(range(num_epochs)):

        # training
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        for i_batch, batched_sample in enumerate(train_loader):
            # send mini-batch to device
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            one_hot_labels = batched_sample['labels'].to(device, dtype=torch.long).squeeze()
            A_S = batched_sample['A_S'].to(device, dtype=torch.float)
            A_L = batched_sample['A_L'].to(device, dtype=torch.float)
            A_U = batched_sample['A_U'].to(device, dtype=torch.float)
            inputs[inputs!=inputs] = 0
            A_S[A_S!=A_S] = 0
            A_L[A_L!=A_L] = 0
            A_U[A_U!=A_U] = 0
            one_hot_labels[one_hot_labels!=one_hot_labels] = 0
            # zero the parameter gradients
            opt.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs, A_S, A_L,A_U)

            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # print statistics
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()


        # record losses and metrics
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        #reset
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            for i_batch, batched_val_sample in enumerate(val_loader):

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                one_hot_labels = batched_val_sample['labels'].to(device, dtype=torch.long).squeeze()
                A_S = batched_val_sample['A_S'].to(device, dtype=torch.float)
                A_L = batched_val_sample['A_L'].to(device, dtype=torch.float)
                A_U = batched_val_sample['A_U'].to(device, dtype=torch.float)
                inputs[inputs!=inputs] = 0
                A_S[A_S!=A_S] = 0
                A_L[A_L!=A_L] = 0
                A_U[A_U!=A_U] = 0
                one_hot_labels[one_hot_labels!=one_hot_labels] = 0
                outputs = model(inputs, A_S, A_L, A_U)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()


            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # reset
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

            # output current status
            print('*****/nEpoch: {}/{}, loss: {}, dsc: {}, sen: {}, ppv: {}/n         val_loss: {}, val_dsc: {}, val_sen: {}, val_ppv: {}/n*****'.format(epoch+1, num_epochs, losses[-1], mdsc[-1], msen[-1], mppv[-1], val_losses[-1], val_mdsc[-1], val_msen[-1], val_mppv[-1]))
            if use_visdom:
                plotter.plot('loss', 'train', 'Loss', epoch+1, losses[-1])
                plotter.plot('DSC', 'train', 'DSC', epoch+1, mdsc[-1])
                plotter.plot('SEN', 'train', 'SEN', epoch+1, msen[-1])
                plotter.plot('PPV', 'train', 'PPV', epoch+1, mppv[-1])
                plotter.plot('loss', 'val', 'Loss', epoch+1, val_losses[-1])
                plotter.plot('DSC', 'val', 'DSC', epoch+1, val_mdsc[-1])
                plotter.plot('SEN', 'val', 'SEN', epoch+1, val_msen[-1])
                plotter.plot('PPV', 'val', 'PPV', epoch+1, val_mppv[-1])

        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+checkpoint_name)

        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        model_path+'{}_best.tar'.format(model_name))

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv('losses_metrics_vs_epoch.csv')
