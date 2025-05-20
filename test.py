import os
from loader import get_loader
from models.Net import MEPSeg
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics
from utils.loss_function import BceDiceLoss
from utils.tools import continue_train, get_logger, calculate_params_flops,set_seed
import torch
import argparse

from micro import TEST,TRAIN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="Kvasir",
        help="input datasets name including ISIC2018, PH2, Kvasir, BUSI, COVID_19,CVC_ClinkDB,Monu_Seg",
    )
    parser.add_argument(
        "--batchsize",
        type=int,
        default="8",
        help="input batch_size",
    )
    parser.add_argument(
        "--imagesize",
        type=int,
        default=256,
        help="input image resolution.",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="log",
        help="input log folder: ./log",
    )
    parser.add_argument(
        "--continues",
        type=int,
        default=0,
        help="1: continue to run; 0: don't continue to run",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default='checkpoints',
        help="the checkpoint path of last model: ./checkpoints",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=6,
        help="gpu_id:",
    )
    parser.add_argument(
        "--random",
        type=int,
        default=42,
        help="random configure:",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="end epoch",
    )
    parser.add_argument(
        "--out_channels",
        type=list,
        default=[10,20,30,40,50],
        help="out_channels",
    )
    parser.add_argument(
        "--save_cycles",
        type=int,
        default=20,
        help="",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="",
    )
    parser.add_argument(
        "--T_max",
        type=int,
        default=50,
        help="",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-2,
        help="",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="",
    )
    parser.add_argument(
        "--input_channels",
        type=int,
        default=3,
        help="Number of input channels",
    )
    parser.add_argument(
        "--kernel_list",
        type=list,
        default=[3,9],
        help="List of kernel sizes",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default='haar',
        help="Type of wavelet",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=1,
        help="Wavelet decomposition level",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="replicate",
        help="Padding mode",
    )
    parser.add_argument(
        "--dilation_values",
        type=list,
        default=[2, 3, 4],
        help="List of dilation values",
    )
    parser.add_argument(
        "--conv_kernels",
        type=list,
        default=[(5, 1), (1, 5)],
        help="List of convolution kernels",
    )

    return parser.parse_args()


def val_epoch(val_loader,model,criterion,logger):
    model.eval()
    loss_list=[]
    preds = []
    gts = []
    with torch.no_grad():
        for data in tqdm(val_loader):
            images, gt,image_name = data
            images, gt = images.cuda().float(), gt.cuda().float()
            pred = model(images)
            loss = criterion(pred[0],gt)
            #record val loss
            loss_list.append(loss.item())
            #record gt and pred for the subsequent metric calculation.
            gts.append(gt.squeeze(1).cpu().detach().numpy())
            preds.append(pred[0].squeeze(1).cpu().detach().numpy()) 
    #calculate metrics
    log_info,miou=get_metrics(preds,gts)
    log_info=f'val loss={np.mean(loss_list):.4f}  {log_info}'
    print(log_info)
    logger.info(log_info)
    return np.mean(loss_list),miou



def main():
    #init GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print("Current device ID:", torch.cuda.current_device())
    else:
        print("no GPU devices")
    
    args=parse_args()
    #random
    set_seed(args.random)
    torch.cuda.empty_cache()
    #check folders
    checkpoint_path=os.path.join(os.getcwd(),args.checkpoint,args.datasets)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #record log
    logger = get_logger('train', os.path.join(os.getcwd(),args.log,args.datasets))
    model = MEPSeg(
        input_channels=args.input_channels,
        out_channels=args.out_channels,
        kernel_list=args.kernel_list,
        wavelet=args.wavelet,
        level=args.level,
        mode=args.mode,
        dilation_values=args.dilation_values,
        conv_kernels=args.conv_kernels
    )
    model = model.cuda()
    
    calculate_params_flops(model,size=256)
    #loss function
    criterion=BceDiceLoss()
    #set optim
    optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = args.lr,
            betas = (0.9,0.999),
            eps = args.eps,
            weight_decay = args.weight_decay,
            amsgrad = False
        )
    #set scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = args.T_max,
            eta_min = 0.00001,
            last_epoch = -1
        )
    start_epoch=0
    min_miou=0
    #continue to train
    if args.continues:
        model,start_epoch,min_miou,optimizer=continue_train(model,optimizer,checkpoint_path)
        lr=optimizer.state_dict()['param_groups'][0]['lr']
        print(f'start_epoch={start_epoch},min_miou={min_miou},lr={lr}')
    #testing sets
    val_loader=get_loader(args.datasets,args.batchsize,args.imagesize,mode=TEST)
    end_epoch=args.epoch
    steps=0
    #start to run the model
    for epoch in range(start_epoch, end_epoch):
        torch.cuda.empty_cache()
        loss,miou=val_epoch(val_loader,model,criterion,logger)

if __name__ == '__main__':
    main()
