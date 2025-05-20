import sys
import os
from loader import get_loader
from models.Net import MEPSeg
import os
import torch
import numpy as np
from tqdm import tqdm
from utils.metrics import get_metrics
sys.path.append(os.getcwd())
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
        help="datasets name",
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
        default=0,
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
        "--betas",
        type=tuple,
        default=(0.9, 0.999),
        help="BetaS",
    )
    parser.add_argument(
        "--amsgrad",
        type=bool,
        default=False,
        help="AMSGrad variant",
    )
    parser.add_argument(
        "--eta_min",
        type=float,
        default=0.00001,
        help="Minimum learning rate",
    )
    parser.add_argument(
        "--last_epoch",
        type=int,
        default=-1,
        help="The index of last epoch",
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



class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        set_seed(args.random)
        
        self.checkpoint_path = os.path.join(os.getcwd(), args.checkpoint, args.datasets)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
            
        self.logger = get_logger('train', os.path.join(os.getcwd(), args.log, args.datasets))
        
        self.model = MEPSeg(
            input_channels=args.input_channels,
            out_channels=args.out_channels,
            kernel_list=args.kernel_list,
            wavelet=args.wavelet,
            level=args.level,
            mode=args.mode,
            dilation_values=args.dilation_values,
            conv_kernels=args.conv_kernels
        ).to(self.device)
        
        calculate_params_flops(self.model, size=args.imagesize)
        
        self.criterion = BceDiceLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=False
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.T_max,
            eta_min=0.00001,
            last_epoch=-1
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            betas=args.betas,
            eps=args.eps,
            weight_decay=args.weight_decay,
            amsgrad=args.amsgrad
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=args.T_max,
            eta_min=args.eta_min,
            last_epoch=args.last_epoch
        )
        
        self.start_epoch = 0
        self.min_miou = 0
        self.steps = 0
        
        if args.continues:
            self._load_checkpoint()

    def _load_checkpoint(self):
        if not os.path.exists(self.checkpoint_path):
            print('no files')
            raise FileNotFoundError(f"No checkpoint found at {self.checkpoint_path}")
        else:
            self.model,self.start_epoch,self.optimizer=continue_train(self.model,self.optimizer,self.checkpoint_path)

    def _train_epoch(self, epoch, train_loader):
        self.model.train()
        loss_list = []
        for step, data in enumerate(train_loader):
            self.steps += step
            self.optimizer.zero_grad()
            images, gts = data
            images, gts = images.to(self.device).float(), gts.to(self.device).float()
            
            pred = self.model(images)
            loss = self.criterion(pred[0], gts)
            for i in range(1, len(pred)):
                loss += self.criterion(pred[i], gts)
                
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
            
            if step % self.args.save_cycles == 0:
                lr = self.optimizer.param_groups[0]['lr']
                log_info = f'Train: Epoch={epoch}, Step={step}, Loss={np.mean(loss_list):.4f}, LR={lr:.7f}'
                print(log_info)
                self.logger.info(log_info)
                
        self.scheduler.step()
        return np.mean(loss_list)

    def _validate(self, val_loader):
        self.model.eval()
        loss_list = []
        preds = []
        gts = []
        with torch.no_grad():
            for data in tqdm(val_loader):
                images, gt, _ = data
                images, gt = images.to(self.device).float(), gt.to(self.device).float()
                pred = self.model(images)
                loss = self.criterion(pred[0], gt)
                loss_list.append(loss.item())
                gts.append(gt.squeeze(1).cpu().numpy())
                preds.append(pred[0].squeeze(1).cpu().numpy())
                
        log_info, miou = get_metrics(preds, gts)
        avg_loss = np.mean(loss_list)
        log_info = f'Val Loss: {avg_loss:.4f} | {log_info}'
        print(log_info)
        self.logger.info(log_info)
        return avg_loss, miou

    def train(self):
        train_loader = get_loader(self.args.datasets, self.args.batchsize, 
                                self.args.imagesize, mode=TRAIN)
        val_loader = get_loader(self.args.datasets, self.args.batchsize,
                              self.args.imagesize, mode=TEST)
                              
        for epoch in range(self.start_epoch, self.args.epoch):
            train_loss = self._train_epoch(epoch, train_loader)
            
            val_loss, current_miou = self._validate(val_loader)
            
            if current_miou > self.min_miou:
                self.min_miou = current_miou
                torch.save({
                    'epoch': epoch,
                    'min_miou': self.min_miou,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, os.path.join(self.checkpoint_path, 'best.pth'))
                self.logger.info(f"Saved best model at epoch {epoch} with mIoU {current_miou:.4f}")
                
                
                
if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    trainer.train()