import argparse
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
import networks.network
from utils.trainer import run_training, val_epoch,test_epoch
from utils.data_utils import get_loader
from monai.transforms import  AsDiscrete
import random
import math

def main(train,val):
    parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
    parser.add_argument("--test_mode", default=True, type=bool, help="test mode or not")
    parser.add_argument("--label", default=False, type=bool, help="test data have label or not")
    parser.add_argument("--model_weight_name", default=r".\runs\model_final.pth", type=str,help="start training from saved checkpoint")
    #parser.add_argument("--model_weight_name", default=None, type=str, help="start training from saved checkpoint")
    parser.add_argument("--logdir", default=r"./runs/oneshot_ckpt", type=str, help="directory to save the tensorboard logs")
    parser.add_argument('--list_dir', type=str, default=r'.\dataset\training data', help='list dir')#
    parser.add_argument('--data_dir', type=str, default=r'.\dataset\training data', help='list dir')  #
    parser.add_argument("--train_text", default=train, type=str, help="training image list text name")
    parser.add_argument("--val_text", default=val, type=str, help="validating image list text name")
    parser.add_argument('--model_name', default='cross_attention_network', type=str, metavar='MODEL',help='Name of model to train')#vit_large_patch16
    parser.add_argument("--save_checkpoint", default=True,type=bool, help="save checkpoint during training")
    parser.add_argument("--max_epochs", default=200, type=int, help="max number of training epochs")
    parser.add_argument("--batch_size", default=24, type=int, help="number of batch size")
    parser.add_argument("--optim_lr", default=5*1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="sgd", type=str, help="optimization algorithm")#adamw
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    parser.add_argument("--val_every", default=10, type=int, help="validation frequency")
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=5, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
    parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")#
    parser.add_argument('--input_size', default=1024, type=int, help='images input size')  #
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument('--line_dim', default=512, type=int, help='images input size')
    parser.add_argument('--line_number', default=1024, type=int, help='images input size')
    parser.add_argument("--dropout_rate", default=0., type=float, help="dropout rate")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")#warmup_cosine, cosine_anneal
    parser.add_argument("--warmup_epochs", default=20, type=int, help="number of warmup epochs")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")

    args = parser.parse_args()
    args.amp = not args.noamp

    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)
    seed = 666
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    if args.label:
        loader = get_loader(args)
    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)

    if (args.model_name is None) or args.model_name in ['cross_attention_network']:
        model = networks.network.__dict__[args.model_name](
            line_dim=args.line_dim,
            line_number=args.line_number,
            dropout_rate=args.dropout_rate)

        # loading weights
        if not args.test_mode:
            if args.model_weight_name != None:
                model_dict = torch.load(args.model_weight_name, map_location="cpu")
                model.load_state_dict(model_dict['state_dict'])
        else:
                assert args.model_weight_name != None, 'No model weights loaded'
                model_dict = torch.load(args.model_weight_name, map_location="cpu")
                model.load_state_dict(model_dict['state_dict'])
    else:
        raise ValueError("Unsupported model " + str(args.model_name))

    post_label = AsDiscrete(to_onehot=True, n_classes=args.out_channels)
    post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.out_channels)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    start_epoch = 0

    model.cuda(args.gpu)

    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        lr_min = 0
        lr_max= args.optim_lr
        lambda0 = lambda cur_iter: (cur_iter / args.warmup_epochs) if (cur_iter < args.warmup_epochs) else \
            (lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos((cur_iter - args.warmup_epochs) / (args.max_epochs - args.warmup_epochs) * math.pi)))/args.optim_lr
        # LambdaLR
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda0])

    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.parent_ckpt is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    if not args.test_mode:
        val_loss = run_training(
            model=model,
            train_loader=loader[0],
            val_loader=loader[1],
            optimizer=optimizer,
            args=args,
            model_inferer=None,
            scheduler=scheduler,
            start_epoch=start_epoch,
            post_label=post_label,
            post_pred=post_pred,
        )
    elif args.test_mode and args.label:
        accuracy, run_loss = val_epoch(
            model=model,
            loader=loader,
            args=args
        )
        print("final acc:", accuracy, 'loss:', run_loss)
    elif args.test_mode and not args.label:
        accuracy, run_loss = test_epoch(
            model=model,
            args=args
        )
    return val_loss


if __name__ == "__main__":
    train = 'train'
    val = 'val'
    main(train,val)
