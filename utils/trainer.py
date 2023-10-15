import os
import shutil
import time
import cv2
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather
from scipy import ndimage
import pandas as pd
import scipy

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, args):
    model.train()
    smoothloss = torch.nn.SmoothL1Loss()

    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):

        ref, obj, label, _ = batch_data
        ref, obj, label = ref.cuda(args.rank), obj.cuda(args.rank), label.cuda(args.rank)

        for param in model.parameters():
            param.grad = None

        with autocast(enabled=args.amp):
            predict_vector,predict_vector2 = model(ref, obj)
            predict_vector, label = predict_vector.squeeze(-1), label.squeeze(-1)

            vector_consistency_loss = smoothloss(predict_vector.float(), label.float())

            front_lines = predict_vector[:,0:args.input_size-1]
            back_lines = predict_vector[:,1:args.input_size]
            smoothness_loss = smoothloss(front_lines.float(),back_lines.float())

            front_lines = predict_vector2[:, 0:args.input_size - 1]
            back_lines = predict_vector2[:, 1:args.input_size]
            smoothness_loss2 = smoothloss(front_lines.float(), back_lines.float())

            ref = torch.mean(ref, dim=-1).squeeze(1)*256
            obj = torch.mean(obj, dim=-1).squeeze(1)*256
            x = torch.arange(0,args.input_size,1).unsqueeze(0).repeat([predict_vector.shape[0],1]).cuda(args.rank)
            predict_vector_int = torch.round(predict_vector + x)
            ref_1 = torch.zeros((predict_vector.shape[0],args.input_size)).cuda(args.rank)
            predict_vector_int[predict_vector_int >= args.input_size] -= args.input_size
            predict_vector_int[predict_vector_int < 0] += args.input_size
            for i in range(predict_vector.shape[0]):
                ref_1[i] = ref[i][predict_vector_int[i].long()]
            image_consistency_loss = smoothloss(ref_1.float(), obj.float())

            predict_vector_int2 = torch.round(predict_vector2 + x)
            obj_2 = torch.zeros((predict_vector2.shape[0], args.input_size)).cuda(args.rank)
            predict_vector_int2[predict_vector_int2 >= args.input_size] -= args.input_size
            predict_vector_int2[predict_vector_int2 < 0] += args.input_size
            for i in range(predict_vector2.shape[0]):
                obj_2[i] = obj[i][predict_vector_int2[i].long()]
            image_consistency_loss2 = smoothloss(obj_2.float(), ref.float())

            loss = vector_consistency_loss + 0.1*(smoothness_loss + image_consistency_loss + smoothness_loss2 + image_consistency_loss2)

        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(loss.item()),
                "loss1:{:.4f}".format(vector_consistency_loss.item()),
                "loss2:{:.4f}".format(smoothness_loss.item()),
                "loss3:{:.4f}".format(smoothness_loss2.item()),
                "loss4:{:.4f}".format(image_consistency_loss.item()),
                "loss5:{:.4f}".format(image_consistency_loss2.item()),
                "lr: {:.8f}".format(optimizer.param_groups[0]['lr']),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None

    return run_loss.avg

def val_epoch(model, loader, epoch=None, args=None):
    model.eval()
    start_time = time.time()
    smoothloss = torch.nn.SmoothL1Loss()
    run_loss = AverageMeter()

    f = open(os.path.join(args.list_dir, 'val_pre.txt'), mode='w')

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            ref, obj, label, name = batch_data
            ref, obj, label = ref.cuda(args.rank), obj.cuda(args.rank), label.cuda(args.rank)
            with autocast(enabled=args.amp):
                predict_vector,predict_vector2 = model(ref,obj)
                predict_vector, label = predict_vector.squeeze(-1), label.squeeze(-1)
                vector_consistency_loss = smoothloss(predict_vector.float(), label.float())

                front_lines = predict_vector[:, 0:args.input_size-1]
                back_lines = predict_vector[:, 1:args.input_size]
                smoothness_loss = smoothloss(front_lines.float(), back_lines.float())

                front_lines = predict_vector2[:, 0:args.input_size - 1]
                back_lines = predict_vector2[:, 1:args.input_size]
                smoothness_loss2 = smoothloss(front_lines.float(), back_lines.float())

                ref = torch.mean(ref, dim=-1).squeeze(1) * 256
                obj = torch.mean(obj, dim=-1).squeeze(1) * 256
                x = torch.arange(0, args.input_size, 1).unsqueeze(0).repeat([predict_vector.shape[0], 1]).cuda(args.rank)
                predict_vector_int = torch.round(predict_vector) + x
                ref_1 = torch.zeros((predict_vector.shape[0], args.input_size)).cuda(args.rank)
                predict_vector_int[predict_vector_int >= args.input_size]  -= args.input_size
                predict_vector_int[predict_vector_int < 0] += args.input_size
                for i in range(predict_vector.shape[0]):
                    ref_1[i] = ref[i][predict_vector_int[i].long()]
                image_consistency_loss = smoothloss(ref_1.float(), obj.float())

                predict_vector_int2 = torch.round(predict_vector2 + x)
                obj_2 = torch.zeros((predict_vector2.shape[0], args.input_size)).cuda(args.rank)
                predict_vector_int2[predict_vector_int2 >= args.input_size] -= args.input_size
                predict_vector_int2[predict_vector_int2 < 0] += args.input_size
                for i in range(predict_vector2.shape[0]):
                    obj_2[i] = obj[i][predict_vector_int2[i].long()]
                image_consistency_loss2 = smoothloss(obj_2.float(), ref.float())

                loss = vector_consistency_loss + 0.1*(smoothness_loss + image_consistency_loss + smoothness_loss2 + image_consistency_loss2)

                run_loss.update(loss.item(), n=predict_vector.shape[0])

                predict_vector = predict_vector.cpu().numpy()
                for i in range(len(predict_vector)):
                    f.write(name[i])
                    vector = ""
                    for j in range(len(predict_vector[i])):
                        vector += (" " + str(predict_vector[i][j]))
                    f.write(vector)
                    f.write("\n")

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "loss1:{:.4f}".format(vector_consistency_loss.item()),
                    "loss2:{:.4f}".format(smoothness_loss.item()),
                    "loss3:{:.4f}".format(smoothness_loss2.item()),
                    "loss4:{:.4f}".format(image_consistency_loss.item()),
                    "loss5:{:.4f}".format(image_consistency_loss2.item()),
                    'loss:',loss.cpu().numpy(),
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    f.close()
    return run_loss.avg

def test_epoch(model, epoch=None, args=None):
    model.eval()

    f = open(os.path.join(args.list_dir, 'test_pre.txt'), mode='w')

    with torch.no_grad():
        index_current = np.zeros((1024),dtype=np.float32)
        raw_path = r".\dataset\test sequence"
        imgNames = os.listdir(raw_path)
        imgNames = sorted(imgNames, key = lambda x:int(x.replace('.png','')))
        print(imgNames)
        timeList = []
        for i in range(len(imgNames)-1):
            obj = cv2.imread(os.path.join(raw_path, imgNames[i]), cv2.IMREAD_GRAYSCALE)
            obj_torch = (obj - np.min(obj)) / (np.max(obj) - np.min(obj))
            obj_torch = torch.from_numpy(np.transpose(obj_torch, (1, 0))).unsqueeze(0).unsqueeze(0).cuda(args.rank).float()

            ref = cv2.imread(os.path.join(raw_path, imgNames[i+1]), cv2.IMREAD_GRAYSCALE)
            ref_torch = (ref - np.min(ref)) / (np.max(ref) - np.min(ref))
            ref_torch = torch.from_numpy(np.transpose(ref_torch, (1, 0))).unsqueeze(0).unsqueeze(0).cuda(args.rank).float()

            with autocast(enabled=args.amp):
                start_time = time.time()
                predict_vector, predict_vector2 = model(obj_torch,ref_torch)

                predict_vector = predict_vector.squeeze(-1)
                predict_vector = predict_vector.cpu().numpy()

                f.write(imgNames[i])
                vector = ""
                for j in range(len(predict_vector[0])):
                    vector += (" " + str(int(predict_vector[0][j])))
                f.write(vector)
                f.write("\n")

                #derived from the relationship between the first frame and the n-th frame
                #by the relationship between n-1-th and the first frame, and the relationship between n-1-th and the n-th frame
                index = predict_vector[0]
                print(index)
                num = len(index)
                index_tmp = np.concatenate([index,index,index],axis=0,dtype=np.float32)
                index_current_tmp = np.concatenate([index_current,index_current,index_current],axis=0,dtype=np.float32)
                index_new = index_tmp*np.nan
                # deal with the boundary exceeding
                for j in range(len(index_tmp)):
                    new_position = int(index_tmp[j]+j)
                    if new_position<0:
                        new_position=0
                    elif new_position>=len(index_tmp):
                        new_position = len(index_tmp)-1
                    # move this line to new position
                    index_new[j] = index_tmp[j] + index_current_tmp[new_position]
                index_new = pd.Series(index_new).interpolate()
                index_new = index_new[num:2*num]#Get the relationship between frame 1-th and frame n-th, and save it for the next loop
                index_current = scipy.ndimage.gaussian_filter1d(index_new, 20)

                #Align the n-th frame to the first frame
                h, w = ref.shape
                ref_before = cv2.imread(os.path.join(raw_path, imgNames[i]), cv2.IMREAD_GRAYSCALE)
                if i+2 == len(imgNames):
                    ref_after = cv2.imread(os.path.join(raw_path, imgNames[i + 1]), cv2.IMREAD_GRAYSCALE)
                else:
                    ref_after = cv2.imread(os.path.join(raw_path, imgNames[i + 2]), cv2.IMREAD_GRAYSCALE)
                ref_tmp = np.concatenate([ref_before, ref, ref_after], axis=1)
                ref_new = ref_tmp*0
                index_current_tmp = np.concatenate([index_current,index_current,index_current],axis=0,dtype=np.float32)
                index_current_tmp = scipy.signal.resample(index_current_tmp,3*w)
                index_current_new = index_current_tmp*np.nan

                for j in range(len(index_current_tmp)):
                    new_position = index_current_tmp[j]+j
                    if new_position<0:
                        new_position=0
                    elif new_position>=len(index_current_tmp):
                        new_position = len(index_current_tmp)-1
                    index_current_new[int(new_position)] = float(j)
                index_current_new = pd.Series(index_current_new).interpolate()
                index_current_new = scipy.ndimage.gaussian_filter1d(index_current_new, 30)

                index_current_new = np.round(index_current_new)
                index_current_new = np.array(index_current_new,dtype=np.int16)
                index_current_new[index_current_new<=0] = 0
                index_current_new[index_current_new >= 3 * w] = 3*w-1
                ref_new = ref_tmp[:,index_current_new]

                ref_pre = ref_new[:,w:2*w]
                if not os.path.exists(os.path.join(os.path.split(raw_path)[0], 'correction')):
                    os.makedirs(os.path.join(os.path.split(raw_path)[0], 'correction'))
                cv2.imwrite(os.path.join(os.path.split(raw_path)[0], 'correction', imgNames[i + 1]), ref_pre)
                timeList.append((time.time() - start_time)*1000)

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs,i,len(imgNames)-1),
                    "time {:.2f}s".format(time.time() - start_time),
                )
        print("average time: ", np.mean(timeList[1:]), "std time: ",np.std(timeList[1:]))

    return 0

def save_checkpoint(model, epoch, args, filename="model.pth", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    spend_time = 0
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_loss_best = 100000
    for epoch in range(start_epoch, args.max_epochs):

        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, args=args
        )

        if scheduler is not None:
            scheduler.step()

        spend_time += time.time() - epoch_time
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
            with open(os.path.join(args.logdir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write("Final training:{}/{},".format(epoch, args.max_epochs - 1) + "loss:{}".format(train_loss) + "\n")
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_loss = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    'loss:',val_loss,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                with open(os.path.join(args.logdir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write("Final validation:{}/{},".format(epoch, args.max_epochs - 1)
                            + "loss:{},".format(val_loss)+ "\n")
                if writer is not None:
                    writer.add_scalar("val_loss", val_loss, epoch)
                if val_loss < val_loss_best:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_loss_best, val_loss))
                    val_loss_best = val_loss
                    b_new_best = True
                    shutil.copyfile(os.path.join(args.list_dir, 'val_pre.txt'), os.path.join(args.list_dir, 'val_pre_best.txt'))

                    # save weights
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_loss_best, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_loss_best, filename="model_final.pth")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pth"),
                                    os.path.join(args.logdir, "model.pth"))

    print("Training Finished !, Best loss: ", val_loss_best, "Total time: {} s.".format(round(spend_time)))

    return val_loss_best