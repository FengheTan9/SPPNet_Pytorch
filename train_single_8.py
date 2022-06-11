import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from src.sppnet import SPPNet
from src.dataset import creat_imagenet_train_dataset, creat_imagenet_val_dataset

# define pytorch device - useful for device-agnostic execution
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# define model parameters
NUM_EPOCHS = 160  # original paper
BATCH_SIZE = 256
MOMENTUM = 0.9
LR_DECAY = 0.0001
# LR_DECAY = 0.0001
LR_INIT = 0.01
NUM_CLASSES = 1000  # 1000 classes for imagenet 2012 dataset
DEVICE_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # GPUs to use
# modify this to point to your data directory
INPUT_ROOT_DIR = 'sppnet_data_in'
TRAIN_IMG_DIR = '/root/dataset/imagenet_original/train'
VAL_IMAGE_DIR = '/root/dataset/imagenet_original/val'
OUTPUT_DIR = 'sppnet_data_out_256_self'
LOG_DIR = OUTPUT_DIR + '/tblogs'  # tensorboard logs
CHECKPOINT_DIR = OUTPUT_DIR + '/models'  # model checkpoints
# make checkpoint path directory
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

torch.cuda.set_device(DEVICE_IDS[0])


def evalute(model, loader, epoch):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = len(loader.dataset)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            logits = model(x)
            maxk = max((1, 5))
            y_resize = y.view(-1, 1)
            _, pred = logits.topk(maxk, 1, True, True)
            correct_top5 += torch.eq(pred, y_resize).sum().float().item()
            pred = logits.argmax(dim=1)
            correct_top1 += torch.eq(pred, y).sum().float().item()
    print("===============", epoch + 1, "===============", flush=True)
    print("top5_acc: {}\t,top1_acc: {}".format(correct_top5 / total, correct_top1 / total), flush=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

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
        self.avg = self.sum / self.count


if __name__ == '__main__':
    # print the seed value
    seed = torch.initial_seed()
    print('Used seed : {}'.format(seed))
    # create model
    sppnet = SPPNet(gpu_num=len(DEVICE_IDS), batch=BATCH_SIZE)

    # train on multiple GPUs
    sppnet = torch.nn.parallel.DataParallel(sppnet, device_ids=DEVICE_IDS)
    print('zfnet created')
    # create optimizer
    # the one that WORKS

    optimizer = optim.SGD(
        params=sppnet.parameters(),
        lr=LR_INIT,
        momentum=MOMENTUM,
        weight_decay=LR_DECAY)
    print('Optimizer created')

    dataloader_224, _ = creat_imagenet_train_dataset(train_dir=TRAIN_IMG_DIR, batch=BATCH_SIZE)
    dataloader_val = creat_imagenet_val_dataset(val_dir=VAL_IMAGE_DIR, batch=BATCH_SIZE)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150, eta_min=0.016)
    print('LR Scheduler created')
    print(device)

    # start training!!
    print('========Starting training=========', flush=True)
    sppnet.cuda()
    loss_fn = AverageMeter()
    for epoch in range(NUM_EPOCHS):
        total_steps = 1
        time_0 = time.time()
        sppnet.train()
        for imgs, classes in dataloader_224:
            imgs, classes = imgs.to(device), classes.to(device)
            optimizer.zero_grad()
            # calculate the loss
            output = sppnet(imgs).to(device)
            loss = nn.CrossEntropyLoss()(output, classes)
            loss_fn.update(loss.item(), imgs.size(0))
            # update the parameters
            loss.backward()
            optimizer.step()
            total_steps += 1

        print('224-Epoch: {} \t loss: {} \t avg:{} \t time:{}'.format(epoch + 1, loss_fn.val, loss_fn.avg,
                                                                      time.time() - time_0), flush=True)
        evalute(sppnet, dataloader_val, epoch)
        lr_scheduler.step()

        if epoch > 50:
            # save checkpoints
            checkpoint_path = os.path.join(CHECKPOINT_DIR, 'sppnet_states_e{}.pkl'.format(epoch + 1))
            state = {
                'epoch': epoch,
                'total_steps': total_steps,
                'optimizer': optimizer.state_dict(),
                'model': sppnet.state_dict(),
                'seed': seed,
            }
            torch.save(state, checkpoint_path)
