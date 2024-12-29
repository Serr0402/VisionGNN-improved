import time
import logging
import configparser
from pathlib import Path
from datetime import datetime
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Classifier
from dataset import ImageNetteDataset

def set_seed(seed):
    """Fija la semilla para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def load_dataset(path, batch_size):
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]
    # is the order of tranfoms important? Is this the best order?
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means,
                             std=pretrained_stds),
    ])

    # cutmix = v2.CutMix(num_classes=NUM_CLASSES)
    # mixup = v2.MixUp(num_classes=NUM_CLASSES)
    # cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    train_dataset = ImageNetteDataset(
        path, split='train', transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_dataset = ImageNetteDataset(path, split='val', transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader


def train_step(model, dataloader, optimizer, criterion, device, epoch=None, mix_aug=None):
    running_loss, correct, total = [], 0, 0
    model.train()

    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device).long()

        if mix_aug is not None:
            x, y = mix_aug(x, y)

        optimizer.zero_grad()

        _, pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # predicted_class = pred.argmax(dim=1, keepdim=False)

        # total += y.numel()
        # correct += (predicted_class == y).sum().item()

        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
    # acc = correct / total
    acc = None
    return sum(running_loss) / len(running_loss), acc


def validation_step(model, dataloader, device):
    correct_top1, correct_top5, total = 0, 0, 0
    model.eval()

    validation_bar = tqdm(dataloader)
    with torch.no_grad():
        for x, y in validation_bar:
            x, y = x.to(device), y.to(device)

            _, pred = model(x)

            # Top-1 accuracy
            predicted_class = pred.argmax(dim=1, keepdim=False)
            correct_top1 += (predicted_class == y).sum().item()

            # Top-5 accuracy
            top5_pred = torch.topk(pred, 5, dim=1).indices
            correct_top5 += sum([y[i] in top5_pred[i] for i in range(len(y))])

            total += y.numel()

            acc_top1 = correct_top1 / total
            acc_top5 = correct_top5 / total
            validation_bar.set_description(
                f'Top-1 accuracy: {round(acc_top1*100, 2)}%, Top-5 accuracy: {round(acc_top5*100, 2)}% until now.')
    return acc_top1, acc_top5



def train(conf, device):
    save_dir = Path(conf['TRAIN']['SAVE_DIR']) / datetime.now().strftime('%Y%m%d_%H%M')
    save_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(save_dir / 'train.log'),
        logging.StreamHandler()
    ])

    # Agregar prints y logs para verificar los parámetros
    print("Configuración cargada del archivo ini:")
    for section in conf.sections():
        print(f"{section}: {dict(conf[section])}")
    logging.info("Configuración cargada del archivo ini:")
    logging.info({section: dict(conf[section]) for section in conf.sections()})

    # Inicializar el modelo
    # model = Classifier(
    #     n_classes=conf['DATASET'].getint('NUM_CLASSES'),
    #     num_ViGBlocks=conf['MODEL'].getint('DEPTH'),
    #     out_feature=conf['MODEL'].getint('DIMENSION'),
    #     num_edges=conf['MODEL'].getint('NUM_EDGES'),
    #     head_num=conf['MODEL'].getint('HEAD_NUM')         #esto era el original
    # )
    model = Classifier(
        n_classes=conf['DATASET'].getint('NUM_CLASSES'),
        num_ViGBlocks=conf['MODEL'].getint('DEPTH'),
        out_feature=conf['MODEL'].getint('DIMENSION'),
        num_edges=conf['MODEL'].getint('NUM_EDGES'),
        head_num=conf['MODEL'].getint('HEAD_NUM'),
        patch_size=conf['MODEL'].getint('PATCH_SIZE')  # Pasar PATCH_SIZE al modelo -> bug inicial
    )

    model.to(device)

    # Imprimir detalles del modelo
    # print("Detalles del modelo:")
    # print(model)
    # logging.info("Detalles del modelo:")
    # logging.info(model)

    train_dataloader, val_dataloader = load_dataset(
        conf['DATASET']['PATH'], conf['TRAIN'].getint('BATCH_SIZE')
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=conf['TRAIN'].getfloat('LR'))

    loss_hisroty, train_acc_hist, val_acc_hist = [], [], []
    max_val_acc = 0

    since = time.time()
    for epoch in range(1, conf['TRAIN'].getint('EPOCHS') + 1):
        loss, train_acc = train_step(model, train_dataloader, optimizer, criterion, device, epoch)
        val_acc_top1, val_acc_top5 = validation_step(model, val_dataloader, device)

        loss_hisroty.append(loss)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc_top1)

        logging.info(f'Epoch: {epoch}, Loss: {loss}, Val Top-1 acc: {val_acc_top1 * 100}, Val Top-5 acc: {val_acc_top5 * 100}')

        if val_acc_top1 > max_val_acc:
            max_val_acc = val_acc_top1
            torch.save(model.state_dict(), save_dir / f'best_model.pth')

    logging.info('Training Finished.')
    logging.info(f'Max validation Top-1 accuracy is {round(max_val_acc * 100, 2)}%')
    logging.info(f'elapsed time is {time.time() - since}')

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf = configparser.ConfigParser()
    conf.read('confs/main.ini')
    train(conf, device)
