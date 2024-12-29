
import time
import logging
import configparser
import json
from pathlib import Path
from datetime import datetime
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
import random

from model2 import Classifier
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

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandAugment(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
        transforms.RandomErasing(),
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=pretrained_means, std=pretrained_stds),
    ])

    train_dataset = ImageNetteDataset(
        path, split='train', transform=transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=4, pin_memory=True)
    val_dataset = ImageNetteDataset(path, split='val', transform=transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
    return train_dataloader, val_dataloader

def train_step(model, dataloader, optimizer, criterion, device, epoch=None, mix_aug=None):
    running_loss = []
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

        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
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

            # Top-1 Accuracy
            predicted_class = pred.argmax(dim=1, keepdim=False)
            correct_top1 += (predicted_class == y).sum().item()

            # Top-5 Accuracy
            top5_pred = torch.topk(pred, 5, dim=1).indices
            correct_top5 += sum([y[i] in top5_pred[i] for i in range(len(y))])

            total += y.size(0)

            acc_top1 = correct_top1 / total
            acc_top5 = correct_top5 / total
            validation_bar.set_description(
                f'Top-1 accuracy: {round(acc_top1 * 100, 2)}%, Top-5 accuracy: {round(acc_top5 * 100, 2)}% so far.')
    
    return acc_top1, acc_top5

def save_checkpoint(save_path, model, optimizer, epoch, hyperparams, best_val_acc):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'hyperparams': hyperparams,
        'best_val_acc': best_val_acc,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")

def load_checkpoint(load_path, model, optimizer):
    if os.path.isfile(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        hyperparams = checkpoint['hyperparams']
        best_val_acc = checkpoint['best_val_acc']
        print(f"Checkpoint loaded from {load_path}")
        return epoch, hyperparams, best_val_acc
    else:
        print(f"No checkpoint found at {load_path}")
        return 0, None, 0  # Return default values if no checkpoint is found

def train(conf, device, hyperparams, save_dir, checkpoint_path):
    # Configurar logging
    logging.basicConfig(level=logging.INFO, handlers=[
        logging.FileHandler(save_dir / 'train.log'),
        logging.StreamHandler()
    ])
    logging.info('Model and training configuration set up.')

    model = Classifier(
        n_classes=conf['DATASET'].getint('NUM_CLASSES'),
        num_ViGBlocks=hyperparams['DEPTH'],
        out_feature=int(conf['MODEL']['DIMENSION']),
        num_edges=hyperparams['NUM_EDGES'],
        head_num=hyperparams['HEAD_NUM'],
        patch_size=hyperparams['PATCH_SIZE'],
        image_size=224
    )
    model.to(device)
    logging.info(f'Model loaded with hyperparameters: {hyperparams}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparams['LR'])

    start_epoch, _, max_val_acc = load_checkpoint(checkpoint_path, model, optimizer)

    train_dataloader, val_dataloader = load_dataset(
        conf['DATASET']['PATH'], hyperparams['BATCH_SIZE']
    )

    since = time.time()
    for epoch in range(start_epoch + 1, conf['TRAIN'].getint('EPOCHS') + 1):
        loss, train_acc = train_step(
            model, train_dataloader, optimizer, criterion, device, epoch)
        val_acc_top1, val_acc_top5 = validation_step(model, val_dataloader, device)

        logging.info(f'Epoch: {epoch}, Loss: {loss}, Val Top-1 acc: {val_acc_top1 * 100}, Val Top-5 acc: {val_acc_top5 * 100}')

        if val_acc_top1 > max_val_acc:
            max_val_acc = val_acc_top1

        # Save checkpoint after each epoch
        save_checkpoint(checkpoint_path, model, optimizer, epoch, hyperparams, max_val_acc)

    logging.info('Training Finished.')
    logging.info(f'Max validation Top-1 accuracy is {round(max_val_acc * 100, 2)}%')
    logging.info(f'Elapsed time is {time.time() - since}')

    return max_val_acc

def grid_search(conf, device):
    model_sizes = {
        "Tiny": {"DIMENSION": "192", "HEAD_NUM": [1, 2, 4], "NUM_EDGES": [6, 8, 9, 10, 12], 
                "PATCH_SIZE": [8], "DEPTH": [10, 12, 16], "LR": [0.00005], "BATCH_SIZE": [16]},
        "Small": {"DIMENSION": "320", "HEAD_NUM": [1, 2, 4], "NUM_EDGES": [6, 8, 10, 12], 
                "PATCH_SIZE": [8], "DEPTH": [10, 12], "LR": [0.00005], "BATCH_SIZE": [16]},
        "Big": {"DIMENSION": "640", "HEAD_NUM": [1, 4, 8], "NUM_EDGES": [6, 8, 9, 10, 12], 
                "PATCH_SIZE": [16], "DEPTH": [10, 14, 18, 20], "LR": [0.00005], "BATCH_SIZE": [16]}
    }


    # Define the model type here (Tiny, Small, Big)
    selected_model = "Big"  
    conf['MODEL']['DIMENSION'] = model_sizes[selected_model]["DIMENSION"]

    # Define hyperparameter space for the selected model
    hyperparams_space = {k: v for k, v in model_sizes[selected_model].items() if k != "DIMENSION"}
    keys, values = zip(*hyperparams_space.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    best_val_acc = 0
    best_hyperparams = None
    best_experiment_number = 0  # Initialize best_experiment_number
    resume_info_path = Path(f"train_log/experiments/{selected_model}_resume_info.json")

    # Load the resume info if it exists
    start_experiment = 0
    if resume_info_path.exists():
        with open(resume_info_path, 'r') as f:
            resume_info = json.load(f)
            start_experiment = resume_info.get('last_experiment', 0)
            best_val_acc = resume_info.get('best_val_acc', 0)
            best_hyperparams = resume_info.get('best_hyperparams', None)
            print(f"Resuming from experiment {start_experiment + 1}")

    for idx, hyperparams in enumerate(experiments[start_experiment:], start=start_experiment):
        print(f"\n----- Ejecutando experimento {idx + 1}/{len(experiments)} -----")
        print(f"Hiperparámetros: {hyperparams}")

        # Define el directorio de guardado para cada experimento
        save_dir = Path(conf['TRAIN']['SAVE_DIR']) / f"experiments/{selected_model}/experiment_{idx + 1}"
        save_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = save_dir / "checkpoint.pth"
        val_acc = train(conf, device, hyperparams, save_dir, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_hyperparams = hyperparams
            best_experiment_number = idx + 1  # Guarda el número de experimento

            # Guarda la configuración del mejor modelo
            best_model_info = {
                'best_val_acc': best_val_acc,
                'best_hyperparams': best_hyperparams,
                'best_experiment_number': best_experiment_number
            }
            with open(save_dir / 'best_model_info.json', 'w') as f:
                json.dump(best_model_info, f)

        # Guarda el progreso tras cada experimento
        with open(resume_info_path, 'w') as f:
            json.dump({
                'last_experiment': idx + 1,
                'best_val_acc': best_val_acc,
                'best_hyperparams': best_hyperparams,
                'best_experiment_number': best_experiment_number
            }, f)

        # Muestra el mejor experimento hasta el momento
        print("\n--- Mejor experimento hasta el momento ---")
        print(f"Número de experimento: {best_experiment_number}")
        print(f"Mejor precisión de validación: {round(best_val_acc * 100, 2)}%")
        print(f"Mejores hiperparámetros: {best_hyperparams}")

    print("\n--- Final de la búsqueda de hiperparámetros ---")
    print(f"Mejor precisión de validación: {best_val_acc}")
    print(f"Mejores hiperparámetros: {best_hyperparams}")
    print(f"Número del mejor experimento: {best_experiment_number}")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf = configparser.ConfigParser()
    conf.read('confs/main.ini')

    grid_search(conf, device)
