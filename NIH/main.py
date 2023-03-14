from dataset import ChestXray14
from losses import ContrastiveLoss, SupConLoss
import os
import math
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve
import warnings
import matplotlib.pyplot as plt
from vit_pytorch.efficient import ViT
from linformer import Linformer
from torch.optim.lr_scheduler import StepLR
from torchvision import models
from model import get_encoder
from ray import tune


warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"


# Training settings
batch_size = 16
lr = 0.00024388078393846276
gamma = 0.7

model = get_encoder("vit").to(device)
# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
# optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=.9)

# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

def train(data_loader, val_data_loader, class_name, epoch_count, preprocess='orig'):
    train_losses, val_losses = [], []
    for epoch in range(1, epoch_count + 1):
        model.train()
        progress_bar = tqdm.tqdm(data_loader)
        epoch_loss = 0
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            loss = criterion(features, labels)  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Training Loss: {loss.item()}")
            epoch_loss += loss.item()
        torch.save(model.state_dict(), 'weights/{}_{}_{}_weights'.format(class_name, preprocess, epoch))
        print("Epoch: {} | Training Loss: {:.2f}".format(epoch, epoch_loss/len(data_loader)))
        train_losses.append(epoch_loss)
        
        val_progress_bar = tqdm.tqdm(val_data_loader)
        
        model.eval()
        val_epoch_loss = 0
        for images, labels in val_progress_bar:
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            val_loss = criterion(features, labels)  
            val_progress_bar.set_description(f"Validation Loss: {val_loss.item()}")
            val_epoch_loss += val_loss.item()
        print("Epoch: {} | Validation Loss: {:.2f}".format(epoch, val_epoch_loss/len(data_loader)))
        val_losses.append(val_epoch_loss)
    plt.plot(range(0, epoch_count), train_losses, label="train")
    plt.plot(range(0, epoch_count), val_losses, label="val")
def get_tuning_func(class_name = "Cardiomegaly"):
    def tuner(config, checkpoint_dir="./tuning_checkpoints"):
        train_dataset = ChestXray14(phase='train', class_name=class_name, transform_type=config['preprocess'])
        train_data_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
        val_dataset = ChestXray14(phase='val', class_name=class_name)
        val_data_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)

        net = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
        num_features = net.head.in_features
        net.head = nn.Linear(num_features, 2)
        train_losses, val_losses = [], []
        net.to('cuda')
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=.9)

        for epoch in range(1, 15):
            net.train()
            epoch_loss = 0
            for images, labels in train_data_loader:
                images, labels = images.to(device), labels.to(device)
                features = net(images)
                loss = criterion(features, labels)  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_losses.append(epoch_loss)

            net.eval()
            val_epoch_loss = 0
            for images, labels in val_data_loader:
                images, labels = images.to(device), labels.to(device)
                features = net(images)
                val_loss = criterion(features, labels)  
                val_epoch_loss += val_loss.item()
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((net.state_dict(), optimizer.state_dict()), path)
            tune.report(loss=(val_epoch_loss / len(val_data_loader)))
    return tuner
def main(class_name='Cardiomegaly', preprocess = "CLAHE", epoch_count=20):
    train_dataset = ChestXray14(phase='train', class_name=class_name)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataset = ChestXray14(phase='val', class_name=class_name)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print('Training on {} images ({})'.format(len(train_dataset), class_name))
    train(data_loader=train_data_loader, val_data_loader=val_data_loader, class_name=class_name, epoch_count=epoch_count, preprocess=preprocess)

if __name__ == '__main__':
    main()
