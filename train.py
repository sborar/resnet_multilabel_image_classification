from tqdm import tqdm
import model
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from dataset import ImageDataset
from torch.utils.data import DataLoader
import wandb
import sklearn.metrics as metrics
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, type=str, metavar='DIR',
                    help='path to checkpoint')
parser.add_argument('--learning_rate', default=0.0001, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=35, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=64, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--pretrained', default=False, type=bool, metavar='DIR',
                    help='pretrain with imagenet')
parser.add_argument('--freeze', default=False, type=bool, metavar='DIR',
                    help='freeze resnet layers')
parser.add_argument('--metric_path_suffix', default='.csv', type=str, metavar='DIR',
                    help='metric path suffix')
args = parser.parse_args()

matplotlib.style.use('ggplot')
# initialize the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

wandb.login(key='ed94033c9c3bebedd51d8c7e1daf4c6eafe44e09')
wandb.init(project='resnet-classification', entity='sborar')
config = wandb.config

# read the training csv file
train_csv = pd.read_csv('data/train_data.csv')
valid_csv = pd.read_csv('data/valid_data.csv')
num_classes = len(train_csv.columns) - 1

# track metrics
columns = ['epoch', 'lr', 'batch_size', 'undersample_frac', 'structure', 'tn', 'fp', 'fn', 'tp', 'f1']
train_metric_table = wandb.Table(columns=columns)
test_metric_table = wandb.Table(columns=columns)
train_metric_df = pd.DataFrame(columns=columns)
test_metric_df = pd.DataFrame(columns=columns)

train_metric_path = 'train_metrics' + args.metric_path_suffix
test_metric_path = 'test_metrics' + args.metric_path_suffix
train_metric_df.to_csv(train_metric_path, index=False)
test_metric_df.to_csv(test_metric_path, index=False)


def print_metrics(target, output, num_classes, structures, train, df):
    print('metrics')
    f1_dict = {}
    for i in range(num_classes):
        tag = structures[i] + '_f1'
        cm = metrics.confusion_matrix(target[:, i], output[:, i])
        f1 = metrics.f1_score(target[:, i], output[:, i], average='macro')
        metric_table_entry = [epoch, lr, batch_size, frac, structures[i], cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1], f1]
        metric_series = pd.Series(metric_table_entry, index=df.columns)

        f1_dict[tag] = f1
        print(tag, f1_dict[tag])
        print(structures[i] + '_classification report', metrics.classification_report(target[:, i], output[:, i]))

        if train:
            print('log train metric data in wandb')
            train_metric_table.add_data(*metric_table_entry)
        else:
            print('log test metric data in wandb')
            test_metric_table.add_data(*metric_table_entry)
        df = df.append(metric_series, ignore_index=True)
    wandb.log(f1_dict)
    return df


# training function
def train(model, train_csv, optimizer, criterion, device, structures, frac, df):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    train_data = ImageDataset(
        train_csv, train=True
    )
    # train data loader
    dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )
    epoch_outputs = np.array([])
    epoch_targets = np.array([])
    for i, data in enumerate(dataloader):
        counter += 1
        data, target = data['image'].to(device), data['label'].to(device)
        w = target != 0
        target[target == -1] = 0
        optimizer.zero_grad()

        output_probs = model(data)
        output_probs = output_probs.view(output_probs.shape[0], num_classes, 2)
        outputs = torch.argmax(output_probs, dim=2)

        if epoch_targets.size == 0:
            epoch_targets = target.detach().cpu().numpy()
        else:
            epoch_targets = np.concatenate((epoch_targets, target.detach().cpu().numpy()))

        if epoch_outputs.size == 0:
            epoch_outputs = outputs.detach().cpu().numpy()
        else:
            epoch_outputs = np.concatenate((epoch_outputs, outputs.detach().cpu().numpy()))

        loss_sum = torch.tensor(0)
        for i in range(num_classes):
            # get outputs and targets for each class
            class_output = output_probs[:, i, :]
            class_target = target[:, i].long()

            loss = criterion(class_output, class_target)
            weighted_loss = loss * w[:, i]
            loss_sum = loss_sum + weighted_loss.sum()
        train_running_loss += loss_sum.item()
        # back propagation
        loss_sum.backward()
        wandb.log({"train_loss": loss_sum.item()})
        # update optimizer parameters
        optimizer.step()
    df = print_metrics(epoch_targets, epoch_outputs, num_classes=num_classes, structures=structures, train=True, df=df)
    train_loss = train_running_loss / counter
    return train_loss, df


def validate(model, dataloader, criterion, device, structures, df):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        epoch_outputs = np.array([])
        epoch_targets = np.array([])
        for i, data in enumerate(dataloader):
            counter += 1
            data, target = data['image'].to(device), data['label'].to(device)
            w = target != 0

            target[target == -1] = 0
            if epoch_targets.size == 0:
                epoch_targets = target.detach().cpu().numpy()
            else:
                epoch_targets = np.concatenate((epoch_targets, target.detach().cpu().numpy()))

            output_probs = model(data)
            output_probs = output_probs.view(output_probs.shape[0], num_classes, 2)
            outputs = torch.argmax(output_probs, dim=2)
            if epoch_outputs.size == 0:
                epoch_outputs = outputs.detach().cpu().numpy()
            else:
                epoch_outputs = np.concatenate((epoch_outputs, outputs.detach().cpu().numpy()))
            loss_sum = torch.tensor(0)

            for i in range(num_classes):
                # get outputs and targets for each class
                class_output = output_probs[:, i, :]
                class_target = target[:, i].long()
                loss = criterion(class_output, class_target)
                weighted_loss = loss * w[:, i]
                loss_sum = loss_sum + weighted_loss.sum()

            wandb.log({"valid_loss": loss_sum.item()})
            val_running_loss += loss_sum.item()
        print_metrics(epoch_targets, epoch_outputs, num_classes=num_classes, structures=structures, train=False, df=df)
        val_loss = val_running_loss / counter
        return val_loss, df


def exclude_bias_and_norm(p):
    return p.ndim == 1


# learning parameters
lr = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
frac = 1

# intialize the model
model = model.resnet(num_classes=num_classes,
                     checkpoint_pth=args.checkpoint,
                     freeze=args.freeze,
                     pretrained=args.pretrained
                     ).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2]).to(device), reduction='none')

wandb.watch(model)

structures = train_csv.columns[1:]

# validation dataset
valid_data = ImageDataset(
    valid_csv
)

# validation data loader
valid_loader = DataLoader(
    valid_data,
    batch_size=batch_size
)

# start the training and validation
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss, train_metric_df = train(
        model, train_csv, optimizer, criterion, device, structures, frac, train_metric_df
    )
    valid_epoch_loss, test_metric_df = validate(
        model, valid_loader, criterion, device, structures, test_metric_df
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)

    print(len(train_metric_df))
    train_metric_df.to_csv(train_metric_path, mode='a', header=None, index=False)
    train_metric_df.to_csv(test_metric_path, mode='a', header=None, index=False)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

    # save the trained model to disk
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': criterion,
    }, 'checkpoint/model.pth')

wandb.log({"train_metrics": train_metric_table})
wandb.log({"test_metrics": test_metric_table})

# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()
