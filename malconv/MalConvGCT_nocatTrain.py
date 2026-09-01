import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import  DataLoader
from MalConvGCT_nocat import MalConvGCT
from binaryLoader import BinaryDataset, RandomChunkSampler, pad_collate_func
from sklearn.metrics import roc_auc_score
import argparse

#------------------------------------------
# RUN command for windows malware: python MalConvGCT_nocatTrain.py ../MalwareDataset/windows/altered/ ../MalwareDataset/windows/goodware/ FAMILY_NAME windows
# RUN command for android malware: python MalConvGCT_nocatTrain.py ../MalwareDataset/android/altered/ ../MalwareDataset/android/goodware/ FAMILY_NAME android
# set the OS variable as "windows" or "android"
#------------------------------------------

#Check if the input is a valid directory
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser(description='Train a MalConv model')

parser.add_argument('--filter_size', type=int, default=256, help='How wide should the filter be')
parser.add_argument('--filter_stride', type=int, default=64, help='Filter Stride')
parser.add_argument('--embd_size', type=int, default=8, help='Size of embedding layer')
parser.add_argument('--num_channels', type=int, default=128, help='Total number of channels in output')
parser.add_argument('--epochs', type=int, default=30, help='How many training epochs to perform')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')
parser.add_argument('--gpus', nargs='+', type=int)
parser.add_argument('mal_dir', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_dir', type=dir_path, help='Path to directory containing benign files for training')
parser.add_argument('target_family', type=str, help='Target malware family')

args = parser.parse_args()

GPUS = args.gpus
NON_NEG = args.non_neg
EMBD_SIZE = args.embd_size
FILTER_SIZE = args.filter_size
FILTER_STRIDE = args.filter_stride
NUM_CHANNELS= args.num_channels
EPOCHS = args.epochs
MAX_FILE_LEN = args.max_len
BATCH_SIZE = args.batch_size
target_family = args.target_family

ben_train = pd.read_csv('../data/real_data/training_goodware.csv')
mal_train = pd.read_csv('../data/real_data/training_malware.csv')

ben_dir = args.ben_dir
mal_dir = args.mal_dir

train_dataset = BinaryDataset(ben_dir, mal_dir, ben_train, mal_train, sort_by_size=True, max_len=MAX_FILE_LEN)
loader_threads = 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func,
                          sampler=RandomChunkSampler(train_dataset, BATCH_SIZE))

if GPUS is None:#use ALL of them! (Default) 
    device_str = "cuda:0"
else:
    if GPUS[0] < 0:
        device_str = "cpu"
    else:
        device_str = "cuda:{}".format(GPUS[0])
    

device = torch.device(device_str if torch.cuda.is_available() else "cpu")

model = MalConvGCT(channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE, low_mem=False).to(device)

if GPUS is None or len(GPUS) > 1:
    model = nn.DataParallel(model, device_ids=GPUS)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
    
print("\n================ TRAINING ================\n")

for epoch in tqdm(range(EPOCHS)):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels, _, _ in train_loader:
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs, penultimate_activ, conv_active = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if NON_NEG:
            for p in model.parameters():
                p.data.clamp_(0)

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    print(
        f"Epoch {epoch+1}/{EPOCHS}  |  "
        f"Train Acc: {train_correct/train_total:.4f}  |  "
        f"Loss: {running_loss:.4f}"
    )

print("\nTraining finished.\n")

print("Saving final model...")

model_path = os.path.join('../models', target_family + ".checkpoint")

if isinstance(model, nn.DataParallel):
    model_state = model.module.state_dict()
else:
    model_state = model.state_dict()

torch.save({
    'epoch': EPOCHS,
    'model_state_dict': model_state,
    'optimizer_state_dict': optimizer.state_dict(),
    'channels': NUM_CHANNELS,
    'filter_size': FILTER_SIZE,
    'stride': FILTER_STRIDE,
    'embd_dim': EMBD_SIZE,
    'non_neg': NON_NEG,
}, model_path)

print(f"Final model saved to {model_path}")