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
parser.add_argument('--epochs', type=int, default=10, help='How many training epochs to perform')
parser.add_argument('--non-neg', type=bool, default=False, help='Should non-negative training be used')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size during training')
#Default is set ot 16 MB! 
parser.add_argument('--max_len', type=int, default=16000000, help='Maximum length of input file in bytes, at which point files will be truncated')

parser.add_argument('--gpus', nargs='+', type=int)


parser.add_argument('mal_dir', type=dir_path, help='Path to directory containing malware files for training')
parser.add_argument('ben_dir', type=dir_path, help='Path to directory containing benign files for training')
# parser.add_argument('mal_test', type=dir_path, help='Path to directory containing malware files for testing')
# parser.add_argument('ben_test', type=dir_path, help='Path to directory containing benign files for testing')

parser.add_argument('family_target', type=str, help='Malware family target')

parser.add_argument('target_operating_system', type=str, help='operating system target')

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
family_target = args.family_target

OS = args.target_operating_system

ben_train = pd.read_csv(f'../MalwareDataset/{OS}/ben_train.csv')
mal_train = pd.read_csv(f'../MalwareDataset/{OS}/mal_train_{family_target}.csv')

ben_test = pd.read_csv(f'../MalwareDataset/{OS}/ben_test.csv')
mal_test = pd.read_csv(f'../MalwareDataset/{OS}/mal_test_{family_target}.csv')

print(f'Goodware. Train: {len(ben_train)}, Test: {len(ben_test)}\n')
print(f'Malware. Train: {len(mal_train)}, Test: {len(mal_test)}\n')

ben_dir = args.ben_dir
mal_dir = args.mal_dir

train_dataset = BinaryDataset(ben_dir, mal_dir, ben_train, mal_train, OS, sort_by_size=True, max_len=MAX_FILE_LEN)
test_dataset = BinaryDataset(ben_dir, mal_dir, ben_test, mal_test, OS, sort_by_size=True, max_len=MAX_FILE_LEN )

# loader_threads = max(multiprocessing.cpu_count()-4, multiprocessing.cpu_count()//2+1)
loader_threads = 0

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func,
                          sampler=RandomChunkSampler(train_dataset, BATCH_SIZE))

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=loader_threads, collate_fn=pad_collate_func, 
                        sampler=RandomChunkSampler(test_dataset,BATCH_SIZE))

if GPUS is None:#use ALL of them! (Default) 
    device_str = "cuda:0"
else:
    if GPUS[0] < 0:
        device_str = "cpu"
    else:
        device_str = "cuda:{}".format(GPUS[0])
    

device = torch.device(device_str if torch.cuda.is_available() else "cpu")

model = MalConvGCT(channels=NUM_CHANNELS, window_size=FILTER_SIZE, stride=FILTER_STRIDE, embd_size=EMBD_SIZE, low_mem=False).to(device)

base_name = "nocat_{}_channels_{}_filterSize_{}_stride_{}_embdSize_{}_maxFileLen_{}MB_target_family_{}".format(
        type(model).__name__,
        NUM_CHANNELS,
        FILTER_SIZE,
        FILTER_STRIDE,
        EMBD_SIZE,
        str(int(MAX_FILE_LEN/1000000)),
        family_target
    )

if NON_NEG:
    base_name = "NonNeg_" + base_name

if GPUS is None or len(GPUS) > 1:
    model = nn.DataParallel(model, device_ids=GPUS)

if not os.path.exists(base_name):
    os.makedirs(base_name)

# file_name = os.path.join(base_name, base_name)
# headers = ['epoch', 'train_acc', 'train_auc', 'test_acc', 'test_auc']
# csv_log_out = open(file_name + ".csv", 'w')
# csv_log_out.write(",".join(headers) + "\n")

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(model.parameters())

# for epoch in tqdm(range(EPOCHS)):
#
#     preds = []
#     truths = []
#     running_loss = 0.0
#
#
#     train_correct = 0
#     train_total = 0
#
#     epoch_stats = {'epoch':epoch}
#
#     model.train()
#     for inputs, labels, _, _ in tqdm(train_loader):
#
#         #inputs, labels = inputs.to(device), labels.to(device)
#         #Keep inputs on CPU, the model will load chunks of input onto device as needed
#         labels = labels.to(device)
#
#         optimizer.zero_grad()
#
#     #     outputs, penultimate_activ, conv_active = model.forward_extra(inputs)
#         outputs, penultimate_activ, conv_active = model(inputs)
#         loss = criterion(outputs, labels)
#         loss = loss #+ decov_lambda*(decov_penalty(penultimate_activ) + decov_penalty(conv_active))
#     #     loss = loss + decov_lambda*(decov_penalty(conv_active))
#         loss.backward()
#         optimizer.step()
#         if NON_NEG:
#             for p in model.parameters():
#                 p.data.clamp_(0)
#
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs.data, 1)
#
#         with torch.no_grad():
#             preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
#             truths.extend(labels.detach().cpu().numpy().ravel())
#
#         train_total += labels.size(0)
#         train_correct += (predicted == labels).sum().item()
#
#
#
#     #print("Training Accuracy: {}".format(train_correct*100.0/train_total))
#
#     epoch_stats['train_acc'] = train_correct*1.0/train_total
#     epoch_stats['train_auc'] = roc_auc_score(truths, preds)
#     #epoch_stats['train_loss'] = roc_auc_score(truths, preds)
#
#     #Save the model and current state!
#     model_path = os.path.join(base_name, "epoch_{}.checkpoint".format(epoch))
#
#
#     #Have to handle model state special if multi-gpu was used
#     if isinstance(model, torch.nn.DataParallel):#  type(model).__name__ is "DataParallel":
#         mstd = model.module.state_dict()
#     else:
#         mstd = model.state_dict()
#
#     torch.save({
#         'epoch': epoch,
#         'model_state_dict': mstd,
#         'optimizer_state_dict': optimizer.state_dict(),
#         'channels': NUM_CHANNELS,
#         'filter_size': FILTER_SIZE,
#         'stride': FILTER_STRIDE,
#         'embd_dim': EMBD_SIZE,
#         'non_neg': NON_NEG,
#     }, model_path)
#
#
#     #Test Set Eval
#     model.eval()
#     eval_train_correct = 0
#     eval_train_total = 0
#
#     preds = []
#     truths = []
#     samples_names = []
#     families = []
#     with torch.no_grad():
#         for inputs, labels, names, family in tqdm(test_loader):
#
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             outputs, penultimate_activ, conv_active  = model(inputs)
#
#             _, predicted = torch.max(outputs.data, 1)
#
#             preds.extend(F.softmax(outputs, dim=-1).data[:,1].detach().cpu().numpy().ravel())
#             truths.extend(labels.detach().cpu().numpy().ravel())
#             samples_names.extend(names)
#             families.extend(family)
#
#             eval_train_total += labels.size(0)
#             eval_train_correct += (predicted == labels).sum().item()
#
#     epoch_stats['test_acc'] = eval_train_correct*1.0/eval_train_total
#     epoch_stats['test_auc'] = roc_auc_score(truths, preds)
#
#     df_results = pd.DataFrame({
#         "pred": preds,
#         "truth": truths,
#         "sha": samples_names,
#         "family": families})
#     df_results.to_excel(f"predictions_{family_target}.xlsx", index=False)
#
#     csv_log_out.write(",".join([str(epoch_stats[h]) for h in headers]) + "\n")
#     csv_log_out.flush()
#
# csv_log_out.close()
    
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

model_path = os.path.join(base_name, family_target + ".checkpoint")

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

print("\n================ TESTING FINAL MODEL ================\n")

model.eval()

preds = []
truths = []
samples_names = []
families = []
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels, names, family in tqdm(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        outputs, penultimate_activ, conv_active = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        preds.extend(F.softmax(outputs, dim=-1).data[:,1].cpu().numpy().ravel())
        truths.extend(labels.cpu().numpy().ravel())
        samples_names.extend(names)
        families.extend(family)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = correct / total
test_auc = roc_auc_score(truths, preds)

print(f"\nFINAL TEST RESULTS")
print(f"Accuracy: {test_acc:.4f}")
print(f"AUC:      {test_auc:.4f}")

# Save predictions
df_results = pd.DataFrame({
    "pred": preds,
    "truth": truths,
    "sha": samples_names,
    "family": families
})

predictions_path = os.path.join(base_name, f"predictions_{family_target}.xlsx")
df_results.to_excel(predictions_path, index=False)


print("Predictions saved.")