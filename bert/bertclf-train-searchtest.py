from decimal import MAX_EMAX
import torch
import numpy as np
from utils.dataloader import load_data, statistics
from utils.dataloader import WebKBDataset
from utils.dataloader import id2label
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, dataloader
from bertclf import BertClassifier
from bertclf import train_loop, test_loop
from utils.vis import draw_hist_loss, draw_hist_acc, draw_confusion_matrix
import argparse
from itertools import product
import os, time, sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor*")


# Parse hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
parser.add_argument('--uni_lt', type=str, nargs='+')
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--batch_siz', type=int, default=16)
parser.add_argument('--lr', type=float, default=5e-6)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--random_seed', type=int, default=2021)
parser.add_argument('--split_id', type=int)
args = parser.parse_args()
# Set random seed and device.
RANDOM_SEED = args.random_seed
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED) # Sklearn uses numpy's random seed.
# 如果不设置随机种子，那么每次运行代码时，随机数生成器将会使用系统当前时间作为种子来生成随机数。这会导致每次运行代码时，生成的随机数序列都不同，从而导致模型训练的结果不稳定，即使在相同的数据集和模型设置下，每次运行得到的结果也会略有不同。

# 为了解决这个问题，我们可以设置随机种子，以确保每次运行代码时，生成的随机数序列都是相同的。这样可以使结果更加稳定，方便我们进行实验和比较不同模型的性能。

# 在上述代码中，torch.manual_seed(RANDOM_SEED) 和 np.random.seed(RANDOM_SEED) 分别设置了 PyTorch 和 NumPy 中的随机种子。这意味着，每次运行代码时，PyTorch 和 NumPy 会使用相同的随机种子来生成随机数序列，从而确保模型训练的结果是可重复的。由于 scikit-learn 库使用 NumPy 的随机数生成器，因此在设置 NumPy 的随机种子后，scikit-learn 库的随机数生成器也会使用相同的随机种子。
device = 'cuda' 
# if torch.cuda.is_available() else 'cpu'
print('Use device:', device)

max_len_list = [128, 256,192]
batch_size_list = [4,8,12]
num_epochs_list = [10,15,20]
learning_rate_list = [1e-5, 5e-6, 5e-5,1e-6]
# Generate all possible combinations of hyperparameters.
hyperparams = list(product(max_len_list, batch_size_list, num_epochs_list, learning_rate_list))
np.random.shuffle(hyperparams)  # shuffle the list to randomize the search order.
# Load data.
cat_lt = ['student', 'faculty', 'project', 'course']
label_lt = list(id2label.values())[:4]
print('uni_lt: ', args.uni_lt)
texts, labels = load_data(
    'dataset.tsv', uni_lt=args.uni_lt, cat_lt=cat_lt)
# Split Train-val-test set.
train_ratio, val_ratio= 0.8, 0.2
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=val_ratio, stratify=labels)
# Print statistics of the splitting.

# Tokenization
tokenizer = AutoTokenizer.from_pretrained(args.bert_name)
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")


# Create pytorch dataset.
train_dataset = WebKBDataset(train_encodings, train_labels)
val_dataset = WebKBDataset(val_encodings, val_labels)

# Deploy model.
model = BertClassifier(args.bert_name)
model.to(device)
# 方法会将模型的参数和缓冲区移动到指定的设备上，因此模型的内容实际上不会发生变化，但是模型现在将在所指定的设备上进行计算。这意味着，如果您在移动模型之前在 CPU 上运行模型，则在移动模型之后，您需要在所指定的设备上重新运行模型才能得到输出。
# Initialize the loss functon and the optimizer.
loss_fn = torch.nn.CrossEntropyLoss()
hist_train_ls, hist_val_ls = [],[]
hist_train_acc, hist_val_acc = [], []
mx_train_acc, mx_val_acc,mx_max_len,mx_batch_size,mx_num_epochs,mx_learning_rate= 0, 0,0,0,0,0
for max_len, batch_size, num_epochs, learning_rate in hyperparams:
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # Training.
    train_loader = DataLoader(train_dataset, batch_size=args.batch_siz, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_siz, shuffle=True)
    # DataLoader 返回的是一个可迭代的对象，它可以产生一个个批次（batch）的数据。在每个迭代中，它会返回一个长度为 2 的元组 (inputs, targets)，其中 inputs 是一个张量或一个字典，存储了当前批次中的输入数据，targets 是一个张量或者一个列表，存储了当前批次中对应的标签。
    #是的，train_loader 返回的是一个可迭代的对象，它会产生一个个形如 (inputs, targets) 的元组。
    # 具体地说，inputs 的形状通常是 (batch_size, ...), 其中 batch_size 表示当前批次的大小，... 表示当前批次中每个样本的维度。而 targets 的形状通常是 (batch_size, )，其中 batch_size 表示当前批次的大小，每个元素代表了对应输入数据的标签。
    log_interval = len(train_loader) // 10
    # Train loops.
    for epoch_id in range(num_epochs):
        train_acc, train_ls, val_acc, val_ls = train_loop(
            train_loader, model, loss_fn, optim, 
            val_loader=val_loader, epoch_id=epoch_id,
            log_interval=log_interval
        )
        print(f"max_len: {max_len}, batch_size: {batch_size}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")
        print(f" train_acc={train_acc:.4f}, train_loss={train_ls:.4f}, val_acc={val_acc:.4f}, val_loss={val_ls:.4f}")
        if val_acc > mx_val_acc :
            mx_val_acc = val_acc 
            mx_learning_rate=learning_rate
            mx_batch_size=batch_size
            mx_max_len=max_len
            mx_num_epochs=num_epochs
            # 是的，当使用逗号将多个赋值表达式连接在一起时，Python 会将其解释为元组的创建或解包操作。如果你在赋值操作中使用逗号分隔多个变量，Python 会尝试将右侧的值解包到左侧的变量中。
            # 解包操作的语法在 C 中通常不存在，而在 C++ 中可以使用结构化绑定来实现。
            # 在 C++ 和python中，使用逗号分隔的赋值语法是不正确的。C++ 的赋值操作符（=）被用于将右侧的值赋给左侧的变量，而不支持同时对多个变量进行赋值。
        print(f"till now fmax_len: {mx_max_len}, fbatch_size: {mx_batch_size}, fnum_epochs: {mx_num_epochs}, flearning_rate: {mx_learning_rate}")
        with open('result.txt', 'w') as file:
            file.write(f"fmax_len: {mx_max_len}, fbatch_size: {mx_batch_size}, fnum_epochs: {mx_num_epochs}, flearning_rate: {mx_learning_rate}")
print(f"fmax_len: {mx_max_len}, fbatch_size: {mx_batch_size}, fnum_epochs: {mx_num_epochs}, flearning_rate: {mx_learning_rate}")
with open('fresult.txt', 'w') as file:
    file.write(f"fmax_len: {mx_max_len}, fbatch_size: {mx_batch_size}, fnum_epochs: {mx_num_epochs}, flearning_rate: {mx_learning_rate}")


del model
