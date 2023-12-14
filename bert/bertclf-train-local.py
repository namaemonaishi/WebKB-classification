import torch
import numpy as np
from utils.dataloader import load_data, statistics
from utils.dataloader import WebKBDataset
from utils.dataloader import id2label
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, dataloader
from bertclf import BertClassifier
# Transformer架构被广泛应用于预训练模型的设计中，其中BERT是其中的一种典型例子。在BERT模型中，通过自注意力机制和多层Transformer编码器，模型能够同时考虑输入序列中的上下文信息，从而学习到更丰富的语义表示。
# 预训练过程通常包括两个阶段：预训练和微调。在预训练阶段，模型使用大量无标签的文本数据进行自监督训练，通过预测掩码、下一句预测等任务来学习语言表示。在微调阶段，模型使用有标签的任务特定数据，在特定任务上进行微调和训练，以适应具体的任务要求。
# Hugging Face库中的一项重要功能是提供了用于加载、使用和训练各种预训练模型的API和工具。这些预训练模型包括BERT以及其他各种经典和最新的NLP模型，如GPT、RoBERTa、DistilBERT等等。
from bertclf import train_loop, test_loop
from utils.vis import draw_hist_loss, draw_hist_acc, draw_confusion_matrix
import argparse
import os, time, sys
from transformers import AutoTokenizer, AutoModelForMaskedLM
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor*")

# Parse hyperparameters.
parser = argparse.ArgumentParser()
parser.add_argument('--local_path', type=str, default='.')
parser.add_argument('--bert_name', type=str, default='bert-mini')
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
# 在机器学习中，我们通常需要比较不同模型的性能。通过固定随机种子，可以确保每次运行实验时使用相同的训练集和验证集划分，从而使模型之间的比较更加公平和可靠。
device = 'cuda' 
# if torch.cuda.is_available() else 'cpu'
print('Use device:', device)


# Load data.
cat_lt = ['student', 'faculty', 'project', 'course']
label_lt = list(id2label.values())[:4]
print('uni_lt: ', args.uni_lt)
texts, labels = load_data(
    'dataset.tsv', uni_lt=args.uni_lt, cat_lt=cat_lt)
# Split Train-val-test set.
train_ratio, val_ratio, test_ratio = 0.6, 0.2, 0.2
try:
    train_texts, valtest_texts, train_labels, valtest_labels = train_test_split(
        texts, labels, test_size=val_ratio+test_ratio, stratify=labels)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        valtest_texts, valtest_labels, test_size=test_ratio/(test_ratio+val_ratio),
        stratify=valtest_labels)
except:
    train_texts, valtest_texts, train_labels, valtest_labels = train_test_split(
        texts, labels, test_size=val_ratio+test_ratio)
    val_texts, test_texts, val_labels, test_labels = train_test_split(
        valtest_texts, valtest_labels, test_size=test_ratio/(test_ratio+val_ratio))
# Print statistics of the splitting.
s_stats = statistics(train_labels, val_labels, test_labels)
print(s_stats)


# Tokenization
tokenizer = AutoTokenizer.from_pretrained(args.local_path)
train_encodings = tokenizer(
    train_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")
val_encodings = tokenizer(
    val_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")
test_encodings = tokenizer(
    test_texts, truncation=True, padding=True, 
    max_length=args.max_len, return_tensors="pt")


# Create pytorch dataset.
train_dataset = WebKBDataset(train_encodings, train_labels)
val_dataset = WebKBDataset(val_encodings, val_labels)
test_dataset = WebKBDataset(test_encodings, test_labels)


# Deploy model.
model = BertClassifier(args.local_path)
model.to(device)
# 方法会将模型的参数和缓冲区移动到指定的设备上，因此模型的内容实际上不会发生变化，但是模型现在将在所指定的设备上进行计算。这意味着，如果您在移动模型之前在 CPU 上运行模型，则在移动模型之后，您需要在所指定的设备上重新运行模型才能得到输出。
# Initialize the loss functon and the optimizer.
loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=args.lr)


# Training.
hist_train_ls, hist_val_ls = [],[]
hist_train_acc, hist_val_acc = [], []
mx_train_acc, mx_val_acc = 0, 0
train_loader = DataLoader(train_dataset, batch_size=args.batch_siz, shuffle=True)#受numpy库影响
val_loader = DataLoader(val_dataset, batch_size=args.batch_siz, shuffle=True)
# DataLoader 返回的是一个可迭代的对象，它可以产生一个个批次（batch）的数据。在每个迭代中，它会返回一个长度为 2 的元组 (inputs, targets)，其中 inputs 是一个张量或一个字典，存储了当前批次中的输入数据，targets 是一个张量或者一个列表，存储了当前批次中对应的标签。
#是的，train_loader 返回的是一个可迭代的对象，它会产生一个个形如 (inputs, targets) 的元组。
# 具体地说，inputs 的形状通常是 (batch_size, ...), 其中 batch_size 表示当前批次的大小，... 表示当前批次中每个样本的维度。而 targets 的形状通常是 (batch_size, )，其中 batch_size 表示当前批次的大小，每个元素代表了对应输入数据的标签。
log_interval = len(train_loader) // 10
# Train loops.
for epoch_id in range(args.num_epoch):
    train_acc, train_ls, val_acc, val_ls = train_loop(
        train_loader, model, loss_fn, optim, 
        val_loader=val_loader, epoch_id=epoch_id,
        log_interval=log_interval
    )
    print(f'Epoch {epoch_id} finished with '
        f'train_acc={train_acc:4f}, val_acc={val_acc:4f}, '
        f'train_ls={train_ls:6f}, val_ls={val_ls:6f}')
    hist_train_acc.append(train_acc)
    hist_val_acc.append(val_acc)
    hist_train_ls.append(train_ls)
    hist_val_ls.append(val_ls)
    mx_train_acc = train_acc if train_acc > mx_train_acc else mx_train_acc
    mx_val_acc = val_acc if val_acc > mx_val_acc else mx_val_acc


# Evaluation.
test_loader = DataLoader(test_dataset, batch_size=args.batch_siz, shuffle=True)
test_acc, test_ls, test_conf_mat = test_loop(test_loader, model, loss_fn)


# Create a folder for saving.
bert_ver = args.bert_name
save_dir = '.\\bertclf-{}-save\\epoch{}\\'.format(
    bert_ver, args.num_epoch
)
fig_dir = save_dir + 'fig\\' 
model_dir = save_dir + 'model\\'
os.makedirs(save_dir)
os.makedirs(fig_dir)
os.makedirs(model_dir)
# Save the command line and dataset splitting statistical info
with open(save_dir + 'info.txt', 'w+') as file:
    file.write('Hyperparameter settings:\n')
    file.write(str(sys.argv[1:]))
    file.write('\n\nSplitting statistical info:\n')
    file.write(s_stats)
    file.write('\n\nEnd with\n')
    file.write(f'mx_train_acc={mx_train_acc:6f}, '
        f'mx_val_acc={mx_val_acc:6f}, test_acc={test_acc:6f}')
# Save the loss and acc history.
with open(save_dir + 'hist_train_ls.txt', 'w+') as file:
    file.write(str(hist_train_ls))
with open(save_dir + 'hist_val_ls.txt', 'w+') as file:
    file.write(str(hist_val_ls))
with open(save_dir + 'hist_train_acc.txt', 'w+') as file:
    file.write(str(hist_train_acc))
with open(save_dir + 'hist_val_acc.txt', 'w+') as file:
    file.write(str(hist_val_acc))
# Visualization
draw_hist_loss(hist_train_ls, hist_val_ls, 
    save_path=fig_dir + 'train_loss.jpg', linetype='-o')
draw_hist_acc(hist_train_acc, hist_val_acc, 
    save_path=fig_dir + 'train_acc.jpg', linetype='-o')
draw_confusion_matrix(test_conf_mat, label_lt,
    save_path=fig_dir + 'conf_mat.jpg')
# Save the model and hyperparamer setup.
torch.save(model, model_dir + 'model.pt')
tokenizer.save_pretrained(model_dir)
del model
