import argparse
import numpy as np
from itertools import product
from bertclf import BertClassifier, train_loop
from utils.dataloader import load_data, statistics, WebKBDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Define hyperparameters to search over.
max_len_list = [128, 256, 512]
batch_size_list = [8, 16, 32]
num_epochs_list = [10, 50, 100]
learning_rate_list = [1e-5, 3e-5, 5e-5]

# Generate all possible combinations of hyperparameters.
hyperparams = list(product(max_len_list, batch_size_list, num_epochs_list, learning_rate_list))
np.random.shuffle(hyperparams)  # shuffle the list to randomize the search order.

# Load data and tokenizer.
train_data, test_data = load_data(...)
tokenizer = AutoTokenizer.from_pretrained(...)                

# Loop over all hyperparameter combinations.
for max_len, batch_size, num_epochs, learning_rate in hyperparams:
    # Create training and validation sets.
    train_set, val_set = train_test_split(train_data, test_size=0.2, random_state=2021)

    # Create dataloaders.
    train_loader = DataLoader(
        WebKBDataset(train_set, tokenizer, max_len=max_len),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        WebKBDataset(val_set, tokenizer, max_len=max_len),
        batch_size=batch_size,
        shuffle=False
    )

    # Create model, loss function, and optimizer.
    model = BertClassifier(...)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model.
    for epoch in range(num_epochs):
        train_acc, train_loss, val_acc, val_loss = train_loop(train_loader, model, loss_fn, optimizer, val_loader=val_loader)

        # Print results.
        print(f"max_len: {max_len}, batch_size: {batch_size}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")
        print(f"Epoch {epoch+1}: train_acc={train_acc:.4f}, train_loss={train_loss:.4f}, val_acc={val_acc:.4f}, val_loss={val_loss:.4f}")
        # 这段代码没有划分测试集是因为它的主要目的是进行超参数搜索和模型选择，而不是进行最终的测试。在进行超参数搜索和模型选择时，我们通常会将数据集划分成训练集和验证集，使用训练集来训练模型，并使用验证集来评估不同超参数组合下的模型性能。因此，这段代码只划分了训练集和验证集，而没有划分测试集。