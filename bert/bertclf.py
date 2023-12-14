import torch
from torch import nn
# torch.nn是PyTorch中的一个模块，提供了用于构建神经网络的类和函数，是PyTorch中非常重要的一个模块。torch.nn模块中包含了各种层（例如全连接层、卷积层、循环神经网络层等）、损失函数、优化器和各种常用的网络组件等，这些组件可以被组合在一起构建出各种复杂的神经网络。
from transformers import AutoModel
from sklearn.metrics import accuracy_score, confusion_matrix


class BertClassifier(nn.Module):
    def __init__(self, bert_name, num_label=5):
        super(BertClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_name)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_label)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask):
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask)
        features = bert_outputs.last_hidden_state[:, 0]
        logits = self.linear(features)
        preds = self.softmax(logits)
        return preds
#定义一个深度学习的过程


def train_loop(train_loader, model, loss_fn, optim,
    val_loader=None, log_interval=5, epoch_id=None):
    """One train loop for one epoch.

    Args:
        train_loader: Pytorch dataloader for training.
        model: A BertForSequentialClassification model.
        loss_fn: Loss function.
        optim: Optimizer.
        val_loader: Pytorch dataloader for validation.
        log_interval: Integer. Zero for no logging.
        epoch_id: Optional integer shown in log string.
    Returns:
        A tuple of (train acc, train loss, val acc, val loss),
        where val_acc, val_ls are optional if val_loader is given.
    """

    step_loss = 0
    tot_train_ls, tot_train_acc = 0, 0

    # Training
    model.train()
    # 将模型设置为训练模式
    # 在深度学习中，模型除了训练模式之外，还有评估模式和推理模式。

# 训练模式（Train Mode）：在训练模式下，模型会自动跟踪梯度信息，以便进行反向传播算法的计算和模型参数的更新。此外，在训练模式下，一些特定的操作，如Dropout和Batch Normalization等，会被启用。

# 评估模式（Eval Mode）：在评估模式下，模型参数不会被更新，而是用于评估模型的性能。在评估模式下，Dropout和Batch Normalization等特定的操作会被禁用或被替换为其他操作。此外，评估模式下还可能会进行一些针对模型精度和速度的优化。model.eval

# 推理模式（Inference Mode）：在推理模式下，模型用于进行实际的预测或推理任务。在推理模式下，模型不会跟踪梯度信息，也不会进行任何参数更新操作。同时，为了提高模型的推理速度和效率，推理模式下可能会对输入数据进行批处理或缓存处理等优化操作。
    for batch_id, batch in enumerate(train_loader):#batch like (inputs, targets)
        optim.zero_grad()
        # optim.zero_grad()就是用来将模型参数的梯度清零的。它应该在每一个batch训练之前调用，以确保梯度信息被正确地计算
        # 反向传播算法是一种高效的计算梯度的方法，它利用了链式法则将损失函数对模型参数的梯度表示为各个网络层输出对该参数的梯度的乘积
        # 反向传播算法首先通过前向传播算法计算模型的输出，然后计算损失函数对输出的梯度，再使用链式法则计算损失函数对每个模型参数的梯度。
        # 计算损失函数对输出的梯度，就是计算损失函数对模型输出中每个元素的偏导数。这个梯度向量告诉我们，当模型输出发生微小变化时，损失函数会相应地发生多大的变化。
        # 优化器通常会根据梯度的大小和方向，计算出一个更新量 $\Delta \theta_i$，然后将其加到原来的参数值 $\theta_i$ 上，得到新的参数值
        # 常见的优化器有 SGD（随机梯度下降）、Adam、Adagrad 等。不同的优化器在计算更新量时采用不同的策略，例如使用动量、自适应学习率等方法，从而可以更加高效地更新参数值。
        # Onto device (GPU).
        device = next(model.parameters()).device
        # 这里是获取了目前模型在gpu上,所以后面把所有的数据都统一弄到gpu上 next是一个迭代器,为了获取parameter的第一个参数
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
#         接下来的三行代码将输入数据转移到 GPU 上，并设置为模型的指定设备。其中，input_ids 是模型的输入数据，attention_mask 是用于 mask 掉 padding 的标记，labels 是模型的输出标签。to() 方法可以将数据从 CPU 转移到 GPU 上，这里使用 device 变量指定了 GPU 设备。这样做的目的是，将数据和模型都放在 GPU 上，能够加速模型的计算和优化，从而提高模型的训练效率。
        # initial_params = {}
        # for name, param in model.named_parameters():
        #     initial_params[name] = param.clone().detach()
# 总之，这几行代码的作用是将数据从 CPU 转移到 GPU 上，并将其设置为模型的指定设备，以加速模型的训练。
        # Forward.
        preds = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(preds, labels)#计算损失函数
        # BP and update.
        loss.backward()
        optim.step()
        # for name, param in model.named_parameters():
        #     if not torch.equal(initial_params[name], param):
        #         print(name)

        # Log the batch.
        tot_train_ls += loss.item()
        tot_train_acc += accuracy_score(
            labels.cpu().flatten(), preds.argmax(axis=1).cpu().flatten())
        step_loss += loss.item()
        if log_interval and (batch_id+1) % log_interval == 0:
            s_epoch = f'Epoch {epoch_id:3} ' if epoch_id is not None else ''
            print('[{}Batch {:>3}/{:3}] train_loss={:.4f}'.format(
                s_epoch, batch_id+1, len(train_loader), 
                step_loss / log_interval
                ))   
            step_loss = 0

    # Log the training epoch.
    avg_train_ls = tot_train_ls / len(train_loader)
    avg_train_acc= tot_train_acc / len(train_loader)

    # Validation
    if val_loader:
        val_acc, val_ls, _ = test_loop(val_loader, model, loss_fn, log_interval=0)
        return avg_train_acc, avg_train_ls, val_acc, val_ls
    else:
        return avg_train_acc, avg_train_ls
    # 并没有单独的代码行启动模型的训练。相反，模型的训练是通过对每个batch数据的遍历和反向传播来实现的。具体来说，这段代码通过遍历训练数据集 train_loader，并使用优化器 optimizer 对模型参数进行更新，从而实现了模型的训练。
    


def test_loop(test_loader, model, loss_fn, log_interval=1):
    """Test loop for validation and evalutaion.

    Args:
        test_loader: Pytorch dataloader.
        model: A BertForSequentialClassification model.
        log_interval: Integer. Zero for no logging on terminal.
    Returns:
        Tuple of (accuracy, loss, confusion matrix).
    """

    tot_test_ls, tot_test_acc = 0, 0
    all_preds, all_labels = [], []

    # Evaluation.
    model.eval()
    for batch_id, batch in enumerate(test_loader):
        with torch.no_grad():
            # Inference.
            device = next(model.parameters()).device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            preds = model(input_ids, attention_mask=attention_mask)
            # Save the batch result.
            all_preds.append(preds.argmax(axis=1))
            all_labels.append(labels)
            # Compute metrics of batch.
            batch_acc = accuracy_score(
                labels.cpu().flatten(), preds.argmax(dim=1).cpu().flatten())
            batch_ls = loss_fn(preds, labels).item()
            tot_test_acc += batch_acc
            tot_test_ls += batch_ls
            # Log on terminal.
            if log_interval and (batch_id+1) % log_interval == 0:
                print('[Batch {:>3}/{:3}] batch_acc={:.4f} batch_ls={:.6f}'.format(
                    batch_id+1, len(test_loader), batch_acc, batch_ls)) 

    test_acc = tot_test_acc / len(test_loader)
    test_ls = tot_test_ls / len(test_loader)
    all_preds = torch.hstack(all_preds).cpu()
    all_labels = torch.hstack(all_labels).cpu()
    conf_mat = confusion_matrix(all_labels.cpu(), all_preds.cpu())
    
    return test_acc, test_ls, conf_mat