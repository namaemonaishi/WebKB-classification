import re
import matplotlib.pyplot as plt
import numpy as np

# 创建颜色列表，用于为每个 hist_train_ls 指定不同的颜色
colors = ['r', 'g', 'b', 'c', 'y', 'k']

# 创建文件路径和标签的对应关系
label_mapping = {
    "1-1.txt": "all-train acc",
    "1-2.txt": "conell-train acc",
    "1-3.txt": "misc-train acc",
    "1-4.txt": "texas-train acc",
    "1-5.txt": "washington-train acc",
    "1-6.txt": "wisconsin-train acc"
}

# 从多个文件中读取数据并绘制图像
file_paths = ["1-1.txt", "1-2.txt", "1-3.txt", "1-4.txt", "1-5.txt", "1-6.txt"]  # 文件路径列表

# 绘制横坐标为1-10的图像
fig1, ax1 = plt.subplots(dpi=200)

for file_path, color in zip(file_paths, colors):
    # 从文件中读取数据并存储到 hist_train_ls 变量中
    with open(file_path, 'r') as f:
        data_str = f.read()

    # 清除非法字符或缺失值
    data_str = re.sub(r'[^\d.,]', '', data_str)

    # 从清洗后的字符串中加载数据
    hist_train_ls = np.fromstring(data_str, dtype=float, sep=',')
    
    print(len(hist_train_ls))  # 检查数据长度

    # 获取对应的标签
    label = label_mapping[file_path]

    # 绘制 hist_train_ls，并指定颜色和标签
    ax1.plot(
        np.arange(1, len(hist_train_ls) + 1), hist_train_ls,
        color=color, label=label
    )

ax1.set_xlabel('epoch')
ax1.set_ylabel('accuracy')
ax1.set_xlim(1, 10)
ax1.legend()

# 保存图像
save_path1 = "train_accuracy_1-10.png"
fig1.savefig(save_path1)

# 绘制横坐标为5-10的图像
fig2, ax2 = plt.subplots(dpi=200)

for file_path, color in zip(file_paths, colors):
    # 从文件中读取数据并存储到 hist_train_ls 变量中
    with open(file_path, 'r') as f:
        data_str = f.read()

    # 清除非法字符或缺失值
    data_str = re.sub(r'[^\d.,]', '', data_str)

    # 从清洗后的字符串中加载数据
    hist_train_ls = np.fromstring(data_str, dtype=float, sep=',')
    hist_train_ls = hist_train_ls[4:]  # 仅保留第5项及之后的数据

    print(len(hist_train_ls))  # 检查数据长度

    # 获取对应的标签
    label = label_mapping[file_path]

    # 绘制 hist_train_ls，并指定颜色和标签
    ax2.plot(
        np.arange(5, len(hist_train_ls) + 5), hist_train_ls,
        color=color, label=label
    )

ax2.set_xlabel('epoch')
ax2.set_ylabel('accuracy')
ax2.set_xlim(5, 10)
ax2.legend()

# 保存图像
save_path2 = "train_accuracy_5-10.png"
fig2.savefig(save_path2)

# 显示图像
plt.show()