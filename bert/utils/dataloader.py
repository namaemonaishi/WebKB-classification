import torch
import numpy as np


# 'department' will be classified as 'course' in GCN
label2id = {
    'student': 0,
    'faculty': 1,
    'project': 2,
    'course': 3,
    'staff': 4,
    'department': 3,
}

id2label = {
    0: 'student',
    1: 'faculty', 
    2: 'project',
    3: 'course',
    4: 'staff',
}



def load_data(dataset_path, 
    uni_lt=['cornell', 'texas', 'wisconsin', 'washington', 'misc'], 
    cat_lt=['student', 'faculty', 'project', 'course', 'staff', 'department']):

    """Load the pre-processed data.
    Make sure all the ids of label in cat_ltare defined in preprocessor.id2label
    dictionary.
    
    Args:
        dataset_path: String of path to the pre-processed dataset,
            which is like './xxx/dataset.tsv'
        uni_lt: List of string containing the university names to load.
            Default as ['cornell', 'texas', 'wisconsin', 'washington', 'other']
        cat_lt: List of string containing the category names to load.
            Default as ['student', 'faculty', 'project', 'course', 'staff', 'department']
    Returns:
        A tuple of two list (texts, labels).
        texts is a list of string containing the clean text for tokenization.
        labels is a list of single integer list in shape (x, 1).
    """

    texts = []
    labels = []
    with open(dataset_path, 'r',encoding='utf-8') as data_file:
        for data_line in data_file:
            uni_name, cat_name, text, url = data_line.strip('\n').split('\t')
            if (uni_name in uni_lt) and (cat_name in cat_lt):
                texts.append(text)
                labels.append(label2id[cat_name])
    return texts, labels



def statistics(train_labels=None, val_labels=None, test_labels=None):
    """Compute the statistics info of the datasets.
    
    Args:
        train_labels: (Optional) List of training labels.
        val_labels: (Optional) List of validation labels.
        test_labels=None: (Optional) List of testing labels.
    Returns:
        String of the statistics info (for logging).
    """
    names = ['train set', 'val set', 'test set']
    tot_siz = 0
    s_log = ''
    train_val_test_labels = [train_labels, val_labels, test_labels]
    for set_name, label_ids in zip(names, train_val_test_labels):
        if label_ids is not None:
            tot_siz += len(label_ids)
            s_log += f'[{set_name:>9}] '
            for label_id, label_name in id2label.items():
                label_siz = (np.array(label_ids) == label_id).sum()
                s_log += f'{label_name}: {label_siz:4} | '
            s_log += f'total: {len(label_ids)}\n'
    s_log += f'total size = {tot_siz}\n'
    return s_log



class WebKBDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



if __name__ == '__main__':
    uni_lt = ['wisconsin']
    cat_lt = ['student', 'faculty', 'project', 'course', 'staff', 'department']
    split_id = 0
    print('uni_lt: ', uni_lt)
    train_texts, train_labels = load_split_data(
        './dataset.tsv', './dataset_split', split_id, 'train',
        uni_lt=uni_lt, cat_lt=cat_lt)
    val_texts, val_labels = load_split_data(
        './dataset.tsv', './dataset_split', split_id, 'val',
        uni_lt=uni_lt, cat_lt=cat_lt)
    test_texts, test_labels = load_split_data(
        './dataset.tsv', './dataset_split', split_id, 'test',
        uni_lt=uni_lt, cat_lt=cat_lt)


    # Print statistics of the splitting.
    s_stats = statistics(train_labels, val_labels, test_labels)
    print(s_stats)
