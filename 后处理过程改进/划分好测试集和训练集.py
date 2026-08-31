from torch.utils.data import Dataset
import numpy as np
import os
import json
from torch.utils.data import DataLoader
import re
from collections import defaultdict
import random


class Mydataset(Dataset):
    def __init__(self):
        self.data = self.__load_data__()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    @staticmethod
    def __load_data__():
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        pre_data_info = list()
        inter_data_info = list()
        inter_folder_path = r'D:\public_data\CHBMIT\1_data_clean\chb%02d' % patient # 前期信号段文件夹路径
        inter_segment_clean = r'D:\public_data\CHBMIT\segment_clean\30-1-240\chb%02d\segment_info.json' % patient
        with open(inter_segment_clean) as f:
            files_list = json.load(f)
        label = 0
        for dict_file in files_list:
            result = re.match("Inter", dict_file["Label"])
            if result:
                # files.append(dict_file["Label"])
                eeg_data = np.load(os.path.join(inter_folder_path, dict_file['File']))
                eeg_data = np.squeeze(eeg_data)
                eeg_start = dict_file['Span'][0]
                eeg_end = dict_file['Span'][1]
                n_slide_sample = 0
                j = 0
                while n_slide_sample <= eeg_end:
                    split_data = eeg_data[:, eeg_start + j * 1280:eeg_start + (j + 1) * 1280]
                    if split_data.shape[0] == 0:
                        print('空数据')
                        continue
                    if np.var(split_data) == 0:
                        print('方差为0')
                    j += 1
                    n_slide_sample = eeg_start + (j + 1) * 1280
                    inter_data_info.append((dict_file["Label"], split_data, label))
        label = 1
        for dict_file in files_list:
            result = re.match("Pre", dict_file["Label"])
            if result:
                # files.append(dict_file["Label"])
                eeg_data = np.load(os.path.join(inter_folder_path, dict_file['File']))
                eeg_data = np.squeeze(eeg_data)
                eeg_start = dict_file['Span'][0]
                eeg_end = dict_file['Span'][1]
                n_slide_sample = 0
                j = 0
                while n_slide_sample <= eeg_end:
                    split_data = eeg_data[:, eeg_start + j * 1280:eeg_start + (j + 1) * 1280]
                    if split_data.shape[0] == 0:
                        print('空数据')
                        continue
                    if np.var(split_data) == 0:
                        print('方差为0')
                        continue
                    j += 1
                    n_slide_sample = eeg_start + (j + 1) * 1280
                    pre_data_info.append((dict_file["Label"], split_data, label))
        data_info = inter_data_info + pre_data_info
        # print(len(data_info) // 719, n_pre, n_inter)
        return data_info



class loocv_split_dataset(Dataset):
    """自定义数据集，构建于字典基础上"""

    def __init__(self, data_dict):
        # assert flag in ['train', 'test', 'valid']
        # self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.data_dict = data_dict
        self.data = self.__split_data__(self.data_dict)

    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return self.data[index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    @staticmethod
    def __split_data__(data_dict):
        train_data_info = []
        valid_data_info = []
        pre_save = ()
        inter_save = ()
        for category, tuples in data_dict.items():
            # if '%d' % current_n in category:
            if category == 'Pre%d' % select_loop_n:
                pre_save = tuples
            elif category == 'inter%d' % select_loop_n:
                inter_save = tuples
            else:
                train_data_info.extend(tuples)
        valid_data_info.extend(inter_save)
        valid_data_info.extend(pre_save)

        return train_data_info, valid_data_info


# -------------------------设置参数 ----------------------------------
global patient
global select_loop_n


def extract_test_dataset(patient_w, select_loop_n_w):
    global patient
    patient = patient_w
    # print('patient%d' % patient)
    batch_size = 128
    test_batch_size = 12
    # -----------------构建MyDataset实例 --------------------------------------
    train_data = Mydataset()
    classified = defaultdict(list)
    for item in train_data:
        # 提取字符串前缀，忽略后面可能的数字和短横线
        prefix = item[0].split('-')[0]
        if 'Pre' in prefix:
            classified[prefix].append(item[1:])
        else:
            classified['inter'].append(item[1:])
    loop_n = len(classified) - 1
    # Dictionary to hold the split lists
    split_dict = defaultdict(list)
    inter_list = classified['inter']
    # Calculate the size of each split
    split_size = len(inter_list) // loop_n if loop_n > 0 else len(inter_list)
    # Avoid having a split with 0 elements in case of more splits than list elements
    split_size = max(split_size, 1)
    for i in range(loop_n):
        # Calculate the start and end indices for the current split
        start_index = i * split_size
        # For the last split, extend to the end of the list to include any remaining elements
        end_index = None if i == loop_n - 1 else start_index + split_size
        # Create a key for each split
        key = f'inter{i + 1}'
        # Assign the sublist to the current key in the split dictionary
        split_dict[key] = inter_list[start_index:end_index]
    classified.update(split_dict)
    del classified['inter']
    global select_loop_n
    select_loop_n = select_loop_n_w
    print('now is %d of %d valid' % (select_loop_n, loop_n))
    train_dataset, test_dataset = loocv_split_dataset(classified)
    # 构建DataLoder
    train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=test_batch_size)
    return train_iter, test_iter
