from torch.utils.data import Dataset
import numpy as np
import os
import json
from torch.utils.data import DataLoader
import re
from collections import defaultdict
import math


class Mydataset(Dataset):
    def __init__(self):
        # assert flag in ['train', 'test', 'valid']
        # self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.data, self.n_pre, self.n_inter = self.__load_data__()


    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return self.data[index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    # def preprocess(self, data):
    #     # 将data 做一些预处理
    #     pass
    @staticmethod
    def __load_data__():
        # 假如从 csv_paths 中加载数据，可能要遍历文件夹读取文件等，这里忽略
        # 可以拆分训练和验证集并返回train_X, train_Y, valid_X, valid_Y
        pre_data_info = list()
        inter_data_info = list()
        inter_folder_path = r'D:\public_data\CHBMIT\1_data_clean_18channels\chb%02d' % patient # 前期信号段文件夹路径
        inter_segment_clean = r'D:\public_data\CHBMIT\segment_clean_18channels\30-1-240\chb%02d\segment_info.json' % patient
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
                    if len(inter_data_info) // 719 < 9:
                        split_data = eeg_data[:, eeg_start + j * 1280:eeg_start + (j + 1) * 1280]
                        j += 1
                        n_slide_sample = eeg_start + (j + 1) * 1280
                        # inter_data_info.append((dict_file["Label"], split_data, label, 20))
                        if split_data.shape[0] == 0:
                            print('空数据')
                            continue
                        if np.var(split_data) == 0:
                            print('方差为0')
                            continue
                        inter_data_info.append((dict_file["Label"], split_data, label))
                    else:
                        break
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
                    split_data = eeg_data[:, eeg_start + j * 640:eeg_start + j * 640 + 1280]
                    if split_data.shape[0] == 0:
                        print('空数据')
                        continue
                    if np.var(split_data) == 0:
                        print('方差为0')
                    j += 1
                    n_slide_sample = eeg_start + j * 640 + 1280
                    seizure = int(dict_file["Label"][-1])
                    # pre_data_info.append((dict_file["Label"], split_data, label, seizure))
                    pre_data_info.append((dict_file["Label"], split_data, label))


        n_inter = len(inter_data_info)
        n_pre = len(pre_data_info)
        data_info = inter_data_info + pre_data_info
        # print(len(data_info) // 719, n_pre, n_inter)
        return data_info, n_pre, n_inter



class loocv_split_dataset(Dataset):
    """自定义数据集，构建于字典基础上"""

    def __init__(self, data_dict, select_loop_n):
        # assert flag in ['train', 'test', 'valid']
        # self.flag = flag
        # 也可以把数据作为一个参数传递给类，__init__(self, data)；
        # self.data = data
        self.data_dict = data_dict
        self.select_loop_n = select_loop_n
        self.data = self.__split_data__(self.data_dict,self.select_loop_n)

    def __getitem__(self, index):
        # 根据索引返回数据
        # data = self.preprocess(self.data[index]) # 如果需要预处理数据的话
        return self.data[index]

    def __len__(self):
        # 返回数据的长度
        return len(self.data)

    @staticmethod
    def __split_data__(data_dict,select_n):
        train_data_info = []
        valid_data_info = []

        # 分类：预处理Pre和inter类分别处理
        grouped_items = {
            'inter': [],
            'Pre': [],
        }

        # Step 1: 分类并收集
        for key, val in data_dict.items():
            if key.startswith('inter'):
                grouped_items['inter'].append((key, val))
            elif key.startswith('Pre'):
                grouped_items['Pre'].append((key, val))

        # Step 2: 排序函数：提取编号
        def extract_number(s):
            match = re.search(r'\d+', s)
            return int(match.group()) if match else -1

        # Step 3: 对每组进行编号排序和划分
        for tag, items in grouped_items.items():
            items.sort(key=lambda x: extract_number(x[0]))
            total_keys = len(items)
            split_index = math.ceil(total_keys / 3)

            # 训练用前 split_index 个 key 的全部数据，其余为验证
            loop = 0
            for i, (key, data_list) in enumerate(items):
                if i < split_index:
                    train_data_info.extend(data_list)
                    # print(f"[{tag} train] {key}: {len(data_list)} samples")
                else:
                    loop += 1
                    if loop == select_n:
                        valid_data_info.extend(data_list)
                        print(f"[{tag} valid] {key}: {len(data_list)} samples")

        return train_data_info, valid_data_info


# -------------------------设置参数 ----------------------------------
global patient
# global select_loop_n


def extract_test_dataset(patient_w,select_loop_n):
    global patient
    patient = patient_w
    batch_size = 128
    test_batch_size = 12
    print('patient%d' % patient)
    # -----------------构建MyDataset实例 --------------------------------------
    train_data = Mydataset()
    label_number_0 = train_data.n_inter
    label_number_1 = train_data.n_pre
    print('label_1 %d '% train_data.n_pre)
    print('label_0 %d '% train_data.n_inter)
    classified = defaultdict(list)
    for item in train_data:
        # 提取字符串前缀，忽略后面可能的数字和短横线
        prefix = item[0].split('-')[0]
        if 'Pre' in prefix:
            classified[prefix].append(item[1:])
        else:
            classified['inter'].append(item[1:])
    loop_n = len(classified) - 1
    select_valid_n = loop_n - 1
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
        if patient == 19:
            key = f'inter{i + 2}'
        else:
            key = f'inter{i + 1}'
        # Assign the sublist to the current key in the split dictionary
        split_dict[key] = inter_list[start_index:end_index]
    classified.update(split_dict)
    del classified['inter']
    train_dataset, test_dataset = loocv_split_dataset(classified,select_loop_n)
    # 构建DataLoder
    train_iter = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = DataLoader(dataset=test_dataset, batch_size=test_batch_size, drop_last=True)
    return train_iter, test_iter
