import torch
from model.EEGInception_SE import EEGInception
# from braindecode.models import EEGNetv4
import numpy as np
import os
import re
from tqdm import tqdm


def evaluate_accuracy(data_windows, net, device_acc):
    pre_counts = 0
    with torch.no_grad():
        slide_i = 0
        while slide_i < data_windows.shape[0]:
            if isinstance(net, torch.nn.Module):
                X = data_windows[slide_i, :, :]
                slide_i += 1
                net.eval()  # 评估模式, 这会关闭dropout
                X = X.to(torch.float32)
                X = X.unsqueeze(0)
                if X.shape[-1] != 1280:
                    print(X.shape)
                    continue
                predict_cla = net(X.to(device_acc)).argmax(dim=1).cpu().numpy()
                num_of_ones = np.count_nonzero(predict_cla == 1)
                pre_counts += num_of_ones
        # print(data_windows.shape[-1])
        # print('\033[91m________________________________________\033[0m')
        return pre_counts


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def sort_key(filename):
    # This regular expression will find the numbers preceded by "FDR" and followed by "-" in the filename
    match = re.search(r'(\d+)-(\d+)', filename)
    if match:
        # Generate a tuple (first_number, second_number)
        return int(match.group(1)), int(match.group(2))
    else:
        return filename


def extract_model_scores(patient_w, select_loop_n_w, test_dataset, model_name):
    # model_name = 'eeginception'
    # model_name = 'eegnet'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_model_path = r'D:\public_data\CHBMIT\weight\组合留一法\%s' % model_name
    # weight_model_path = r'E:\public database\CHBMIT\weight\LOOCV\%s' % model_name
    weight_sub_dirs = os.listdir(weight_model_path)
    weight_sub_dirs = sorted(weight_sub_dirs, key=natural_keys)
    patient = patient_folder = patient_w
    # print("now is patient%d" % patient)
    if patient > 12:
        patient_folder = patient - 1
        print("now is patient%d" % patient)
    elif patient == 12:
        print('exclude 12,now is patient13')
        patient += 1
        patient_folder = patient - 1
    each_patient_model_path = os.path.join(weight_model_path, weight_sub_dirs[patient_folder - 1])
    weight_each_patient_sub_dirs = os.listdir(each_patient_model_path)
    # Sort filenames using the sort_key function
    weight_each_patient_sub_dirs = sorted(weight_each_patient_sub_dirs, key=sort_key)
    weight_each_patient_sub_dirs = list(filter(lambda x: x.endswith('.pth'), weight_each_patient_sub_dirs))
    n_seizure = select_loop_n_w - 1
    model_path = os.path.join(each_patient_model_path, weight_each_patient_sub_dirs[n_seizure])
    print('model_path is %s' % model_path)
    data_root_dir = r'D:\public_data\CHBMIT\1_data_clean\chb%02d' % patient
    # ________________________________________________________________________________________________________________________
    data_sub_dirs = os.listdir(data_root_dir)
    data_sub_dirs = list(filter(lambda x: x.endswith('.npy'), data_sub_dirs))
    # ___________________________为了找到通道数___________________________________________________________________________________
    example = data_sub_dirs[0]
    example_sub_patient_dirs = os.path.join(data_root_dir, example)
    example_data = np.load(example_sub_patient_dirs)
    example_data = np.squeeze(example_data)
    n_chans = example_data.shape[0]
    n_classes = 2
    model = EEGInception(input_time=1280, fs=256, ncha=n_chans, n_classes=n_classes)
    # model = EEGNetv4(n_chans, n_classes, final_conv_length='auto', input_window_samples=1280)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    # print('parameters:', sum(param.numel() for param in model.parameters() if param.requires_grad))
    counts = list()
    for X, y in tqdm(test_dataset):
        X = X.to(device)
        # X = X.unsqueeze(dim=1)
        # y = y.to(device)
        X = X.to(torch.float32)
        data_counts = evaluate_accuracy(X, net=model, device_acc=device)
        counts.append(data_counts)
    return counts
