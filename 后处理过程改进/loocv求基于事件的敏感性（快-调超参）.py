from 划分好测试集和训练集 import extract_test_dataset
from model_eegnet_scores_chb import get_information
# from oral_model_scores_chb import get_information
from tqdm import tqdm
import csv
import time

if __name__ == '__main__':
    seizures_numbers = {'patient1': 7, 'patient2': 3, 'patient3': 7, 'patient5': 5, 'patient7': 3, 'patient9': 4, 'patient10': 7, 'patient13': 12, 'patient14': 8, 'patient16': 8,
                        'patient17': 3, 'patient18': 6, 'patient19': 3, 'patient20': 8, 'patient21': 4, 'patient23': 7, 'patient4': 4, 'patient6': 10, 'patient8': 5, 'patient11': 3,
                        'patient15': 20, 'patient22': 3}
    records = []
    pre_information = get_information()
    # 对应的权重
    weight1 = 0.5
    weight2 = 1 - weight1
    for patient in [2,3,6,7,9,10,14 ]:  # 1, 2, 3, 5, 7, 9, 10, 13, 14, 16, 17, 18, 19, 20, 21, 23, 6, 8, 11, 22
        loop_n = seizures_numbers['patient%d' % patient]
        get_n = []
        error_n = []
        for select_loop_n in range(1, loop_n + 1):
            train_dataset, test_dataset = extract_test_dataset(patient_w=patient, select_loop_n_w=select_loop_n)
            true_label = []
            interictal_time_numbers = 0
            for X, y in tqdm(test_dataset):
                tensor_sum = y.sum()
                true_label.append(tensor_sum.item())
                if tensor_sum < 12:
                    if all(element == 0 for element in y):
                        interictal_time_numbers += 12
                    else:
                        interictal_time_numbers += 12 - tensor_sum.item()
            interictal_time = interictal_time_numbers / (12 * 60)
            model_scores = pre_information['patient%d-valid%d' % (patient, select_loop_n)]
            score_index = 0
            count_error_get = 0
            count_each_get = 0
            while score_index < len(model_scores):
                if model_scores[score_index] >= 10:
                    if true_label[score_index + 1] == 0:
                        print('没有找到发作')
                        score_index += 21
                        count_error_get += 1
                    else:
                        print('successful!')
                        count_each_get += 1
                        break
                score_index += 1
            get_n.append(count_each_get)
            error_n.append(count_error_get / interictal_time)
        print('---------------------patient%d----------------------------------' % patient)
        records.append(['patient%d ' % patient, get_n, error_n])
        print(get_n, error_n)
        print('---------------------patient%d----------------------------------' % patient)
        # break
    current_time = time.localtime()
    month = current_time.tm_mon
    day = current_time.tm_mday
    # 写入CSV文件
    model_name = 'eeginception+se'
    with open('patient%02dweight%f_%s_%02d-%02d.csv' % (patient, weight1, model_name, month, day), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(records)
