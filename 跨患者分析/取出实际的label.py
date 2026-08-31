from 划分好测试集和训练集 import extract_test_dataset
# from 取出重要片段层次聚类先验信息的分数2 import extract_critical_information_scores
# from 取出深度学习模型的分数 import extract_model_scores
from tqdm import tqdm
import time

if __name__ == '__main__':
    seizures_numbers = {'patient1': 4, 'patient2': 2, 'patient3': 4, 'patient4': 2, 'patient5': 3, 'patient6': 6 , 'patient7': 2, 'patient8': 3, 'patient9': 2, 'patient10': 4, 'patient11': 2, 'patient13': 5, 'patient14': 5,'patient15': 9, 'patient16': 3,
                        'patient17': 3, 'patient18': 3, 'patient19': 2, 'patient20': 4, 'patient21': 2,'patient22': 2,'patient23': 3}
    results = []
    for patient in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:  # 1, 2, 3, 5, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        loop_n = seizures_numbers['patient%d' % patient]
        get_n = []
        error_n = []
        for select_loop_n in range(1, loop_n + 1):
            train_dataset, test_dataset = extract_test_dataset(patient_w=patient,select_loop_n=select_loop_n)
            true_label = []
            interictal_time_numbers = 0
            for X, y in tqdm(test_dataset):
                tensor_sum = y.sum()
                true_label.append(tensor_sum.item())
                # if tensor_sum < 12:
                #     if all(element == 0 for element in y):
                #         interictal_time_numbers += 12
                #     else:
                #         interictal_time_numbers += 12 - tensor_sum.item()
            # interictal_time = interictal_time_numbers / (12 * 60)
            # ----------------------  1 --------------------------------
            # critical_information_scores = extract_critical_information_scores(patient_w=patient, test_dataset=test_dataset)
            # results.append({'patient%d-valid%d' % (patient, select_loop_n): critical_information_scores})
            # file_name = 'critical_information_scores_chb信息'
            # ----------------------  2 --------------------------------
            # model_scores = extract_model_scores(patient_w=patient, select_loop_n_w=select_loop_n, test_dataset=test_dataset)
            results.append({'patient%d-valid%d' % (patient, select_loop_n): true_label})

    file_name = 'true_label_chb信息'
    current_time = time.localtime()
    month = current_time.tm_mon
    day = current_time.tm_mday
    # 写入CSV文件
    with open('%s_%02d-%02d.txt' % (file_name, month, day), 'w', newline='') as file:
        for item in results:
            file.write(str(item) + '\n')
