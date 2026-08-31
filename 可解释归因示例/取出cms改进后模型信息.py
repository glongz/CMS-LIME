from 划分好测试集和训练集 import extract_test_dataset
# from biomarker_enhanced_prediction_v2 import extract_model_scores
from 取出深度学习模型的分数 import extract_model_scores
from tqdm import tqdm
import time

if __name__ == '__main__':
    seizures_numbers = {'patient1': 7, 'patient2': 3, 'patient3': 7, 'patient5': 5, 'patient7': 3, 'patient9': 4, 'patient10': 7, 'patient13': 12, 'patient14': 8, 'patient16': 8,
                        'patient17': 3, 'patient18': 6, 'patient19': 3, 'patient20': 8, 'patient21': 4, 'patient23': 7, 'patient4': 4, 'patient6': 10, 'patient8': 5, 'patient11': 3,
                        'patient15': 20, 'patient22': 3}
    results = []
    for patient in [16,18,19,22]:  #     2,3,6,7,9,10,14,16,18,19,22             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
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
            # ----------------------  1 --------------------------------
            # critical_information_scores = extract_critical_information_scores(patient_w=patient, test_dataset=test_dataset)
            # results.append({'patient%d-valid%d' % (patient, select_loop_n): critical_information_scores})
            # file_name = 'critical_information_scores_chb信息'
            # ----------------------  2 --------------------------------
            model_scores = extract_model_scores(patient_w=patient, select_loop_n_w=select_loop_n, test_dataset=test_dataset, model_name='deep4')
            results.append({'patient%d-valid%d' % (patient, select_loop_n): model_scores})

    file_name = 'model_eegnet_scores_chb_2'
    current_time = time.localtime()
    month = current_time.tm_mon
    day = current_time.tm_mday

    # 合并为单个字典
    merged_dict = {}
    for item in results:
        merged_dict.update(item)

    # 将列表格式化为多行（每行约 25 个元素，便于阅读）
    def format_list(lst, indent='    ', per_line=25):
        if not lst:
            return '[]'
        lines = []
        for i in range(0, len(lst), per_line):
            chunk = lst[i:i + per_line]
            lines.append(indent + ', '.join(str(x) for x in chunk) + ',')
        return '[\n' + '\n'.join(lines) + '\n    ]'

    # 写入 Python 文件，包含 get_information() 函数
    out_path = '%s.py' % file_name
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('def get_information():\n')
        f.write('    return {\n')
        for key, value in merged_dict.items():
            if isinstance(value, list):
                list_str = format_list(value, indent='        ', per_line=25)
                f.write("        '%s': %s,\n" % (key, list_str))
            else:
                f.write("        '%s': %s,\n" % (key, repr(value)))
        f.write('    }\n')
    print('已保存到: %s' % out_path)
