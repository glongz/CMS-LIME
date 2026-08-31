# 改进的混沌决策分析：先识别混沌状态，再在小窗内检测异常预测，触发两级预警并做事件敏感性评估。

from 划分好测试集和训练集 import extract_test_dataset
# from model_eegnet_scores_chb import get_information
# from oral_model_scores_chb import get_information
from tqdm import tqdm
import csv
import time
import numpy as np

# ---------- 可调超参数 ----------
CHAOS_WINDOW_LEN = 5          # 混沌检测滑动窗口长度（步数）
CHAOS_STD_THRESHOLD = 3     # 窗口内预测值标准差阈值，超过则判为混沌
SMALL_WINDOW_LEN = 10          # 混沌后检测异常的小窗长度（步数）

SKIP_AFTER_FALSE_ALARM = SMALL_WINDOW_LEN   # 误报后跳过的步数（与参考一致）

# 患者与发作次数（与参考一致）
SEIZURES_NUMBERS = {
    'patient1': 7, 'patient2': 3, 'patient3': 7, 'patient5': 5, 'patient7': 3,
    'patient9': 4, 'patient10': 7, 'patient13': 12, 'patient14': 8, 'patient16': 8,
    'patient17': 3, 'patient18': 6, 'patient19': 3, 'patient20': 8, 'patient21': 4,
    'patient23': 7, 'patient4': 4, 'patient6': 10, 'patient8': 5, 'patient11': 3,
    'patient15': 20, 'patient22': 3,
}
THRESHOLD_NUMBERS = {
    'patient1': 7, 'patient2': 10, 'patient3': 10, 'patient5': 11, 'patient7': 11,
    'patient9': 10, 'patient10': 11, 'patient13': 11, 'patient14': 11, 'patient16': 11,
    'patient17': 11, 'patient18': 11, 'patient19': 11, 'patient20': 11, 'patient21': 11,
    'patient23': 11, 'patient4': 11, 'patient6': 11, 'patient8': 11, 'patient11': 11,
    'patient15': 11, 'patient22': 11,
}
PATIENT_LIST = [16, 18, 19, 22]  # 2, 3, 6, 7, 9, 10, 14, 16, 18, 19, 22
MODEL_NAME = 'chaos_decision'


def detect_first_chaos_index(scores, window_len, std_threshold):
    """
    对预测序列做滑动窗口，用标准差识别混沌状态，返回首次超过阈值时的起始索引。
    scores: list 或 array，预测值时间序列
    window_len: 窗口长度（步数）
    std_threshold: 窗口内标准差超过此值判为混沌
    返回: 首个混沌窗口的起始索引，若无混沌则返回 None
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < window_len:
        return None
    for i in range(n - window_len + 1):
        w = scores[i : i + window_len]
        if np.mean(w) >= 10 or np.std(w) > std_threshold:
            return i
    return None


def collect_final_alert_indices(scores, first_chaos_idx, small_window_len, score_threshold):
    """
    从第一次混沌开始，按不重叠小窗检查；若某窗内存在预测值 > score_threshold，在该窗结束位置触发最终预警。
    返回: 所有“最终预警”时刻的索引列表（每个为小窗的最后一个索引）。
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    alerts = []
    start = first_chaos_idx
    while start + small_window_len <= n:
        w = scores[start: start + small_window_len]
        if np.max(w) >= score_threshold:
            # 窗结束时刻触发
            alerts.append(start + small_window_len - 1)
        start += small_window_len

    # 检查剩余的部分，如果有不足small_window_len的窗口，单独作为一个新窗口处理
    if start < n:
        w = scores[start: n]  # 取剩余的部分
        if np.max(w) >= score_threshold:
            # 窗结束时刻触发
            alerts.append(n - 1)

    return alerts


def evaluate_one_seizure(model_scores, true_label, interictal_time):
    """
    对当前发作条带：先混沌检测，再小窗异常检测，得到最终预警序列；
    按与参考一致的事件评估：首次预警若对应发作则 get=1，否则误报+1 并跳过 21 步，直至命中或遍历完。
    返回: (count_each_get: 0 或 1, error_n: 误报次数/interictal_time)
    """
    first_chaos = detect_first_chaos_index(
        model_scores, CHAOS_WINDOW_LEN, CHAOS_STD_THRESHOLD
    )
    if first_chaos is None:
        return 0, 0.0

    alert_indices = collect_final_alert_indices(
        model_scores, first_chaos, SMALL_WINDOW_LEN, SCORE_THRESHOLD
    )
    if not alert_indices:
        return 0, 0.0

    count_error_get = 0
    count_each_get = 0
    idx = 0
    while idx < len(alert_indices):
        pos = alert_indices[idx]
        # 与参考一致：用 true_label[pos+1] 判断是否命中发作
        if pos + 1 > len(true_label):
            idx += 1
            continue
        if true_label[pos] == 0:
            # 误报：计数后跳过 21 步内的后续预警
            count_error_get += 1
            idx += 1
            while idx < len(alert_indices) and alert_indices[idx] <= pos + SKIP_AFTER_FALSE_ALARM:
                idx += 1
        else:
            count_each_get = 1
            break
    error_n = count_error_get / interictal_time if interictal_time > 0 else 0.0
    return count_each_get, error_n


if __name__ == '__main__':
    records = []
    pre_information = get_information()

    for patient in PATIENT_LIST:
        loop_n = SEIZURES_NUMBERS['patient%d' % patient]
        SCORE_THRESHOLD = THRESHOLD_NUMBERS['patient%d' % patient]  # 小窗内触发最终预警的预测值阈值
        get_n = []
        error_n = []
        for select_loop_n in range(1, loop_n + 1):
            train_dataset, test_dataset = extract_test_dataset(
                patient_w=patient, select_loop_n_w=select_loop_n
            )
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

            count_each_get, err_n = evaluate_one_seizure(
                model_scores, true_label, interictal_time
            )
            get_n.append(count_each_get)
            error_n.append(err_n)
            if count_each_get:
                print('successful!')
            elif err_n > 0:
                print('没有找到发作')

        print('---------------------patient%d----------------------------------' % patient)
        records.append(['patient%d ' % patient, get_n, error_n])
        print(get_n, error_n)
        print('---------------------patient%d----------------------------------' % patient)

    current_time = time.localtime()
    month = current_time.tm_mon
    day = current_time.tm_mday
    with open(
        'patient%02d_chaos_%s_%02d-%02d.csv' % (patient, MODEL_NAME, month, day),
        'w',
        newline='',
    ) as file:
        writer = csv.writer(file)
        writer.writerows(records)
