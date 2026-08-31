# 改进的混沌决策分析：先识别混沌状态，再在小窗内检测异常预测，触发两级预警并做事件敏感性评估。
# 合并后：分析完成后自动处理 CSV，计算各患者成功率与误报均值，并追加 overall_avg 行。

from 划分好测试集和训练集 import extract_test_dataset
from model_scores.model_deep4_scores_chb import get_information

from tqdm import tqdm
import ast
import csv
import time
import numpy as np
from pathlib import Path

# ---------- 可调超参数 ----------
CHAOS_WINDOW_LEN = 5          # 混沌检测滑动窗口长度（步数）
CHAOS_STD_THRESHOLD = 3       # 窗口内预测值标准差阈值，超过则判为混沌
SMALL_WINDOW_LEN = 10         # 混沌后检测异常的小窗长度（步数）
SKIP_AFTER_FALSE_ALARM = SMALL_WINDOW_LEN   # 误报后跳过的步数

# 患者与发作次数
SEIZURES_NUMBERS = {
    'patient1': 7,  'patient2': 3,  'patient3': 7, 'patient4': 4, 'patient5': 5, 'patient6': 10,  'patient7': 3, 'patient8': 5,
    'patient9': 4,  'patient10': 7,  'patient11': 3, 'patient13': 12,'patient14': 8, 'patient15': 20, 'patient16': 8,'patient17': 3,
    'patient18': 6, 'patient19': 3, 'patient20': 8, 'patient21': 4,'patient22': 3,'patient23': 7,
}
THRESHOLD_NUMBERS = {
    'patient1': 7, 'patient2': 10, 'patient3': 10, 'patient4': 11, 'patient5': 11, 'patient6': 11, 'patient7': 11,'patient8': 11,
    'patient9': 10, 'patient10': 11, 'patient11': 11, 'patient13': 7, 'patient14': 11, 'patient15': 11, 'patient16': 11,'patient17': 7,
    'patient18': 11, 'patient19': 11, 'patient20': 7, 'patient21': 11, 'patient22': 11, 'patient23': 7,
    # 'patient1': 7,  'patient2': 10, 'patient3': 10, 'patient5': 11, 'patient7': 11,
    # 'patient9': 10, 'patient10': 11,'patient13': 11,'patient14': 11,'patient16': 11,
    # 'patient17': 11,'patient18': 11,'patient19': 11,'patient20': 7,'patient21': 11,
    # 'patient23': 7,'patient4': 11, 'patient6': 11, 'patient8': 11, 'patient11': 11,
    # 'patient15': 11,'patient22': 11,
}
PATIENT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
MODEL_NAME = "cms_deep4"  # deep4 ATCNet BIOT EEGConformer SPaRCNet ContraWR


# ==================== 混沌分析核心函数 ====================

def detect_first_chaos_index(scores, window_len, std_threshold):
    """
    滑动窗口识别混沌状态，返回首次超过阈值时的起始索引，无混沌则返回 None。
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    if n < window_len:
        return None
    for i in range(n - window_len + 1):
        w = scores[i: i + window_len]
        if np.mean(w) >= 10 or np.std(w) > std_threshold:
            return i
    return None


def collect_final_alert_indices(scores, first_chaos_idx, small_window_len, score_threshold):
    """
    从第一次混沌开始，按不重叠小窗检查；
    若某窗内存在预测值 >= score_threshold，在窗结束位置触发最终预警。
    返回所有最终预警时刻的索引列表。
    """
    scores = np.asarray(scores, dtype=float)
    n = len(scores)
    alerts = []
    start = first_chaos_idx
    while start + small_window_len <= n:
        w = scores[start: start + small_window_len]
        if np.max(w) >= score_threshold:
            alerts.append(start + small_window_len - 1)
        start += small_window_len
    # 处理末尾不足 small_window_len 的剩余部分
    if start < n:
        w = scores[start: n]
        if np.max(w) >= score_threshold:
            alerts.append(n - 1)
    return alerts


def evaluate_one_seizure(model_scores, true_label, interictal_time, score_threshold):
    """
    对当前发作条带进行混沌检测 + 小窗异常检测，返回 (count_each_get, error_n)。
    """
    first_chaos = detect_first_chaos_index(model_scores, CHAOS_WINDOW_LEN, CHAOS_STD_THRESHOLD)
    if first_chaos is None:
        return 0, 0.0

    alert_indices = collect_final_alert_indices(
        model_scores, first_chaos, SMALL_WINDOW_LEN, score_threshold
    )
    if not alert_indices:
        return 0, 0.0

    count_error_get = 0
    count_each_get = 0
    idx = 0
    while idx < len(alert_indices):
        pos = alert_indices[idx]
        if pos + 1 > len(true_label):
            idx += 1
            continue
        if true_label[pos] == 0:
            count_error_get += 1
            idx += 1
            while idx < len(alert_indices) and alert_indices[idx] <= pos + SKIP_AFTER_FALSE_ALARM:
                idx += 1
        else:
            count_each_get = 1
            break

    error_n = count_error_get / interictal_time if interictal_time > 0 else 0.0
    return count_each_get, error_n


# ==================== CSV 后处理函数 ====================

def _compute_success_rate(values):
    """计算列表中 1 的占比。"""
    if not values:
        return ""
    return sum(1 for v in values if v == 1) / len(values)


def _compute_mean(values):
    """计算列表均值。"""
    if not values:
        return ""
    return sum(values) / len(values)


def process_csv(input_path: Path, output_path: Path) -> int:
    """
    读取原始结果 CSV（patient, get_n列表, error_n列表），
    计算每患者成功率与误报均值，追加 overall_avg 行后写入新文件。
    返回写入行数（含表头与汇总行）。
    """
    rows = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 3:
                continue
            patient = row[0].strip()
            get_n_array   = ast.literal_eval(row[1].strip())
            error_n_array = ast.literal_eval(row[2].strip())
            rows.append([
                patient,
                _compute_success_rate(get_n_array),
                _compute_mean(error_n_array),
            ])

    avg_success = _compute_mean([r[1] for r in rows if r[1] != ""])
    avg_error   = _compute_mean([r[2] for r in rows if r[2] != ""])
    rows.append(["overall_avg", avg_success, avg_error])

    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient", "success_rate", "third_col_mean"])
        writer.writerows(rows)

    return len(rows) + 1  # +1 为表头行


# ==================== 主流程 ====================

if __name__ == '__main__':
    records = []
    pre_information = get_information()

    for patient in PATIENT_LIST:
        # patient = 23
        loop_n = SEIZURES_NUMBERS['patient%d' % patient]
        score_threshold = THRESHOLD_NUMBERS['patient%d' % patient]
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
                model_scores, true_label, interictal_time, score_threshold
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

    # ---------- 写入原始结果 CSV ----------
    current_time = time.localtime()
    month = current_time.tm_mon
    day   = current_time.tm_mday
    raw_csv_path = Path(
        'cross_patients_weight0.500000_%s_%02d-%02d.csv' % (MODEL_NAME, month, day)
    )
    with raw_csv_path.open('w', encoding="utf-8-sig", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(records)
    print(f"\n原始结果已写入: {raw_csv_path}")

    # ---------- 自动后处理：计算成功率与误报均值 ----------
    processed_csv_path = raw_csv_path.with_name(raw_csv_path.stem + "_processed.csv")
    row_count = process_csv(raw_csv_path, processed_csv_path)
    print(f"后处理完成，共 {row_count} 行（含表头与汇总）。")
    print(f"汇总结果已写入: {processed_csv_path}")