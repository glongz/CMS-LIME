from 划分好测试集和训练集 import extract_test_dataset
from model_scores.oral_ContraWR_scores_chb import get_information
from tqdm import tqdm
import csv
import time
from pathlib import Path


def compute_success_rate(values):
    if not values:
        return ""
    return sum(1 for v in values if v == 1) / len(values)


def compute_mean(values):
    if not values:
        return ""
    return sum(values) / len(values)


if __name__ == "__main__":
    seizures_numbers = {
        "patient1": 4, "patient2": 2, "patient3": 4, "patient4": 2, "patient5": 3,
        "patient6": 6, "patient7": 2, "patient8": 3, "patient9": 2, "patient10": 4,
        "patient11": 2, "patient13": 5, "patient14": 5, "patient15": 9, "patient16": 3,
        "patient17": 3, "patient18": 3, "patient19": 2, "patient20": 4, "patient21": 2,
        "patient22": 2, "patient23": 3,
    }

    weight1 = 0.5
    weight2 = 1 - weight1
    model_name = "ContraWR"  # deep4 ATCNet BIOT EEGConformer eegnet SPaRCNet ContraWR

    pre_information = get_information()
    records_raw = []  # [patient_str, get_n, error_n]

    for patient in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:  # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
        loop_n = seizures_numbers["patient%d" % patient]
        get_n = []
        error_n = []
        for select_loop_n in range(1, loop_n + 1):
            train_dataset, test_dataset = extract_test_dataset(
                patient_w=patient, select_loop_n=select_loop_n
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
            interictal_time = interictal_time_numbers / (12 * 60) + 0.00000001
            model_scores = pre_information["patient%d-valid%d" % (patient, select_loop_n)]
            score_index = 0
            count_error_get = 0
            count_each_get = 0
            while score_index < len(model_scores):
                if model_scores[score_index] >= 10:
                    if true_label[score_index] == 0:
                        print("没有找到发作")
                        score_index += 21
                        count_error_get += 1
                    else:
                        print("successful!")
                        count_each_get += 1
                        break
                score_index += 1
            get_n.append(count_each_get)
            error_n.append(count_error_get / interictal_time)

        print("---------------------patient%d----------------------------------" % patient)
        records_raw.append(["patient%d" % patient, get_n, error_n])
        print(get_n, error_n)
        print("---------------------patient%d----------------------------------" % patient)

    # 在内存中汇总为一张表
    rows = []
    for patient_str, get_n, error_n in records_raw:
        rows.append(
            [
                patient_str.strip(),
                compute_success_rate(get_n),
                compute_mean(error_n),
            ]
        )

    avg_success = compute_mean([r[1] for r in rows if r[1] != ""])
    avg_third = compute_mean([r[2] for r in rows if r[2] != ""])
    rows.append(["overall_avg", avg_success, avg_third])

    t = time.localtime()
    out_name = "表格/cross_patients_weight%f_%s_%02d-%02d.csv" % (
        weight1, model_name, t.tm_mon, t.tm_mday,
    )
    out_path = Path(__file__).resolve().parent / out_name
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["patient", "success_rate", "third_col_mean"])
        w.writerows(rows)

    print("Wrote:", out_path)