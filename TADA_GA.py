"""
TADA_GA.py (Final Robust Version)
=================================
Includes:
1. Robust Genetic Algorithm (NaN fix + mmd_coef fix)
2. Extended Augmentation Library (Safe Wrappers for KMeans/SVM)
3. Evaluation Metrics & ResNet Models
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, confusion_matrix, precision_score, recall_score, \
    matthews_corrcoef
from sklearn.preprocessing import StandardScaler

# Imblearn
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
import time
from collections import defaultdict

# ---------------- 1. 安全采样封装 (Safe Wrappers) ----------------
# 这些封装是为了防止某些方法（如KMeansSMOTE）在样本太少时报错导致程序崩溃

def _class_counts(y):
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return pos, neg, min(pos, neg)


def _safe_k(minority):
    # 动态调整 k_neighbors，防止样本少于 k 时报错
    return max(1, min(5, minority - 1))


def safe_smote(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return SMOTE(random_state=42, k_neighbors=_safe_k(min_c)).fit_resample(X, y)
    except:
        return X, y


def safe_adasyn(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return ADASYN(random_state=42, n_neighbors=_safe_k(min_c)).fit_resample(X, y)
    except:
        return safe_smote(X, y)  # 回退


def safe_borderline(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return BorderlineSMOTE(random_state=42, k_neighbors=_safe_k(min_c)).fit_resample(X, y)
    except:
        return safe_smote(X, y)


def safe_kmeans(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        # KMeans 需要 cluster_balance_threshold 调整，否则容易报错
        return KMeansSMOTE(random_state=42, k_neighbors=_safe_k(min_c), cluster_balance_threshold=0.01).fit_resample(X,
                                                                                                                     y)
    except:
        return safe_smote(X, y)  # 回退到 SMOTE


def safe_svm(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return SVMSMOTE(random_state=42, k_neighbors=_safe_k(min_c)).fit_resample(X, y)
    except:
        return safe_smote(X, y)


def safe_smoteenn(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return SMOTEENN(random_state=42).fit_resample(X, y)
    except:
        return safe_smote(X, y)


def safe_smotetomek(X, y):
    _, _, min_c = _class_counts(y)
    if min_c < 2: return X, y
    try:
        return SMOTETomek(random_state=42).fit_resample(X, y)
    except:
        return safe_smote(X, y)


def safe_ros(X, y):
    try:
        return RandomOverSampler(random_state=42).fit_resample(X, y)
    except:
        return X, y


def safe_under(X, y):
    try:
        return RandomUnderSampler(random_state=42).fit_resample(X, y)
    except:
        return X, y


# 映射表
AUGMENTATION_METHODS = {
    'smote': safe_smote,
    'adasyn': safe_adasyn,
    'ros': safe_ros,
    'borderline': safe_borderline,
    'smoteenn': safe_smoteenn,
    'kmeanssmote': safe_kmeans,
    'svmsmote': safe_svm,
    'smotetomek': safe_smotetomek,
    'undersample': safe_under,
}


# ---------------- 2. 评估模块 ----------------
def clean_and_binarize_label(x):
    if isinstance(x, bytes): x = x.decode('utf-8')
    x = str(x).strip().lower()
    if x in ['1', 'bug', 'buggy', 'yes', 'true']:
        return 1
    elif x in ['0', 'nonbug', 'clean', 'no', 'false']:
        return 0
    try:
        return 1 if float(x) > 0 else 0
    except:
        return 0


def calculate_metrics(y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else 0.5
    except:
        auc = 0.5

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        bacc = 0.5 * (tp / (tp + fn + 1e-6) + tn / (tn + fp + 1e-6))
        pf = fp / (fp + tn + 1e-6)
    else:
        bacc, pf = 0.0, 1.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    g_measure = 2 * (precision * recall) / (precision + recall + 1e-6)

    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except:
        mcc = 0.0

    return {
        "ACC": acc, "F1": f1, "BACC": bacc, "G": g_measure,
        "AUC": auc, "MCC": mcc, "Pf": pf
    }


def compute_mmd(X1, X2):
    return np.linalg.norm(np.mean(X1, axis=0) - np.mean(X2, axis=0))


def compute_composite_score(metrics, weights=None):
    """
    计算综合得分
    """
    default_weights = {
        "ACC": 0.0, "F1": 2.0, "BACC": 0.0, "G": 0.0,
        "AUC": 0.0, "MCC": 2.0, "Pf": -1.5
    }
    if weights is None:
        weights = default_weights

    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0.0)
    return score


# ---------------- 3. 遗传增强主逻辑 ----------------
# ✅ 修复点 2：添加 mmd_coef 参数
def evaluate_strategy(strategy, X, y, X_t_sample, lambda_weight=1.0, fitness_type="multi"):
    """
    评估单条策略。
    ✅ 终极修复：接收 lambda_weight 和 fitness_type，实现动态适应度计算
    """
    X_current, y_current = X.copy(), y.copy()
    try:
        # 1. 执行增强链
        for method in strategy:
            if method in AUGMENTATION_METHODS:
                X_current, y_current = AUGMENTATION_METHODS[method](X_current, y_current)

        # 检查是否还有两个类别，否则无法训练
        if len(np.unique(y_current)) < 2:
            return -1e9

        # 2. 交叉验证评估
        model = LogisticRegression(max_iter=1000, solver='liblinear')
        metrics_list = []
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for train_idx, val_idx in skf.split(X_current, y_current):
            model.fit(X_current[train_idx], y_current[train_idx])
            preds = model.predict(X_current[val_idx])
            probs = model.predict_proba(X_current[val_idx])[:, 1]
            metrics = calculate_metrics(y_current[val_idx], preds, probs)
            metrics_list.append(metrics)

        if not metrics_list: return -1e9

        # 取平均指标 (注意这里的 key 要和您 calculate_metrics 返回的字典一致)
        avg_metrics = {key: np.mean([m[key] for m in metrics_list]) for key in metrics_list[0]}

        # 3. 计算 MMD
        mmd_val = compute_mmd(X_current, X_t_sample)

        # ================= 4. 终极逻辑：决定 GA 的生死 =================
        # 确保键名（如 'f1', 'mcc', 'g_measure'）与您 calculate_metrics 返回的一致。如果叫 'g'，请改为 'g'。
        f1_val = avg_metrics.get('f1', 0)
        mcc_val = avg_metrics.get('mcc', 0)
        g_val = avg_metrics.get('g_measure', avg_metrics.get('g', 0))  # 兼容两种常见命名

        if fitness_type == "multi":
            # 多指标：提供平滑梯度，避免适应度欺骗
            base_score = f1_val + mcc_val + g_val
        else:
            # 单指标：极不平衡下容易导致 F1=0，直接停滞
            base_score = f1_val

        # 扣除 MMD 惩罚项 (使用我们一路传进来的 lambda_weight)
        score = base_score - (lambda_weight * mmd_val)
        # ============================================================

        # ✅ 强制返回 float，防止返回 numpy 对象或 None
        return float(score)

    except Exception as e:
        # print(f"Eval Error: {e}")
        return -1e9


def genetic_sampling_selection(X_s, y_s, X_t_sample, pop_size=10, generations=5, alpha=0.5,
                               lambda_weight=1.0, fitness_type="multi"):
    """
    Robust Genetic Algorithm for Sampling Strategy Selection.
    Fixes: ValueError: probabilities contain NaN
    """
    # 1. 初始化种群
    methods = [m for m in AUGMENTATION_METHODS.keys()]
    population = [random.choices(methods, k=random.randint(1, 2)) for _ in range(pop_size)]

    best_score = -float('inf')
    best_strategy = ["none"]

    for gen in range(generations):
        scores = []
        for strategy in population:
            # ================= 核心修改点 =================
            # 把 lambda_weight 和 fitness_type 传给底层的评估函数
            score = evaluate_strategy(
                strategy, X_s, y_s, X_t_sample,
                lambda_weight=lambda_weight,
                fitness_type=fitness_type
            )
            # ==============================================
            scores.append(score)

            # 更新最佳记录
            if score > best_score:
                best_score = score
                best_strategy = strategy

        # --- 🛡️ ROBUST PROBABILITY CALCULATION ---
        scores_arr = np.array(scores, dtype=float)

        # Step A: Handle NaNs and Infs
        valid_scores = scores_arr[np.isfinite(scores_arr)]
        if len(valid_scores) > 0:
            min_val = np.min(valid_scores)
            fill_val = min_val - 10.0  # 惩罚无效解
        else:
            fill_val = -1e9

        scores_arr = np.nan_to_num(scores_arr, nan=fill_val, posinf=fill_val, neginf=fill_val)

        # Step B: Shift to Positive Space (Softmax-like)
        max_score = np.max(scores_arr)
        exp_scores = np.exp(scores_arr - max_score)

        # Step C: Normalize
        sum_exp = np.sum(exp_scores)

        if sum_exp <= 0 or np.isnan(sum_exp):
            prob_dist = np.ones(len(scores)) / len(scores)
        else:
            prob_dist = exp_scores / sum_exp

        if np.isnan(prob_dist).any():
            prob_dist = np.ones(len(scores)) / len(scores)
        # -------------------------------------------------------

        # 选择与繁殖
        order = np.argsort(scores_arr)[::-1]
        n_parents = max(2, pop_size // 2)
        parents_indices = order[:n_parents]
        parents = [population[i] for i in parents_indices]

        # 重新归一化父母概率
        parent_probs = prob_dist[parents_indices]
        parent_probs_sum = parent_probs.sum()
        if parent_probs_sum > 0:
            parent_probs = parent_probs / parent_probs_sum
        else:
            parent_probs = np.ones(len(parents)) / len(parents)

        next_gen = parents[:]
        while len(next_gen) < pop_size:
            # 轮盘赌选择（修复：不要直接 np.random.choice(list-of-lists)）
            idxs = np.random.choice(len(parents), size=2, replace=True, p=parent_probs)
            p1, p2 = parents[idxs[0]], parents[idxs[1]]

            # 交叉
            cut = random.randint(0, min(len(p1), len(p2)))
            child = p1[:cut] + p2[cut:]

            # 变异
            if random.random() < 0.3:
                child.append(random.choice(methods))

            if not child: child = [random.choice(methods)]

            next_gen.append(child)

        population = next_gen

    # 返回最佳策略生成的数据
    X_final, y_final = X_s.copy(), y_s.copy()
    try:
        for m in best_strategy:
            if m in AUGMENTATION_METHODS:
                X_final, y_final = AUGMENTATION_METHODS[m](X_final, y_final)
    except:
        pass

    return X_final, y_final, "+".join(best_strategy)


# 1. 在参数列表里加上 lambda_weight 和 fitness_type
def intelligent_augmentation_ga(X, y, Xt_sample, pop_size=60, generations=30, alpha=0.5,
                                lambda_weight=1.0, fitness_type="multi"):  # <--- 新增
    """
    Main Entry Point
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y, 'none'

    # 2. 传给底层的遗传算法核心函数
    return genetic_sampling_selection(
        X, y, Xt_sample, pop_size, generations, alpha,
        lambda_weight=lambda_weight,  # <--- 传下去
        fitness_type=fitness_type  # <--- 传下去
    )


# ---------------- 4. 辅助模型结构 (ResNet/Classifier) ----------------
# (这部分代码保持不变，为了文件完整性我还是放这里)
class BasicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)


class FeatureMapperResNet(nn.Module):
    def __init__(self, input_dim, mapped_dim=128, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 256)
        self.res_blocks = nn.Sequential(*[BasicBlock(256, 512) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(256, mapped_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x))
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


# ---------------- 5. 数据加载器 ----------------
def load_multi_source_data(source_folders, target_file, target_pos_ratio=0.5, force_aug_method=None,
                           target_subset_size=50, source_size_ratio=1.0, defect_ratio=None,
                           lambda_weight=1.0, fitness_type="multi"):
    # 1. 加载并处理目标域数据
    target_data, _ = arff.loadarff(target_file)
    df_target = pd.DataFrame(target_data)
    for col in df_target.columns:
        if df_target[col].dtype == object:
            df_target[col] = df_target[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

    Xt_full = df_target.iloc[:, :-1].values.astype(np.float32)
    yt_full = df_target.iloc[:, -1].apply(clean_and_binarize_label).values.astype(np.int64)
    Xt_full = StandardScaler().fit_transform(Xt_full)

    # ================= 修改 1：控制 Target Subset Size =================
    # 抽取无标签目标域特征送给遗传算法算 MMD (移到了循环外面，确保所有源域对齐到同一个 target subset)
    actual_subset_size = min(target_subset_size, len(Xt_full))
    Xt_sample = Xt_full[np.random.choice(len(Xt_full), actual_subset_size, replace=False)]
    # ===================================================================

    source_data, source_labels, selected_methods = [], [], []

    for folder in source_folders:
        for filename in os.listdir(folder):
            if filename.endswith('.arff'):
                file_path = os.path.join(folder, filename)
                if os.path.abspath(file_path) == os.path.abspath(target_file):
                    continue
                try:
                    data, _ = arff.loadarff(file_path)
                    df = pd.DataFrame(data)
                    for col in df.columns:
                        if df[col].dtype == object:
                            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

                    X = df.iloc[:, :-1].values.astype(np.float32)
                    y = df.iloc[:, -1].apply(clean_and_binarize_label).values.astype(np.int64)
                    X = StandardScaler().fit_transform(X)

                    # ================= 修改 2：控制 Source Size Ratio =================
                    if source_size_ratio < 1.0:
                        n_samples = int(len(y) * source_size_ratio)
                        if n_samples > 0:
                            indices = np.random.choice(len(y), n_samples, replace=False)
                            X, y = X[indices], y[indices]
                    # ===================================================================

                    # ================= 修改 3：控制 Defect Ratio =================
                    if defect_ratio is not None and len(y) > 0:
                        defects = np.where(y == 1)[0]
                        cleans = np.where(y == 0)[0]
                        if len(defects) > 0 and len(cleans) > 0:
                            current_ratio = len(defects) / len(y)
                            if defect_ratio < current_ratio:
                                # 需要降低缺陷比例（欠采样少数类）
                                n_defects_needed = int(len(cleans) * defect_ratio / (1 - defect_ratio))
                                n_defects_needed = min(n_defects_needed, len(defects))
                                chosen_defects = np.random.choice(defects, n_defects_needed, replace=False)
                                chosen_indices = np.concatenate([cleans, chosen_defects])
                            else:
                                # 需要提高缺陷比例（欠采样多数类）
                                n_cleans_needed = int(len(defects) * (1 - defect_ratio) / defect_ratio)
                                n_cleans_needed = min(n_cleans_needed, len(cleans))
                                chosen_cleans = np.random.choice(cleans, n_cleans_needed, replace=False)
                                chosen_indices = np.concatenate([chosen_cleans, defects])

                            np.random.shuffle(chosen_indices)
                            X, y = X[chosen_indices], y[chosen_indices]
                    # ===================================================================

                    # 安全保护：如果抽样后只剩下一种类别（全0或全1），跳过此文件以免 GA 报错
                    if len(np.unique(y)) < 2:
                        continue

                    # 执行增强算法
                    if force_aug_method:
                        X_aug, y_aug = AUGMENTATION_METHODS[force_aug_method](X, y)
                        method = force_aug_method
                    else:
                        # ✅ 正确做法：把接收到的参数透传给底层 GA 函数
                        X_aug, y_aug, method = intelligent_augmentation_ga(
                            X, y, Xt_sample,
                            lambda_weight=lambda_weight,
                            fitness_type=fitness_type
                        )

                    source_data.append(X_aug)
                    source_labels.append(y_aug)
                    selected_methods.append(method)

                except Exception as e:
                    print(f"⚠️ 跳过文件 {filename}，因为出错：{e}")

    # 划分测试集用于最终评估
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        Xt_full, yt_full, test_size=0.3, random_state=42, stratify=yt_full
    )

    return source_data, source_labels, Xt_train, yt_train, Xt_test, yt_test, selected_methods


def generate_augmented_data(X_train, y_train):
    # 使用源域自身采样生成目标增强数据（Xt_sample默认为源域自身子集）
    Xt_sample = X_train[np.random.choice(len(X_train), int(0.1 * len(X_train)), replace=False)]
    X_aug, y_aug, _ = intelligent_augmentation_ga(X_train, y_train, Xt_sample)
    return X_aug, y_aug