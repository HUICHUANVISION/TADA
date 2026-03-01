# ===========================================
# CPDP-HDP 实验主脚本（单源→目标）
# 依赖：你已有的 clean_and_binarize_label / calculate_metrics /
#       intelligent_augmentation_ga / FeatureMapperResNet / Classifier
# 步骤：
#   1) 源域采样增强（遗传搜索）
#   2) 两域映射器训练：源域监督 + 源/目标 MMD 对齐
#   3) 目标域评测
#   4) 导出：采样前后数据对比（类分布/特征变化/快照） & 指标明细+汇总
# ===========================================

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

# ------------------ 工具 ------------------
def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ======== TAGS 核心：采样方法 + 评分 + 遗传搜索 ========
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE
from imblearn.combine import SMOTEENN
# 可按需解注释更多方法
# from imblearn.over_sampling import KMeansSMOTE, SVMSMOTE
# from imblearn.combine import SMOTETomek
# from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np
import random

AUGMENTATION_METHODS = {
    "smote":       lambda X, y: SMOTE(random_state=42).fit_resample(X, y),
    "adasyn":      lambda X, y: ADASYN(random_state=42).fit_resample(X, y),
    "ros":         lambda X, y: RandomOverSampler(random_state=42).fit_resample(X, y),
    "borderline":  lambda X, y: BorderlineSMOTE(random_state=42).fit_resample(X, y),
    "smoteenn":    lambda X, y: SMOTEENN(random_state=42).fit_resample(X, y),
    # "kmeanssmote": lambda X, y: KMeansSMOTE(random_state=42).fit_resample(X, y),
    # "svmsmote":    lambda X, y: SVMSMOTE(random_state=42).fit_resample(X, y),
    # "smotetomek":  lambda X, y: SMOTETomek(random_state=42).fit_resample(X, y),
    # "undersample": lambda X, y: RandomUnderSampler(random_state=42).fit_resample(X, y),
}

def compute_mmd(X1, X2):
    """与你之前一致的均值差范式（L2 距离）。"""
    return np.linalg.norm(np.mean(X1, axis=0) - np.mean(X2, axis=0))

def _composite_score(metrics: dict, weights: dict | None = None) -> float:
    """
    复合评分：默认只用 F1、MCC（正权重）和 Pf（负权重）。
    可传入自定义权重覆盖。
    """
    default_weights = {"F1": 2.0, "MCC": 2.0, "Pf": -1.5,
                       "ACC": 0.0, "BACC": 0.0, "G": 0.0, "AUC": 0.0}
    w = weights or default_weights
    return sum(w[k] * metrics.get(k, 0.0) for k in w)

def evaluate_strategy(strategy, X, y, X_t_sample, score_weights=None, mmd_coef=0.5):
    """
    对一个增强调度序列进行评估：交叉验证得到平均指标 -> 复合分数 + MMD 正则。
    依赖你已有的 calculate_metrics(y_true, y_pred, y_score)。
    """
    Xc, yc = X.copy(), y.copy()
    try:
        # 顺序执行增强
        for m in strategy:
            Xc, yc = AUGMENTATION_METHODS[m](Xc, yc)

        # 3 折 CV 评估
        model = LogisticRegression(max_iter=1000)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        metrics_list = []
        for tr, va in skf.split(Xc, yc):
            model.fit(Xc[tr], yc[tr])
            pred = model.predict(Xc[va])
            prob = model.predict_proba(Xc[va])[:, 1]
            metrics_list.append(calculate_metrics(yc[va], pred, prob))

        avg = {k: np.mean([m[k] for m in metrics_list]) for k in metrics_list[0]}
        # 分布对齐：-MMD（越小越好 -> 加负号）
        mmd_term = -compute_mmd(Xc, X_t_sample)
        return _composite_score(avg, score_weights) + mmd_coef * mmd_term
    except Exception:
        return -np.inf  # 该策略失败则给极小值

def genetic_sampling_selection(X_s, y_s, X_t_sample,
                               pop_size=10, generations=5,
                               score_weights=None, mmd_coef=0.5):
    methods = list(AUGMENTATION_METHODS.keys())
    population = [random.choices(methods, k=random.randint(1, 3)) for _ in range(pop_size)]

    # 至少保证有 2 个父代
    n_parents = max(2, pop_size // 2)

    for _ in range(generations):
        scores = [evaluate_strategy(ind, X_s, y_s, X_t_sample, score_weights, mmd_coef)
                  for ind in population]

        # 将非有限值替换为极小值，避免 NaN 传播
        scores = [s if np.isfinite(s) else -1e15 for s in scores]

        order = np.argsort(scores)[::-1]
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]

        # 选父代
        parents = population[:n_parents]
        parent_scores = np.array(scores[:n_parents], dtype=float)

        # 如果父代分数全相等或全无效，使用均匀概率
        # 否则用 softmax（更稳健）
        if not np.isfinite(parent_scores).all():
            probs = np.ones(len(parents), dtype=float) / len(parents)
        else:
            z = parent_scores - np.max(parent_scores)  # 防止溢出
            expz = np.exp(z)
            s = np.sum(expz)
            if not np.isfinite(s) or s <= 0:
                probs = np.ones(len(parents), dtype=float) / len(parents)
            else:
                probs = expz / s

        # 万一仍有 NaN，做最终兜底
        if (np.isnan(probs).any()) or (probs.sum() <= 0):
            probs = np.ones(len(parents), dtype=float) / len(parents)

        # 交叉生成子代
        children = []
        while len(children) < pop_size - n_parents:
            # 父代数量可能只有 2，允许有放回抽样
            i1, i2 = np.random.choice(len(parents), size=2, replace=True, p=probs)
            p1, p2 = parents[i1], parents[i2]

            # 单点交叉
            min_len = min(len(p1), len(p2))
            if min_len <= 1:
                child = p1.copy()
            else:
                cut = random.randint(1, min_len - 1)
                child = p1[:cut] + p2[cut:]

            # 以 0.3 概率变异
            if random.random() < 0.3:
                mut = random.choice(methods)
                if mut not in child:
                    child.append(mut)

            # 避免空策略
            if len(child) == 0:
                child = [random.choice(methods)]
            children.append(child)

        population = parents + children

    # 返回最优策略及其应用后的数据
    best = population[0]
    Xf, yf = X_s, y_s
    for m in best:
        Xf, yf = AUGMENTATION_METHODS[m](Xf, yf)
    return Xf, yf, "+".join(best)

def intelligent_augmentation_ga(X, y, Xt_sample,
                                pop_size=60, generations=30,
                                alpha=0.5,  # 与你原先一致，作为 mmd 系数
                                score_weights=None):
    """
    外部调用接口：做一次智能增强。
    """
    # 若某类为空，直接返回
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return X, y, "none"
    return genetic_sampling_selection(
        X, y, Xt_sample, pop_size=pop_size, generations=generations,
        score_weights=score_weights, mmd_coef=alpha
    )
# ======== TAGS 核心结束 ========
def clean_and_binarize_label(x):
    """
    统一二分类标签到 {0,1}：
      - 正类(1): 'bug', 'buggy', 'yes', 'true', 'y', '1'
      - 负类(0): 'clean', 'nonbug', 'no', 'false', 'n', '0'
    兼容 bytes/str/int/float；其它值按 >0 -> 1，否则 0。
    """
    # 处理字节串
    if isinstance(x, bytes):
        try:
            x = x.decode("utf-8")
        except Exception:
            x = str(x)

    # 字符串归一化
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"bug", "buggy", "yes", "true", "y", "1"}:
            return 1
        if s in {"clean", "nonbug", "no", "false", "n", "0"}:
            return 0
        # 兜底：能转数字则看是否 > 0
        try:
            return 1 if float(s) > 0 else 0
        except Exception:
            # 未知字符串，保守按 0 处理（也可改为 raise）
            return 0

    # 数值类型
    if isinstance(x, (int, float)):
        return 1 if x > 0 else 0

    # 其他类型兜底
    return 0
def load_single_arff(path):
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df.iloc[:, -1].apply(clean_and_binarize_label).values.astype(np.int64)
    X = StandardScaler().fit_transform(X)
    return X, y

# ---------------- 源域采样增强 ----------------
def augment_source_once(Xs, ys, Xt, target_subsample_ratio=0.1):
    n_t = max(1, int(target_subsample_ratio * len(Xt)))
    idx_t = np.random.choice(len(Xt), n_t, replace=False)
    Xt_sample = Xt[idx_t]
    Xs_aug, ys_aug, aug_strategy = intelligent_augmentation_ga(
        Xs, ys, Xt_sample, pop_size=60, generations=30, alpha=0.5
    )
    return Xs_aug, ys_aug, aug_strategy

# ------------- 两域映射器 + 对齐训练 -------------
# ==== ResNet-style feature mappers + shared classifier ====
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    f1_score, accuracy_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score, matthews_corrcoef
)
import numpy as np

def calculate_metrics(y_true, y_pred, y_score):
    """
    统一返回论文里会用到的所有指标：
    ACC, F1, BACC, G, AUC, MCC, Pf
    - 全部做了除零与二分类判断保护
    """
    # 基础
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # AUC（需要两类都出现）
    try:
        if len(np.unique(y_true)) == 2:
            auc = roc_auc_score(y_true, y_score)
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        # BACC
        tpr = tp / (tp + fn + 1e-6)
        tnr = tn / (tn + fp + 1e-6)
        bacc = 0.5 * (tpr + tnr)
        # Pf (false alarm rate = FP / (FP + TN))
        pf = fp / (fp + tn + 1e-6)
    else:
        bacc, pf = 0.0, 1.0

    # 精确率/召回率 -> G-measure
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    g_measure = 2 * (precision * recall) / (precision + recall + 1e-6)

    # MCC
    try:
        mcc = matthews_corrcoef(y_true, y_pred)
    except Exception:
        mcc = 0.0

    return {
        "ACC": acc,
        "F1": f1,
        "BACC": bacc,
        "G": g_measure,
        "AUC": auc,
        "MCC": mcc,
        "Pf": pf
    }
class BasicBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out

class FeatureMapperResNet(nn.Module):
    """
    输入: 原始特征 (in_dim)
    输出: 共享潜空间表示 (mapped_dim)
    结构: 线性 -> 多个残差块(维度不变) -> 线性到 mapped_dim
    """
    def __init__(self, input_dim, mapped_dim=128, num_blocks=3):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, 256)
        self.res_blocks = nn.Sequential(*[BasicBlock(256, 512) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(256, mapped_dim)

    def forward(self, x):
        x = F.relu(self.input_layer(x), inplace=True)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.fc2(x)
        return x
# ==== end of ResNet mapper + classifier ====
class HDPModel(nn.Module):
    """ 源/目标各一个映射器 -> 共享潜空间；一个共享分类器（源域监督） """
    def __init__(self, in_dim_src, in_dim_tgt, mapped_dim=128, num_blocks=3, cls_hidden=64):
        super().__init__()
        self.mapper_s = FeatureMapperResNet(in_dim_src, mapped_dim=mapped_dim, num_blocks=num_blocks)
        self.mapper_t = FeatureMapperResNet(in_dim_tgt, mapped_dim=mapped_dim, num_blocks=num_blocks)
        self.cls = Classifier(mapped_dim, hidden_dim=cls_hidden, num_classes=2)

    def forward_src(self, xs):
        z = self.mapper_s(xs)
        logits = self.cls(z)
        return z, logits

    def forward_tgt(self, xt):
        z = self.mapper_t(xt)
        return z

def compute_mmd_torch(Zs, Zt):
    mu1 = Zs.mean(dim=0)
    mu2 = Zt.mean(dim=0)
    return torch.norm(mu1 - mu2, p=2)

def train_hdp(model, Xs_aug, ys_aug, Xt_train,
              epochs=50, batch_size=128, lr=1e-3, lambda_mmd=0.5, wd=1e-4, log=False):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    ce = nn.CrossEntropyLoss()

    ds_src = TensorDataset(torch.tensor(Xs_aug, dtype=torch.float32),
                           torch.tensor(ys_aug, dtype=torch.long))
    dl_src = DataLoader(ds_src, batch_size=batch_size, shuffle=True, drop_last=True)

    ds_tgt = TensorDataset(torch.tensor(Xt_train, dtype=torch.float32))
    dl_tgt = DataLoader(ds_tgt, batch_size=batch_size, shuffle=True, drop_last=True)

    for ep in range(epochs):
        model.train()
        iters = min(len(dl_src), len(dl_tgt))
        src_it = iter(dl_src); tgt_it = iter(dl_tgt)
        loss_cls_avg, loss_mmd_avg = 0.0, 0.0
        for _ in range(iters):
            xs, ys = next(src_it); (xt,) = next(tgt_it)
            xs=xs.to(DEVICE); ys=ys.to(DEVICE); xt=xt.to(DEVICE)

            z_s, logits = model.forward_src(xs)
            z_t = model.forward_tgt(xt)

            loss_cls = ce(logits, ys)
            loss_mmd = compute_mmd_torch(z_s, z_t)
            loss = loss_cls + lambda_mmd * loss_mmd

            opt.zero_grad(); loss.backward(); opt.step()
            loss_cls_avg += loss_cls.item(); loss_mmd_avg += loss_mmd.item()

        if log:
            print(f"[Epoch {ep+1:03d}] cls={loss_cls_avg/max(1,iters):.4f} mmd={loss_mmd_avg/max(1,iters):.4f}")
    return model

@torch.no_grad()
def evaluate_on_target(model, Xt, yt):
    model.eval()
    Xt = torch.tensor(Xt, dtype=torch.float32).to(DEVICE)
    zt = model.forward_tgt(Xt)
    logits = model.cls(zt)
    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    preds = logits.argmax(dim=1).cpu().numpy()
    yt_np = yt.astype(int)
    return calculate_metrics(yt_np, preds, probs)

# --------- 导出：采样前后 数据对比（类分布/特征变化/快照） ---------
try:
    from scipy.stats import ks_2sample
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _class_stats(y):
    pos = int((y==1).sum()); neg = int((y==0).sum()); n=pos+neg
    return {"n_total": n, "n_pos": pos, "n_neg": neg, "pos_ratio": round(pos/max(1,n),6)}

def _pooled_std(s1, s2, n1, n2):
    if n1<=1 or n2<=1:
        return np.sqrt((s1**2 + s2**2)/2.0 + 1e-12)
    return np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2)/max(1,(n1+n2-2)) + 1e-12)

def compare_and_export_pre_post(X_pre, y_pre, X_post, y_post,
                                feature_names=None, out_dir="results_hdp/compare_pre_post", tag="src_aug"):
    os.makedirs(out_dir, exist_ok=True)

    def _mmd_mean_diff(A,B): return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))

    stat_pre = _class_stats(y_pre); stat_post = _class_stats(y_post)
    mmd_pre_post = _mmd_mean_diff(X_pre, X_post)

    # summary
    summary = pd.DataFrame([{
        "tag": tag,
        "pre_n": stat_pre["n_total"], "pre_pos": stat_pre["n_pos"], "pre_neg": stat_pre["n_neg"], "pre_pos_ratio": stat_pre["pos_ratio"],
        "post_n": stat_post["n_total"], "post_pos": stat_post["n_pos"], "post_neg": stat_post["n_neg"], "post_pos_ratio": stat_post["pos_ratio"],
        "mmd_mean_diff": round(mmd_pre_post,6)
    }])
    summary_path = os.path.join(out_dir, f"summary_{tag}.csv")
    summary.to_csv(summary_path, index=False)

    # feature shift
    n_feat = X_pre.shape[1]
    if feature_names is None or len(feature_names)!=n_feat:
        feature_names=[f"f{i}" for i in range(n_feat)]
    rows=[]
    for j in range(n_feat):
        pre_col=X_pre[:,j]; post_col=X_post[:,j]
        m1=float(np.mean(pre_col)); s1=float(np.std(pre_col, ddof=1)); n1=len(pre_col)
        m2=float(np.mean(post_col)); s2=float(np.std(post_col, ddof=1)); n2=len(post_col)
        delta=m2-m1; ps=_pooled_std(s1,s2,n1,n2); d=delta/(ps+1e-12)
        if _HAS_SCIPY:
            ks_stat, ks_p = ks_2sample(pre_col, post_col, alternative="two-sided", mode="auto")
        else:
            ks_stat, ks_p = np.nan, np.nan
        rows.append({"feature":feature_names[j],
                     "pre_mean":m1,"pre_std":s1,"post_mean":m2,"post_std":s2,
                     "delta_mean":delta,"cohens_d":d,"ks_stat":ks_stat,"ks_pvalue":ks_p})
    feature_df=pd.DataFrame(rows)
    feature_path=os.path.join(out_dir, f"feature_shift_{tag}.csv")
    feature_df.to_csv(feature_path, index=False)

    # snapshots
    def _snapshot(X,y,k=1000):
        n=len(y); k=min(k,n); idx=np.random.choice(n,size=k,replace=False)
        df=pd.DataFrame(X[idx,:], columns=feature_names); df["label"]=y[idx]; return df
    snap_pre=_snapshot(X_pre,y_pre); snap_post=_snapshot(X_post,y_post)
    snap_pre_path=os.path.join(out_dir, f"snapshot_pre_{tag}.csv")
    snap_post_path=os.path.join(out_dir, f"snapshot_post_{tag}.csv")
    snap_pre.to_csv(snap_pre_path, index=False); snap_post.to_csv(snap_post_path, index=False)

    print(f"[OK] pre/post summary  -> {summary_path}")
    print(f"[OK] feature shift     -> {feature_path}")
    print(f"[OK] snapshots         -> {snap_pre_path} | {snap_post_path}")

    return {"summary_csv":summary_path,"feature_shift_csv":feature_path,
            "snapshot_pre_csv":snap_pre_path,"snapshot_post_csv":snap_post_path}

# --------- 多次运行：指标明细/汇总导出 ---------
def mean_ci95(x):
    x=np.asarray(x,dtype=float); n=len(x)
    if n==0: return np.nan, np.nan
    m=np.mean(x); se=(np.std(x,ddof=1)/np.sqrt(n)) if n>1 else 0.0
    return m, 1.96*se

def run_cpdp_hdp_experiment(SOURCE_ARFF, TARGET_ARFF,
                            mapped_dim=128, num_blocks=3, cls_hidden=64,
                            aug_first=True, lambda_mmd=0.5, seed=42,
                            export_prepost_dir=None):
    set_seed(seed)
    # 加载异构源/目标
    Xs_raw, ys_raw = load_single_arff(SOURCE_ARFF)
    Xt_raw, yt_raw = load_single_arff(TARGET_ARFF)

    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        Xt_raw, yt_raw, test_size=0.3, random_state=42, stratify=yt_raw
    )

    # 1) 源域采样增强
    if aug_first:
        Xs_aug, ys_aug, aug_strategy = augment_source_once(Xs_raw, ys_raw, Xt_train, 0.1)
    else:
        Xs_aug, ys_aug, aug_strategy = Xs_raw, ys_raw, "none"

    # 导出“采样前/后”对比
    if export_prepost_dir is not None:
        pair = f"{os.path.basename(SOURCE_ARFF).replace('.arff','')}__{os.path.basename(TARGET_ARFF).replace('.arff','')}"
        out_dir = os.path.join(export_prepost_dir, pair)
        compare_and_export_pre_post(
            X_pre=Xs_raw, y_pre=ys_raw,
            X_post=Xs_aug, y_post=ys_aug,
            feature_names=[f"f{i}" for i in range(Xs_raw.shape[1])],
            out_dir=out_dir, tag=f"{aug_strategy}"
        )

    # 2) 两域映射器训练（源域监督 + MMD 对齐）
    model = HDPModel(in_dim_src=Xs_aug.shape[1], in_dim_tgt=Xt_train.shape[1],
                     mapped_dim=mapped_dim, num_blocks=num_blocks, cls_hidden=cls_hidden)
    model = train_hdp(model, Xs_aug, ys_aug, Xt_train,
                      epochs=50, batch_size=128, lr=1e-3, lambda_mmd=lambda_mmd, wd=1e-4, log=False)

    # 3) 目标域评测
    metrics = evaluate_on_target(model, Xt_test, yt_test)

    # 记录
    def ratio(y):
        pos=int((y==1).sum()); neg=int((y==0).sum()); return pos,neg, round(pos/max(1,pos+neg),4)
    src_pos,src_neg,src_r = ratio(ys_raw)
    aug_pos,aug_neg,aug_r = ratio(ys_aug)
    tgt_pos,tgt_neg,tgt_r = ratio(yt_train)

    record = {
        "source": os.path.basename(SOURCE_ARFF).replace(".arff",""),
        "target": os.path.basename(TARGET_ARFF).replace(".arff",""),
        "seed": seed,
        "aug_strategy": aug_strategy,
        "src_pos": src_pos, "src_neg": src_neg, "src_pos_ratio": src_r,
        "aug_pos": aug_pos, "aug_neg": aug_neg, "aug_pos_ratio": aug_r,
        "tgt_train_pos": tgt_pos, "tgt_train_neg": tgt_neg, "tgt_train_pos_ratio": tgt_r,
        "mapped_dim": mapped_dim, "num_blocks": num_blocks, "cls_hidden": cls_hidden,
        "lambda_mmd": lambda_mmd
    }
    record.update(metrics)  # ACC/F1/BACC/G/AUC/MCC/Pf
    return record

def run_and_export(SOURCE_ARFF, TARGET_ARFF, out_dir="results_hdp",
                   n_runs=30, seeds=None,
                   mapped_dim=128, num_blocks=3, cls_hidden=64,
                   aug_first=True, lambda_mmd=0.5,
                   export_prepost_dir="results_hdp/compare_pre_post"):
    os.makedirs(out_dir, exist_ok=True)
    seeds = seeds or [42+i for i in range(n_runs)]
    records=[]
    for s in seeds:
        rec = run_cpdp_hdp_experiment(
            SOURCE_ARFF, TARGET_ARFF,
            mapped_dim, num_blocks, cls_hidden,
            aug_first, lambda_mmd, seed=s,
            export_prepost_dir=export_prepost_dir
        )
        records.append(rec)

    df = pd.DataFrame(records)
    pair = f"{os.path.basename(SOURCE_ARFF).replace('.arff','')}__{os.path.basename(TARGET_ARFF).replace('.arff','')}"
    detail_path = os.path.join(out_dir, f"detail_{pair}.csv")
    df.to_csv(detail_path, index=False)

    # 汇总（均值±95%CI）
    metrics_cols = ["F1","MCC","Pf","G","ACC","AUC","BACC"]  # 论文用哪些保留哪些
    summary = {"source":df["source"].iloc[0],"target":df["target"].iloc[0],
               "n_runs":len(df),"aug_strategy_mode":df["aug_strategy"].value_counts().idxmax()}
    for m in metrics_cols:
        mean, ci = mean_ci95(df[m].values)
        summary[m] = f"{mean:.4f} ± {ci:.4f}"
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(out_dir, f"summary_{pair}.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"[OK] per-run detail -> {detail_path}")
    print(f"[OK] mean±95%CI     -> {summary_path}")
    return detail_path, summary_path

# ---------------- 入口 ----------------
if __name__ == "__main__":
    # 修改为你的实际路径
    SOURCE_ARFF = "/Users/ding/PycharmProjects/continue_transfer/Datasets/AEEEM/EQ.arff"   # 源域（Java/class/61）
    TARGET_ARFF = "/Users/ding/PycharmProjects/continue_transfer/Datasets/NASA/CM1.arff"   # 目标域（C/function/38）

    # 单次快速跑（含采样前后对比导出）
    _ = run_cpdp_hdp_experiment(
        SOURCE_ARFF, TARGET_ARFF,
        mapped_dim=128, num_blocks=3, cls_hidden=64,
        aug_first=True, lambda_mmd=0.5, seed=42,
        export_prepost_dir="results_hdp/compare_pre_post"
    )

    # 多次重复 + 导出明细/汇总
    run_and_export(
        SOURCE_ARFF, TARGET_ARFF,
        out_dir="results_hdp",
        n_runs=30,
        mapped_dim=128, num_blocks=3, cls_hidden=64,
        aug_first=True, lambda_mmd=0.5,
        export_prepost_dir="results_hdp/compare_pre_post"
    )