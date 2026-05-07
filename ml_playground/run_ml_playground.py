import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.cluster.hierarchy import linkage, dendrogram


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    out = np.zeros_like(a, dtype="float64")
    np.divide(a, b, out=out, where=b != 0)
    return out


def log1p_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(np.clip(s, 0, np.inf))


def winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    q_low, q_high = s.quantile([lower, upper])
    return s.clip(q_low, q_high)


def plot_pca_scatter(pca_xy, labels, title, out_path, annotate_labels=None):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    sc = ax.scatter(pca_xy[:, 0], pca_xy[:, 1], c=labels, cmap="tab10", s=90, alpha=0.9, edgecolor="black")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    if annotate_labels is not None:
        for i, t in enumerate(annotate_labels):
            ax.text(pca_xy[i, 0], pca_xy[i, 1], str(t), fontsize=8, ha="left", va="bottom")
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("cluster")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_feature_importance(importances, feature_names, title, out_path, top_n=20):
    imp = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    imp = imp.head(top_n)
    plt.figure(figsize=(10, max(4, 0.35 * len(imp))))
    ax = sns.barplot(data=imp, y="feature", x="importance", orient="h")
    ax.set_title(title)
    ax.set_xlabel("importance")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    return imp


def label_clusters(profile_df):
    prof = profile_df.copy()
    prof["label"] = "均衡全能型博主"

    a = prof["anxiety_score_mean"]
    v = prof["view_mean_log1p"]
    r = prof["collect_like_ratio_mean"]
    cpe = prof["video_cpe_calc_mean_log1p"]

    a_hi = a >= a.median()
    v_hi = v >= v.median()
    r_hi = r >= r.median()
    cpe_lo = cpe <= cpe.median()

    prof.loc[a_hi & r_hi & cpe_lo, "label"] = "高焦虑变现套利型博主"
    prof.loc[(~a_hi) & v_hi & (~r_hi), "label"] = "低焦虑流量型博主"

    tail = (v <= v.quantile(0.25)) & (prof["follower_count_mean_log1p"] <= prof["follower_count_mean_log1p"].quantile(0.25))
    prof.loc[tail, "label"] = "小众低权重尾部博主"

    return prof["label"].to_dict()


def main():
    input_path = r"d:\investment\DMS\DMS001_enriched.csv"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    fig_dir = os.path.join(out_dir, "figs")
    ensure_dir(fig_dir)

    df = pd.read_csv(input_path)

    need = [
        "koc_id",
        "follower_count",
        "anxiety_score",
        "view",
        "like",
        "collect",
        "comments",
        "share",
        "ces_total",
        "note_quote",
        "video_cpe",
        "video_cpe_calc",
        "is_commercial",
        "high_or_low",
        "cat",
    ]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    for c in [
        "follower_count",
        "anxiety_score",
        "view",
        "like",
        "collect",
        "comments",
        "share",
        "ces_total",
        "note_quote",
        "video_cpe",
        "video_cpe_calc",
        "is_commercial",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["high_or_low"] = df["high_or_low"].astype(str)
    df["cat"] = df["cat"].astype(str)

    df["collect_like_ratio"] = np.where(df["like"] > 0, df["collect"] / df["like"], np.nan)
    df["collect_like_ratio"] = df["collect_like_ratio"].fillna(0.0)

    sns.set_theme(style="whitegrid", font="Arial")

    koc_agg = (
        df.groupby("koc_id", as_index=False)
        .agg(
            follower_count=("follower_count", "mean"),
            anxiety_score=("anxiety_score", "mean"),
            view=("view", "mean"),
            collect_like_ratio=("collect_like_ratio", "mean"),
            ces_total=("ces_total", "mean"),
            video_cpe=("video_cpe", "mean"),
            video_cpe_calc=("video_cpe_calc", "mean"),
            note_quote=("note_quote", "mean"),
            high_or_low=("high_or_low", "first"),
            cat=("cat", "first"),
        )
        .copy()
    )

    koc_agg["follower_count_log1p"] = log1p_series(koc_agg["follower_count"])
    koc_agg["view_log1p"] = log1p_series(koc_agg["view"])
    koc_agg["ces_total_log1p"] = log1p_series(koc_agg["ces_total"])
    koc_agg["note_quote_log1p"] = log1p_series(koc_agg["note_quote"])
    koc_agg["video_cpe_calc_log1p"] = log1p_series(koc_agg["video_cpe_calc"])

    cluster_features = [
        "follower_count_log1p",
        "anxiety_score",
        "view_log1p",
        "collect_like_ratio",
        "ces_total_log1p",
        "video_cpe_calc_log1p",
        "note_quote_log1p",
    ]

    X = koc_agg[cluster_features].to_numpy(dtype="float64")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    k = 4
    km = KMeans(n_clusters=k, n_init=50, random_state=42)
    koc_agg["cluster_kmeans"] = km.fit_predict(Xs)
    sil = silhouette_score(Xs, koc_agg["cluster_kmeans"]) if len(koc_agg) > k else np.nan

    pca = PCA(n_components=2, random_state=42)
    pca_xy = pca.fit_transform(Xs)
    plot_pca_scatter(
        pca_xy,
        koc_agg["cluster_kmeans"].to_numpy(),
        f"KMeans (k=4) on creator-level means (silhouette={sil:.3f})",
        os.path.join(fig_dir, "kmeans_pca_scatter.png"),
        annotate_labels=koc_agg["koc_id"].tolist(),
    )

    Z = linkage(Xs, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(Z, labels=koc_agg["koc_id"].astype(str).tolist(), leaf_rotation=90)
    plt.title("Hierarchical clustering dendrogram (Ward linkage)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "hierarchical_dendrogram.png"), dpi=200, bbox_inches="tight")
    plt.close()

    profile = (
        koc_agg.groupby("cluster_kmeans", as_index=True)
        .agg(
            n=("koc_id", "count"),
            follower_count_mean_log1p=("follower_count_log1p", "mean"),
            anxiety_score_mean=("anxiety_score", "mean"),
            view_mean_log1p=("view_log1p", "mean"),
            collect_like_ratio_mean=("collect_like_ratio", "mean"),
            ces_total_mean_log1p=("ces_total_log1p", "mean"),
            video_cpe_calc_mean_log1p=("video_cpe_calc_log1p", "mean"),
            note_quote_mean_log1p=("note_quote_log1p", "mean"),
        )
        .reset_index()
    )

    label_map = label_clusters(profile.set_index("cluster_kmeans"))
    koc_agg["cluster_label"] = koc_agg["cluster_kmeans"].map(label_map)

    profile["cluster_label"] = profile["cluster_kmeans"].map(label_map)
    profile_path = os.path.join(out_dir, "cluster_profile.csv")
    profile.to_csv(profile_path, index=False, encoding="utf-8-sig")

    assign_path = os.path.join(out_dir, "creator_clusters.csv")
    koc_agg[["koc_id", "high_or_low", "cat", "cluster_kmeans", "cluster_label"] + cluster_features].to_csv(
        assign_path, index=False, encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 5))
    heat = koc_agg.sort_values(["cluster_kmeans", "koc_id"])[cluster_features]
    heat_z = (heat - heat.mean()) / heat.std(ddof=0)
    sns.heatmap(heat_z, cmap="RdBu_r", center=0)
    plt.title("Creator-level feature heatmap (z-score)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cluster_feature_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()

    note_feats = pd.DataFrame(
        {
            "koc_id": df["koc_id"].astype(str),
            "view_log1p": log1p_series(df["view"]),
            "ces_total_log1p": log1p_series(df["ces_total"]),
            "collect_like_ratio": df["collect_like_ratio"].astype("float64"),
            "note_quote_log1p": log1p_series(df["note_quote"]),
            "video_cpe_calc_log1p": log1p_series(df["video_cpe_calc"]),
        }
    )

    for c in ["view_log1p", "ces_total_log1p", "collect_like_ratio", "note_quote_log1p", "video_cpe_calc_log1p"]:
        note_feats[c] = winsorize_series(note_feats[c], 0.01, 0.99).fillna(0.0)

    Xn = note_feats[["view_log1p", "ces_total_log1p", "collect_like_ratio", "note_quote_log1p", "video_cpe_calc_log1p"]].to_numpy(
        dtype="float64"
    )
    Xn = StandardScaler().fit_transform(Xn)

    outlier_labels = None
    for eps in [0.9, 1.0, 1.1, 1.2, 1.3]:
        db = DBSCAN(eps=eps, min_samples=6)
        lab = db.fit_predict(Xn)
        outliers = np.sum(lab == -1)
        if 1 <= outliers <= max(5, int(0.1 * len(lab))):
            outlier_labels = lab
            break
    if outlier_labels is None:
        db = DBSCAN(eps=1.2, min_samples=6)
        outlier_labels = db.fit_predict(Xn)

    df_out = df.copy()
    df_out["dbscan_label"] = outlier_labels
    outliers_df = df_out[df_out["dbscan_label"] == -1].copy()
    outliers_path = os.path.join(out_dir, "dbscan_outliers_notes.csv")
    outliers_df.sort_values(["view"], ascending=False).to_csv(outliers_path, index=False, encoding="utf-8-sig")

    pca_note = PCA(n_components=2, random_state=42)
    note_xy = pca_note.fit_transform(Xn)
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    colors = np.where(outlier_labels == -1, "red", "steelblue")
    ax.scatter(note_xy[:, 0], note_xy[:, 1], c=colors, s=18, alpha=0.7)
    ax.set_title("DBSCAN outlier detection on notes (red = outliers)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "dbscan_outliers_pca.png"), dpi=200, bbox_inches="tight")
    plt.close()

    feature_base = pd.DataFrame(
        {
            "follower_count_log1p": log1p_series(df["follower_count"]),
            "anxiety_score": df["anxiety_score"].astype("float64"),
            "anxiety_sq": (df["anxiety_score"] ** 2).astype("float64"),
            "is_commercial": df["is_commercial"].astype("float64"),
            "like_log1p": log1p_series(df["like"]),
            "collect_log1p": log1p_series(df["collect"]),
            "comments_log1p": log1p_series(df["comments"]),
            "share_log1p": log1p_series(df["share"]),
            "ln_view_plus1": log1p_series(df["view"]),
            "collect_like_ratio": df["collect_like_ratio"].astype("float64"),
            "note_quote_log1p": log1p_series(df["note_quote"]),
        }
    )

    cat_dum = pd.get_dummies(df["cat"].astype(str), prefix="cat", drop_first=True)
    hol_dum = pd.get_dummies(df["high_or_low"].astype(str), prefix="high_or_low", drop_first=True)
    X_all = pd.concat([feature_base, cat_dum, hol_dum], axis=1).fillna(0.0)

    tasks = [
        ("predict_view", log1p_series(df["view"]), ["follower_count_log1p", "anxiety_score", "anxiety_sq", "is_commercial", "like_log1p", "collect_log1p", "comments_log1p", "share_log1p"] + cat_dum.columns.tolist() + hol_dum.columns.tolist()),
        ("predict_ces_total", log1p_series(df["ces_total"]), ["follower_count_log1p", "anxiety_score", "anxiety_sq", "is_commercial", "ln_view_plus1", "collect_like_ratio"] + cat_dum.columns.tolist() + hol_dum.columns.tolist()),
        ("predict_video_cpe_calc", log1p_series(df["video_cpe_calc"]), ["follower_count_log1p", "anxiety_score", "anxiety_sq", "is_commercial", "ln_view_plus1", "collect_like_ratio", "note_quote_log1p"] + cat_dum.columns.tolist() + hol_dum.columns.tolist()),
    ]

    ml_rows = []
    for task_name, y, cols in tasks:
        X_task = X_all[cols].to_numpy(dtype="float64")
        y_task = pd.to_numeric(y, errors="coerce").fillna(0.0).to_numpy(dtype="float64")

        X_train, X_test, y_train, y_test = train_test_split(X_task, y_task, test_size=0.25, random_state=42)

        rf = RandomForestRegressor(
            n_estimators=800,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)
        rf_r2 = rf.score(X_test, y_test)

        hgb = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=4,
            learning_rate=0.06,
            max_iter=600,
        )
        hgb.fit(X_train, y_train)
        hgb_r2 = hgb.score(X_test, y_test)

        imp_path = os.path.join(fig_dir, f"{task_name}_rf_feature_importance.png")
        plot_feature_importance(rf.feature_importances_, cols, f"RF feature importance ({task_name})", imp_path, top_n=25)

        perm = permutation_importance(hgb, X_test, y_test, n_repeats=30, random_state=42)
        perm_path = os.path.join(fig_dir, f"{task_name}_hgb_permutation_importance.png")
        plot_feature_importance(perm.importances_mean, cols, f"HGB permutation importance ({task_name})", perm_path, top_n=25)

        if "anxiety_score" in cols:
            idx = cols.index("anxiety_score")
            fig, ax = plt.subplots(figsize=(7, 4))
            PartialDependenceDisplay.from_estimator(hgb, X_task, [idx], ax=ax)
            plt.tight_layout()
            plt.savefig(os.path.join(fig_dir, f"{task_name}_pdp_anxiety.png"), dpi=200, bbox_inches="tight")
            plt.close()

        ml_rows.append(
            {
                "task": task_name,
                "model": "RandomForest",
                "r2_test": rf_r2,
                "n": len(y_task),
                "features": len(cols),
            }
        )
        ml_rows.append(
            {
                "task": task_name,
                "model": "HistGradientBoosting",
                "r2_test": hgb_r2,
                "n": len(y_task),
                "features": len(cols),
            }
        )

    ml_summary = pd.DataFrame(ml_rows)
    ml_summary_path = os.path.join(out_dir, "ml_model_summary.csv")
    ml_summary.to_csv(ml_summary_path, index=False, encoding="utf-8-sig")

    md_path = os.path.join(out_dir, "ml_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🤖 机器学习方向：聚类 + 树模型（无 SHAP 依赖版）\n\n")
        f.write("数据源：`DMS001_enriched.csv`\n\n")
        f.write("说明：当前环境未安装 `shap / xgboost / lightgbm`，树模型解释采用：Permutation Importance + PDP（部分依赖图）。\n\n")

        f.write("## 1) 无监督聚类（按博主聚合均值）\n\n")
        f.write("- 方法：KMeans(k=4) + 层次聚类（Ward）+ PCA(2D)\n")
        f.write(f"- KMeans silhouette = {sil:.4f}\n\n")
        f.write(f"- 产出：`{os.path.basename(assign_path)}`（每个博主的集群归属）\n")
        f.write(f"- 产出：`{os.path.basename(profile_path)}`（各集群画像均值）\n\n")
        f.write("![kmeans_pca_scatter](figs/kmeans_pca_scatter.png)\n\n")
        f.write("![hierarchical_dendrogram](figs/hierarchical_dendrogram.png)\n\n")
        f.write("![cluster_feature_heatmap](figs/cluster_feature_heatmap.png)\n\n")

        f.write("## 2) DBSCAN 异常值检测（按笔记）\n\n")
        f.write(f"- 产出：`{os.path.basename(outliers_path)}`（DBSCAN 标记为 outlier 的笔记）\n\n")
        f.write("![dbscan_outliers_pca](figs/dbscan_outliers_pca.png)\n\n")

        f.write("## 3) 树模型预测 + 可解释性（无 SHAP 依赖版）\n\n")
        f.write(f"- 产出：`{os.path.basename(ml_summary_path)}`（各任务模型 R2 汇总）\n\n")
        f.write("### 任务 A：预测 view（ln(view+1)）\n\n")
        f.write("![predict_view_rf_feature_importance](figs/predict_view_rf_feature_importance.png)\n\n")
        f.write("![predict_view_hgb_permutation_importance](figs/predict_view_hgb_permutation_importance.png)\n\n")
        f.write("![predict_view_pdp_anxiety](figs/predict_view_pdp_anxiety.png)\n\n")

        f.write("### 任务 B：预测 CES 总分（ln(ces_total+1)）\n\n")
        f.write("![predict_ces_total_rf_feature_importance](figs/predict_ces_total_rf_feature_importance.png)\n\n")
        f.write("![predict_ces_total_hgb_permutation_importance](figs/predict_ces_total_hgb_permutation_importance.png)\n\n")
        f.write("![predict_ces_total_pdp_anxiety](figs/predict_ces_total_pdp_anxiety.png)\n\n")

        f.write("### 任务 C：预测视频 CPE（ln(video_cpe_calc+1)）\n\n")
        f.write("![predict_video_cpe_calc_rf_feature_importance](figs/predict_video_cpe_calc_rf_feature_importance.png)\n\n")
        f.write("![predict_video_cpe_calc_hgb_permutation_importance](figs/predict_video_cpe_calc_hgb_permutation_importance.png)\n\n")
        f.write("![predict_video_cpe_calc_pdp_anxiety](figs/predict_video_cpe_calc_pdp_anxiety.png)\n\n")

    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()

