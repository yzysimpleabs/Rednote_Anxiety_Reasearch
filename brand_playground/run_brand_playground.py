import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def log1p_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(np.clip(s, 0, np.inf))


def winsorize_series(s, lower=0.01, upper=0.99):
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return s
    ql, qh = s.quantile([lower, upper])
    return s.clip(ql, qh)


def safe_ratio(numer, denom):
    numer = np.asarray(numer, dtype="float64")
    denom = np.asarray(denom, dtype="float64")
    out = np.zeros_like(numer, dtype="float64")
    np.divide(numer, denom, out=out, where=denom != 0)
    return out


def plot_heatmap(df, value_col, title, out_path, fmt=".2f", cmap="viridis"):
    plt.figure(figsize=(10, 5.5))
    ax = sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("anxiety_level")
    ax.set_ylabel("follower_tier")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_importance(importances, feature_names, title, out_path, top_n=25):
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


def main():
    input_path = r"d:\investment\DMS\DMS001_enriched.csv"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "output")
    fig_dir = os.path.join(out_dir, "figs")
    ensure_dir(fig_dir)

    df = pd.read_csv(input_path)
    sns.set_theme(style="whitegrid", font="Arial")

    required = [
        "koc_id",
        "follower_count",
        "anxiety_score",
        "view",
        "like",
        "collect",
        "comments",
        "share",
        "is_commercial",
        "note_quote",
        "video_cpe_calc",
        "ad_value_per_1000_view",
    ]
    missing = [c for c in required if c not in df.columns]
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
        "is_commercial",
        "note_quote",
        "video_cpe_calc",
        "ad_value_per_1000_view",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["collect_like_ratio"] = np.where(df["like"] > 0, df["collect"] / df["like"], np.nan)
    df["collect_like_ratio"] = df["collect_like_ratio"].fillna(0.0)

    df["anxiety_level"] = df["anxiety_score"].round().astype("Int64")
    levels = [0, 1, 3, 5]
    df = df[df["anxiety_level"].isin(levels)].copy()

    bins = [0, 50000, 100000, 200000, np.inf]
    labels = ["<=50k", "50k-100k", "100k-200k", "200k+"]
    df["follower_tier"] = pd.cut(df["follower_count"].fillna(0.0), bins=bins, labels=labels, include_lowest=True, right=True)

    df_brand = df[df["is_commercial"] == 1].copy()
    if df_brand.empty:
        df_brand = df.copy()

    df_brand["roi_view_per_yuan"] = safe_ratio(df_brand["view"], df_brand["note_quote"])

    g = (
        df_brand.groupby(["follower_tier", "anxiety_level"], as_index=False)
        .agg(
            avg_cpe=("video_cpe_calc", "mean"),
            avg_ad_value_1000=("ad_value_per_1000_view", "mean"),
            roi_view_per_yuan=("roi_view_per_yuan", "mean"),
            n=("koc_id", "count"),
        )
        .copy()
    )
    g_path = os.path.join(out_dir, "brand_heatmap_cells.csv")
    g.to_csv(g_path, index=False, encoding="utf-8-sig")

    pivot_cpe = g.pivot(index="follower_tier", columns="anxiety_level", values="avg_cpe").reindex(labels)
    pivot_val = g.pivot(index="follower_tier", columns="anxiety_level", values="avg_ad_value_1000").reindex(labels)
    pivot_roi = g.pivot(index="follower_tier", columns="anxiety_level", values="roi_view_per_yuan").reindex(labels)

    plot_heatmap(
        pivot_cpe,
        "avg_cpe",
        "Avg CPE (video_cpe_calc) by follower tier × anxiety level",
        os.path.join(fig_dir, "heatmap_avg_cpe.png"),
        fmt=".2f",
        cmap="mako",
    )
    plot_heatmap(
        pivot_val,
        "avg_ad_value_1000",
        "Avg ad value per 1000 views (ad_value_per_1000_view) by tier × anxiety",
        os.path.join(fig_dir, "heatmap_ad_value_1000.png"),
        fmt=".2f",
        cmap="viridis",
    )
    plot_heatmap(
        pivot_roi,
        "roi_view_per_yuan",
        "Brand ROI proxy: views per yuan by tier × anxiety",
        os.path.join(fig_dir, "heatmap_roi_views_per_yuan.png"),
        fmt=".4f",
        cmap="crest",
    )

    koc = (
        df.groupby("koc_id", as_index=False)
        .agg(
            follower_count=("follower_count", "mean"),
            anxiety_score=("anxiety_score", "mean"),
            view_mean=("view", "mean"),
            collect_like_ratio=("collect_like_ratio", "mean"),
            comments_mean=("comments", "mean"),
            share_mean=("share", "mean"),
            is_commercial_rate=("is_commercial", "mean"),
            commercial_count=("is_commercial", "sum"),
            note_quote=("note_quote", "mean"),
        )
        .copy()
    )
    koc["commercial_income"] = koc["note_quote"] * koc["commercial_count"]
    koc["follower_count_log1p"] = log1p_series(koc["follower_count"])
    koc["view_mean_log1p"] = log1p_series(koc["view_mean"])
    koc["note_quote_log1p"] = log1p_series(koc["note_quote"])

    for c in ["commercial_income", "view_mean", "collect_like_ratio", "comments_mean", "share_mean", "follower_count"]:
        koc[c] = winsorize_series(koc[c], 0.01, 0.99).fillna(0.0)

    y = log1p_series(koc["commercial_income"]).fillna(0.0).to_numpy(dtype="float64")
    feature_cols = [
        "anxiety_score",
        "view_mean_log1p",
        "collect_like_ratio",
        "comments_mean",
        "share_mean",
        "follower_count_log1p",
        "is_commercial_rate",
    ]
    X = koc[feature_cols].fillna(0.0).to_numpy(dtype="float64")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf = RandomForestRegressor(n_estimators=1200, random_state=42, min_samples_leaf=2, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_r2 = rf.score(X_test, y_test)

    hgb = HistGradientBoostingRegressor(random_state=42, max_depth=4, learning_rate=0.06, max_iter=800)
    hgb.fit(X_train, y_train)
    hgb_r2 = hgb.score(X_test, y_test)

    rf_imp = plot_importance(
        rf.feature_importances_,
        feature_cols,
        f"RF feature importance (target=ln(commercial_income+1), R2={rf_r2:.3f})",
        os.path.join(fig_dir, "ltv_rf_feature_importance.png"),
        top_n=len(feature_cols),
    )

    perm = permutation_importance(hgb, X_test, y_test, n_repeats=50, random_state=42)
    _ = plot_importance(
        perm.importances_mean,
        feature_cols,
        f"HGB permutation importance (target=ln(commercial_income+1), R2={hgb_r2:.3f})",
        os.path.join(fig_dir, "ltv_hgb_permutation_importance.png"),
        top_n=len(feature_cols),
    )

    if "anxiety_score" in feature_cols:
        idx = feature_cols.index("anxiety_score")
        fig, ax = plt.subplots(figsize=(7, 4))
        PartialDependenceDisplay.from_estimator(hgb, X, [idx], ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "ltv_pdp_anxiety.png"), dpi=200, bbox_inches="tight")
        plt.close()

    pred = hgb.predict(X).astype("float64")
    pred_df = koc[["koc_id"] + feature_cols].copy()
    pred_df["target_ln_income_plus1"] = y
    pred_df["pred_ln_income_plus1"] = pred
    pred_df["pred_income"] = np.expm1(pred_df["pred_ln_income_plus1"]).round(2)
    pred_df["actual_income"] = np.expm1(pred_df["target_ln_income_plus1"]).round(2)
    pred_df = pred_df.sort_values("pred_income", ascending=False)
    pred_path = os.path.join(out_dir, "creator_ltv_predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.scatter(pred_df["actual_income"], pred_df["pred_income"], s=70, alpha=0.8, edgecolor="black")
    lim = max(pred_df["actual_income"].max(), pred_df["pred_income"].max(), 1.0)
    ax.plot([0, lim], [0, lim], color="red", linewidth=1)
    ax.set_xlabel("actual income")
    ax.set_ylabel("predicted income")
    ax.set_title("Creator LTV model: predicted vs actual")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ltv_pred_vs_actual.png"), dpi=200, bbox_inches="tight")
    plt.close()

    actions = ["like", "collect", "comments", "share"]
    flow_rows = []
    for a in levels:
        sub = df[df["anxiety_level"] == a].copy()
        total_view = float(sub["view"].sum())
        for act in actions:
            per_1000 = 0.0 if total_view == 0 else float(sub[act].sum()) / total_view * 1000
            flow_rows.append({"anxiety_level": int(a), "action": act, "per_1000_view": per_1000})
    flow = pd.DataFrame(flow_rows)
    flow_path = os.path.join(out_dir, "behavior_flow_per_1000_view.csv")
    flow.to_csv(flow_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))
    ax = sns.barplot(data=flow, x="anxiety_level", y="per_1000_view", hue="action")
    ax.set_title("User action intensity per 1000 views by anxiety level")
    ax.set_xlabel("anxiety_level")
    ax.set_ylabel("actions per 1000 views")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "behavior_action_per_1000_view.png"), dpi=200, bbox_inches="tight")
    plt.close()

    from matplotlib.sankey import Sankey

    fig = plt.figure(figsize=(14, 6))
    for i, a in enumerate(levels, start=1):
        sub = df[df["anxiety_level"] == a].copy()
        total_view = float(sub["view"].sum())
        like_v = 0.0 if total_view == 0 else float(sub["like"].sum()) / total_view * 1000
        collect_v = 0.0 if total_view == 0 else float(sub["collect"].sum()) / total_view * 1000
        comment_v = 0.0 if total_view == 0 else float(sub["comments"].sum()) / total_view * 1000
        share_v = 0.0 if total_view == 0 else float(sub["share"].sum()) / total_view * 1000
        total = like_v + collect_v + comment_v + share_v

        ax = fig.add_subplot(2, 2, i)
        sk = Sankey(ax=ax, scale=1.0 / max(total, 1e-6), offset=0.2, head_angle=120, gap=0.5)
        sk.add(
            flows=[total, -like_v, -collect_v, -comment_v, -share_v],
            labels=["total", "like", "collect", "comment", "share"],
            orientations=[0, 1, 1, 1, 1],
            trunklength=1.2,
        )
        sk.finish()
        ax.set_title(f"anxiety={a} (per 1000 views)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "behavior_sankey_by_anxiety.png"), dpi=200, bbox_inches="tight")
    plt.close()

    summary = pd.DataFrame(
        [
            {"model": "RF", "r2_test": float(rf_r2), "n_creators": int(len(koc)), "n_features": int(len(feature_cols))},
            {"model": "HGB", "r2_test": float(hgb_r2), "n_creators": int(len(koc)), "n_features": int(len(feature_cols))},
        ]
    )
    summary_path = os.path.join(out_dir, "ltv_model_summary.csv")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

