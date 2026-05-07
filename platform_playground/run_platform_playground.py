import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def fmt(x, digits=6):
    if x is None:
        return ""
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return ""
        return f"{xf:.{digits}f}"
    except Exception:
        return str(x)


def stars(p):
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.1:
        return "*"
    return ""


def log1p_series(s):
    s = pd.to_numeric(s, errors="coerce")
    return np.log1p(np.clip(s, 0, np.inf))


def build_design_matrix(df, numeric_cols, categorical_cols=None, add_intercept=True, drop_first=True):
    categorical_cols = categorical_cols or []
    parts = []
    names = []
    if add_intercept:
        parts.append(pd.DataFrame({"const": np.ones(len(df), dtype="float64")}, index=df.index))
        names.append("const")

    if numeric_cols:
        xn = df[numeric_cols].astype("float64")
        parts.append(xn)
        names.extend(numeric_cols)

    for c in categorical_cols:
        dum = pd.get_dummies(df[c].astype(str), prefix=c, drop_first=drop_first)
        if dum.shape[1] > 0:
            parts.append(dum.astype("float64"))
            names.extend(dum.columns.tolist())

    Xdf = pd.concat(parts, axis=1)
    col_std = Xdf.std(axis=0, ddof=0).to_numpy(dtype="float64")
    keep = np.asarray(Xdf.columns == "const") | (col_std > 1e-12)
    Xdf = Xdf.loc[:, keep]
    names = [n for n, ok in zip(names, keep) if ok]
    return Xdf.to_numpy(dtype="float64"), names


def ols_white_hc1(X, y):
    y = np.asarray(y, dtype="float64")
    X = np.asarray(X, dtype="float64")
    keep = np.isfinite(y) & np.isfinite(X).all(axis=1)
    X = X[keep, :]
    y = y[keep]

    n, k = X.shape
    if n <= k:
        raise ValueError(f"Not enough observations (n={n}) for k={k}.")

    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta

    xtx = X.T @ X
    bread = np.linalg.pinv(xtx)
    df_resid = n - k

    xu = X * resid.reshape(-1, 1)
    meat = xu.T @ xu
    vcov = bread @ meat @ bread
    if df_resid > 0:
        vcov = vcov * (n / df_resid)

    se = np.sqrt(np.clip(np.diag(vcov), 0, np.inf))
    tval = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se != 0)
    pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=max(df_resid, 1)))
    tcrit = stats.t.ppf(0.975, df=max(df_resid, 1))
    ci_low = beta - tcrit * se
    ci_high = beta + tcrit * se

    y_mean = float(np.mean(y))
    sst = float(np.sum((y - y_mean) ** 2))
    ssr = float(np.sum(resid**2))
    r2 = np.nan if sst == 0 else 1 - ssr / sst

    return {
        "beta": beta,
        "se": se,
        "t": tval,
        "p": pval,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": n,
        "k": k,
        "df_resid": int(df_resid),
        "r2": r2,
        "resid": resid,
        "keep_mask": keep,
    }


def simulate_platform_scenarios(df_notes, scenarios):
    base = df_notes.copy()
    base["income"] = np.where(base["is_commercial"] == 1, base["note_quote"], 0.0)

    rows = []
    for sc in scenarios:
        name = sc["name"]
        mult = np.ones(len(base), dtype="float64")
        mult *= np.where(base["is_commercial"] == 1, sc["commercial_mult"], 1.0)
        mult *= np.where(base["anxiety_score"] >= sc["hi_anxiety_cut"], sc["hi_anxiety_mult"], 1.0)
        mult *= np.where(base["anxiety_score"] <= sc["lo_anxiety_cut"], sc["lo_anxiety_mult"], 1.0)

        sim_view = (base["view"].to_numpy(dtype="float64") * mult).clip(0, np.inf)
        sim_total_view = float(sim_view.sum())
        sim_income = float(base["income"].sum())
        sim_brand_spend = sim_income
        sim_commercial_view = float(sim_view[base["is_commercial"].to_numpy(dtype="int64") == 1].sum())
        roi_view_per_yuan = 0.0 if sim_brand_spend == 0 else sim_commercial_view / sim_brand_spend

        rows.append(
            {
                "scenario": name,
                "total_view": sim_total_view,
                "commercial_view": sim_commercial_view,
                "total_income": sim_income,
                "brand_spend": sim_brand_spend,
                "brand_roi_view_per_yuan": roi_view_per_yuan,
                "hi_anxiety_mult": sc["hi_anxiety_mult"],
                "commercial_mult": sc["commercial_mult"],
            }
        )

    return pd.DataFrame(rows)


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
        "high_or_low",
        "cat",
        "follower_count",
        "anxiety_score",
        "view",
        "like",
        "collect",
        "comments",
        "share",
        "is_commercial",
        "ces_total",
        "note_quote",
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
        "ces_total",
        "note_quote",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["high_or_low"] = df["high_or_low"].astype(str)
    df["cat"] = df["cat"].astype(str)

    df["ces_public"] = df["like"] + df["collect"] + df["comments"] * 4 + df["share"] * 4
    df["anxiety_sq"] = df["anxiety_score"] ** 2

    # 1) Reverse calibration: regress ces_total on interactions + interactions×high_anxiety
    df["high_anxiety_flag"] = (df["anxiety_score"] >= 3).astype("int64")
    for c in ["like", "collect", "comments", "share"]:
        df[f"{c}_x_high_anx"] = df[c] * df["high_anxiety_flag"]

    X1, names1 = build_design_matrix(
        df,
        numeric_cols=["like", "collect", "comments", "share", "like_x_high_anx", "collect_x_high_anx", "comments_x_high_anx", "share_x_high_anx"],
        categorical_cols=["is_commercial"],
        add_intercept=True,
    )
    y1 = df["ces_total"].to_numpy(dtype="float64")
    fit1 = ols_white_hc1(X1, y1)

    calib = pd.DataFrame(
        {
            "term": names1,
            "coef": fit1["beta"],
            "se_white_hc1": fit1["se"],
            "t": fit1["t"],
            "p": fit1["p"],
            "ci_low_95": fit1["ci_low"],
            "ci_high_95": fit1["ci_high"],
        }
    )
    calib["stars"] = calib["p"].apply(stars)
    calib_path = os.path.join(out_dir, "ces_weight_calibration.csv")
    calib.to_csv(calib_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))
    sns.histplot(fit1["resid"], bins=40, kde=True)
    plt.title("Residuals: calibration regression (Y=ces_total)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "calibration_residuals.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 4) Commercial traffic drop by anxiety
    g = (
        df.groupby(["anxiety_score", "is_commercial"], as_index=False)["view"]
        .mean()
        .rename(columns={"view": "mean_view"})
    )
    pivot = g.pivot(index="anxiety_score", columns="is_commercial", values="mean_view").reset_index()
    pivot.columns = ["anxiety_score", "mean_view_noncommercial", "mean_view_commercial"] if pivot.shape[1] == 3 else pivot.columns
    if "mean_view_noncommercial" in pivot.columns and "mean_view_commercial" in pivot.columns:
        pivot["drop_pct"] = np.where(
            pivot["mean_view_noncommercial"] > 0,
            (pivot["mean_view_noncommercial"] - pivot["mean_view_commercial"]) / pivot["mean_view_noncommercial"] * 100,
            np.nan,
        )
    drop_path = os.path.join(out_dir, "commercial_drop_by_anxiety.csv")
    pivot.to_csv(drop_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8, 4))
    sns.lineplot(data=g, x="anxiety_score", y="mean_view", hue="is_commercial", marker="o")
    plt.title("Mean view by anxiety_score × is_commercial")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "view_by_anxiety_commercial.png"), dpi=200, bbox_inches="tight")
    plt.close()

    if "drop_pct" in pivot.columns:
        plt.figure(figsize=(8, 4))
        sns.barplot(data=pivot, x="anxiety_score", y="drop_pct")
        plt.title("Commercial traffic drop (%) by anxiety_score")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "commercial_drop_by_anxiety.png"), dpi=200, bbox_inches="tight")
        plt.close()

    # 2) Platform policy simulation
    scenarios = [
        {"name": "neutral", "hi_anxiety_cut": 3, "hi_anxiety_mult": 1.0, "lo_anxiety_cut": 1, "lo_anxiety_mult": 1.0, "commercial_mult": 1.0},
        {"name": "suppress_high_anxiety", "hi_anxiety_cut": 3, "hi_anxiety_mult": 0.80, "lo_anxiety_cut": 1, "lo_anxiety_mult": 1.0, "commercial_mult": 1.0},
        {"name": "support_anxiety_commercial", "hi_anxiety_cut": 3, "hi_anxiety_mult": 1.05, "lo_anxiety_cut": 1, "lo_anxiety_mult": 1.0, "commercial_mult": 1.10},
    ]
    sim = simulate_platform_scenarios(df[["view", "is_commercial", "anxiety_score", "note_quote"]].copy(), scenarios)
    sim_path = os.path.join(out_dir, "policy_simulation.csv")
    sim.to_csv(sim_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 4))
    sim_m = sim.melt(id_vars=["scenario"], value_vars=["total_view", "commercial_view", "total_income", "brand_roi_view_per_yuan"], var_name="metric", value_name="value")
    sns.barplot(data=sim_m, x="scenario", y="value", hue="metric")
    plt.title("Policy simulation: platform totals / income / brand ROI")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "policy_simulation_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Optimal anxiety mix strategy (grid over proportions)
    levels = [0, 1, 3, 5]
    df["anxiety_level"] = df["anxiety_score"].round().astype("Int64")
    df = df[df["anxiety_level"].isin(levels)].copy()

    agg = (
        df.groupby(["koc_id", "anxiety_level"], as_index=False)
        .agg(
            mean_view=("view", "mean"),
            commercial_rate=("is_commercial", "mean"),
            mean_income_per_note=("note_quote", "mean"),
        )
    )
    agg["mean_income_per_note"] = agg["mean_income_per_note"] * agg["commercial_rate"]
    agg_view = agg.pivot(index="koc_id", columns="anxiety_level", values="mean_view").fillna(0.0)
    agg_inc = agg.pivot(index="koc_id", columns="anxiety_level", values="mean_income_per_note").fillna(0.0)

    grid = np.arange(0, 1.0001, 0.1)
    mixes = []
    for p0 in grid:
        for p1 in grid:
            for p3 in grid:
                p5 = 1.0 - (p0 + p1 + p3)
                if p5 < -1e-9:
                    continue
                if p5 < 0:
                    p5 = 0.0
                mixes.append((p0, p1, p3, p5))
    mixes = np.array(mixes, dtype="float64")

    mix_rows = []
    for koc_id in agg_view.index.tolist():
        v = np.array([agg_view.loc[koc_id].get(l, 0.0) for l in levels], dtype="float64")
        inc = np.array([agg_inc.loc[koc_id].get(l, 0.0) for l in levels], dtype="float64")

        ev = mixes @ v
        ei = mixes @ inc

        best_v = int(np.argmax(ev))
        best_i = int(np.argmax(ei))
        mix_rows.append(
            {
                "koc_id": koc_id,
                "best_mix_for_view": f"{mixes[best_v, 0]:.1f}/{mixes[best_v, 1]:.1f}/{mixes[best_v, 2]:.1f}/{mixes[best_v, 3]:.1f}",
                "best_mix_for_income": f"{mixes[best_i, 0]:.1f}/{mixes[best_i, 1]:.1f}/{mixes[best_i, 2]:.1f}/{mixes[best_i, 3]:.1f}",
                "best_view": float(ev[best_v]),
                "best_income": float(ei[best_i]),
            }
        )
    mix_df = pd.DataFrame(mix_rows)
    mix_path = os.path.join(out_dir, "optimal_mix_by_koc.csv")
    mix_df.to_csv(mix_path, index=False, encoding="utf-8-sig")

    md_path = os.path.join(out_dir, "platform_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📱 平台算法 & 策略视角\n\n")
        f.write("数据源：`DMS001_enriched.csv`\n\n")

        f.write("## 1) 反向校准 CES 权重（回归拟合）\n\n")
        f.write("回归：`ces_total ~ like + collect + comments + share + (like/collect/comments/share)×high_anxiety_flag + is_commercial`\n\n")
        f.write(f"- 产出：`{os.path.basename(calib_path)}`\n")
        f.write(f"- N={fit1['n']}, K={fit1['k']}, df_resid={fit1['df_resid']}, R2={fmt(fit1['r2'], 6)}\n\n")
        f.write("![calibration_residuals](figs/calibration_residuals.png)\n\n")

        f.write("## 2) 平台限流策略博弈仿真\n\n")
        f.write(f"- 产出：`{os.path.basename(sim_path)}`\n\n")
        f.write("![policy_simulation_summary](figs/policy_simulation_summary.png)\n\n")

        f.write("## 3) 博主最优内容配比策略（0/1/3/5 焦虑占比网格搜索）\n\n")
        f.write("配比格式：`p0/p1/p3/p5`（四个比例之和为 1）\n\n")
        f.write(f"- 产出：`{os.path.basename(mix_path)}`（每个博主最大化 view / 收入 的最优配比）\n\n")

        f.write("## 4) 商业化流量折损（按焦虑档位）\n\n")
        f.write(f"- 产出：`{os.path.basename(drop_path)}`\n\n")
        f.write("![view_by_anxiety_commercial](figs/view_by_anxiety_commercial.png)\n\n")
        if os.path.exists(os.path.join(fig_dir, "commercial_drop_by_anxiety.png")):
            f.write("![commercial_drop_by_anxiety](figs/commercial_drop_by_anxiety.png)\n")

    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()

