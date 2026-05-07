# 回归结果（原版：混合 OLS + White Robust / HC1；无缩尾、无拆分、无聚类）

数据来源：DMS001_enriched.csv

对数变换：ln(X+1)

标准误：怀特稳健标准误（White Robust / HC1）

![coef_key_terms](figs/coef_key_terms.png)

## 残差直方图

![resid_m1](figs/residuals_model1.png)

![resid_m2](figs/residuals_model2.png)

![resid_m3](figs/residuals_model3.png)

## Model1_CollectLikeRatio

### 模型构成

- 因变量（Y）：ratio_collect — 收藏/点赞（collect/like；当 like<=0 时记为缺失）
- 自变量（X）：const, anxiety_score, anxiety_sq, is_commercial

### 变量释义（参数/term 含义）

| term | 含义 |
|---|---|
| anxiety_score | 焦虑情绪评分（原始字段；数值越高表示越焦虑） |
| anxiety_sq | anxiety_score 的平方项（用于捕捉非线性） |
| const | 截距项（常数项） |
| is_commercial | 是否商业合作笔记（1=商业，0=非商业） |

- N = 400, K = 4, df_resid = 396, R2 = 0.033352

| term | coef | se | t | p | 95% CI low | 95% CI high | |
|---|---:|---:|---:|---:|---:|---:|:--:|
| anxiety_score | -0.14233324 | 0.06531795 | -2.1791 | 0.029914 | -0.27074655 | -0.01391993 | ** |
| anxiety_sq | 0.02087508 | 0.01123527 | 1.8580 | 0.063912 | -0.00121315 | 0.04296331 | * |
| const | 0.59370672 | 0.07067389 | 8.4007 | 0.000000 | 0.45476379 | 0.73264964 | *** |
| is_commercial | 0.06388355 | 0.05343271 | 1.1956 | 0.232573 | -0.04116370 | 0.16893081 |  |

## Model2_lnCESper1000Plus1

### 模型构成

- 因变量（Y）：ln_ces_per_1000_view_plus1 — ln(CES_per_1000_view + 1)，其中 CES=like+collect+4*comments+4*share；CES_per_1000_view=CES/view*1000
- 自变量（X）：const, anxiety_score, anxiety_sq, is_commercial, high_or_low（会展开为哑变量）

### 变量释义（参数/term 含义）

| term | 含义 |
|---|---|
| anxiety_score | 焦虑情绪评分（原始字段；数值越高表示越焦虑） |
| anxiety_sq | anxiety_score 的平方项（用于捕捉非线性） |
| const | 截距项（常数项） |
| high_or_low_low | 达人分层 high_or_low 的哑变量：high_or_low==low 记为 1，否则为 0；基准组（被 drop_first 删除）=high |
| is_commercial | 是否商业合作笔记（1=商业，0=非商业） |

- N = 400, K = 5, df_resid = 395, R2 = 0.016349

| term | coef | se | t | p | 95% CI low | 95% CI high | |
|---|---:|---:|---:|---:|---:|---:|:--:|
| anxiety_score | -0.04812867 | 0.06508967 | -0.7394 | 0.460091 | -0.17609418 | 0.07983684 |  |
| anxiety_sq | 0.00979883 | 0.01234844 | 0.7935 | 0.427947 | -0.01447805 | 0.03407572 |  |
| const | 4.72810748 | 0.07478320 | 63.2242 | 0.000000 | 4.58108462 | 4.87513035 | *** |
| high_or_low_low | 0.04072622 | 0.06373660 | 0.6390 | 0.523208 | -0.08457916 | 0.16603160 |  |
| is_commercial | 0.15265529 | 0.07520034 | 2.0300 | 0.043028 | 0.00481234 | 0.30049824 | ** |

## Model3_lnNoteQuotePlus1

### 模型构成

- 因变量（Y）：ln_note_quote_plus1 — ln(note_quote + 1)，note_quote 为报价（优先视频报价 quote_video，否则图文报价 quote_post）
- 自变量（X）：const, anxiety_score, ln_view_plus1, is_commercial

### 变量释义（参数/term 含义）

| term | 含义 |
|---|---|
| anxiety_score | 焦虑情绪评分（原始字段；数值越高表示越焦虑） |
| const | 截距项（常数项） |
| is_commercial | 是否商业合作笔记（1=商业，0=非商业） |
| ln_view_plus1 | ln(view + 1)，view 为曝光/浏览量（原始字段） |

- N = 400, K = 4, df_resid = 396, R2 = 0.006438

| term | coef | se | t | p | 95% CI low | 95% CI high | |
|---|---:|---:|---:|---:|---:|---:|:--:|
| anxiety_score | 0.01269275 | 0.02861005 | 0.4436 | 0.657541 | -0.04355383 | 0.06893932 |  |
| const | 8.44931915 | 0.22475086 | 37.5942 | 0.000000 | 8.00746511 | 8.89117318 | *** |
| is_commercial | -0.05801864 | 0.10302950 | -0.5631 | 0.573667 | -0.26057181 | 0.14453453 |  |
| ln_view_plus1 | 0.03809580 | 0.02544383 | 1.4973 | 0.135125 | -0.01192608 | 0.08811767 |  |

