"""
meta_analysis_app.py  —  Proportion meta-analysis, 9 tabs.
Multiple grouping columns; ZIP save with structured folder output.
"""

import io
import re
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
from scipy.special import expit, logit
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory

st.set_page_config(page_title="Meta-Analysis", layout="wide", page_icon="📊")

CB = dict(blue="#0072B2", orange="#E69F00", green="#009E73", vermil="#D55E00",
          lblue="#56B4E9", pink="#CC79A7", black="#000000", grey="#777777")
PAL = [CB["blue"], CB["orange"], CB["green"], CB["vermil"], CB["lblue"], CB["pink"]]

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

st.markdown("""
<style>
[data-testid="stSidebar"] { background:#EEF2FA; }
[data-testid="stSidebar"] * { color:#1A2744 !important; }
[data-testid="stSidebar"] label { color:#2C4A8A !important; font-weight:600; }
[data-testid="stSidebar"] .stMarkdown p { color:#1A2744 !important; }
.stTabs [data-baseweb="tab"] { font-size:14px; font-weight:600; }
h1,h2,h3 { color:#1A2744; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HELPERS
# =============================================================================

def sanitize(s):
    """Make a string safe for use as a folder / file name."""
    return re.sub(r'[^\w\-]', '_', str(s)).strip('_') or 'unnamed'

# =============================================================================
# STATISTICS
# =============================================================================

def fleiss_cc(events, n, cc=0.5):
    e, s = events.astype(float).copy(), n.astype(float).copy()
    m = (e == 0) | (e == s)
    e[m] += cc; s[m] += 2*cc
    return e, s

def logit_vi(events, n):
    e, s = fleiss_cc(np.asarray(events, float), np.asarray(n, float))
    yi = logit(e / s)
    vi = 1.0/e + 1.0/(s - e)
    return yi, vi

def dl_heterogeneity(yi, vi):
    k = len(yi)
    if k < 2:
        return 0.0, 0.0, k-1, 1.0, 0.0
    w = 1.0/vi
    mu_fe = np.dot(w, yi)/w.sum()
    Q = float(np.sum(w*(yi - mu_fe)**2))
    df = k - 1
    C = w.sum() - np.sum(w**2)/w.sum()
    tau2 = max(0.0, (Q - df)/C)
    pval_Q = float(1 - stats.chi2.cdf(Q, df))
    I2 = max(0.0, 100.0*(Q - df)/Q) if Q > df else 0.0
    return tau2, Q, df, pval_Q, I2

def pool_re(yi, vi, tau2):
    k = len(yi)
    W = 1.0/(vi + tau2)
    mu = np.dot(W, yi)/W.sum()
    if k < 2:
        var_h = 1.0/W.sum()
        return dict(mu=mu, ci_lo=mu - 1.96*np.sqrt(var_h),
                    ci_hi=mu + 1.96*np.sqrt(var_h),
                    pi_lo=mu - 1.96*np.sqrt(tau2 + var_h),
                    pi_hi=mu + 1.96*np.sqrt(tau2 + var_h),
                    var=var_h, W=W, k=k)
    q_adj = max(1.0, float(np.sum(W*(yi - mu)**2)) / (k - 1))
    var_h = q_adj/W.sum()
    t_crit = stats.t.ppf(0.975, df=k-1)
    ci_lo = mu - t_crit*np.sqrt(var_h)
    ci_hi = mu + t_crit*np.sqrt(var_h)
    t_pi = stats.t.ppf(0.975, df=max(2, k-2))
    pi_lo = mu - t_pi*np.sqrt(tau2 + var_h)
    pi_hi = mu + t_pi*np.sqrt(tau2 + var_h)
    return dict(mu=mu, ci_lo=ci_lo, ci_hi=ci_hi,
                pi_lo=pi_lo, pi_hi=pi_hi, var=var_h, W=W, k=k)

def run_meta(df):
    yi, vi = logit_vi(df["Cases"].values, df["Sample"].values)
    tau2, Q, df_Q, pval_Q, I2 = dl_heterogeneity(yi, vi)
    res = pool_re(yi, vi, tau2)
    res.update(tau2=tau2, Q=Q, df_Q=df_Q, pval_Q=pval_Q, I2=I2, yi=yi, vi=vi)
    W_fe = 1.0 / vi
    mu_fe = np.dot(W_fe, yi) / W_fe.sum()
    var_fe = 1.0 / W_fe.sum()
    fe_df = max(1, len(yi) - 1)
    t_crit = stats.t.ppf(0.975, df=fe_df)
    res["fe_mu"]    = mu_fe
    res["fe_ci_lo"] = mu_fe - t_crit * np.sqrt(var_fe)
    res["fe_ci_hi"] = mu_fe + t_crit * np.sqrt(var_fe)
    res["W_fe"]     = W_fe
    return res

def fig_png(fig, dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    return buf.getvalue()

def fmt(v, prec=None):
    """Format a float to the globally selected decimal places."""
    p = st.session_state.get("global_prec", 4) if prec is None else prec
    return f"{v:.{p}f}"

def fmt_pct(v, prec=None):
    """Format a back-transformed proportion as a percentage string."""
    p = st.session_state.get("pct_prec", 2) if prec is None else prec
    return f"{float(expit(v))*100:.{p}f}%"

# =============================================================================
# FOREST PLOTS
# =============================================================================

def _iq_box(ax, res):
    re_p = expit(res["mu"]);    re_lo = expit(res["ci_lo"]); re_hi = expit(res["ci_hi"])
    fe_p = expit(res["fe_mu"]); fe_lo = expit(res["fe_ci_lo"]); fe_hi = expit(res["fe_ci_hi"])
    txt = (f"RE: {re_p:.3f} [{re_lo:.3f}, {re_hi:.3f}]     "
           f"FE: {fe_p:.3f} [{fe_lo:.3f}, {fe_hi:.3f}]\n"
           f"k = {res['k']}     I² = {res['I2']:.1f}%     τ² = {res['tau2']:.4f}     "
           f"Q = {res['Q']:.2f} (df={res['df_Q']})     p(het) = {res['pval_Q']:.4f}")
    ax.text(0.5, -0.07, txt,
            transform=ax.transAxes,
            fontsize=9.5, va="top", ha="center",
            family="monospace", color="#D0D8E8", clip_on=False,
            bbox=dict(boxstyle="round,pad=0.45", fc="#1A2744",
                      ec="#0A1628", alpha=0.97, lw=1.2))


def forest_overall(df, res):
    n = len(df)
    yi, vi, W = res["yi"], res["vi"], res["W"]
    fig_h = max(5.5, n*0.42 + 3.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    fig.subplots_adjust(left=0.24, right=0.66, top=0.93, bottom=0.16)
    tr = blended_transform_factory(ax.transAxes, ax.transData)

    ax.text(1.03, n+1.5, "Proportion  [95% CI]",
            transform=tr, fontsize=10, fontweight="bold",
            va="center", ha="left", clip_on=False)
    ax.text(1.38, n+1.5, "Weight",
            transform=tr, fontsize=10, fontweight="bold",
            va="center", ha="left", clip_on=False)

    total_W = W.sum()
    for i, row in enumerate(df.itertuples(index=False)):
        y = n - i
        p  = expit(yi[i])
        pl = expit(yi[i] - 1.96*np.sqrt(vi[i]))
        ph = expit(yi[i] + 1.96*np.sqrt(vi[i]))
        pct = 100.0*W[i]/total_W
        ax.plot([pl, ph], [y, y], color=CB["grey"], lw=1.3,
                solid_capstyle="round", zorder=2)
        ax.plot(p, y, "s", color=CB["blue"], ms=3.5+9.0*pct/100.0,
                mec=CB["black"], mew=0.35, zorder=3)
        ax.text(1.03, y, f"{p:.{st.session_state.get('global_prec',3)}f}  [{pl:.{st.session_state.get('global_prec',3)}f}, {ph:.{st.session_state.get('global_prec',3)}f}]",
                transform=tr, fontsize=9.5, va="center", ha="left",
                clip_on=False, color="#1A2744")
        ax.text(1.38, y, f"{pct:.1f}%",
                transform=tr, fontsize=9.5, va="center", ha="left",
                clip_on=False, color=CB["grey"])

    m  = expit(res["mu"]); cl = expit(res["ci_lo"]); ch = expit(res["ci_hi"])
    plo = expit(res["pi_lo"]); phi_v = expit(res["pi_hi"])
    ax.fill([cl, m, ch, m], [0, 0.42, 0, -0.42], color=CB["vermil"], zorder=4, alpha=0.93)
    ax.plot([plo, phi_v], [0, 0], color=CB["vermil"], lw=1.6, ls="--", zorder=3, alpha=0.60)
    ax.axvline(m, color=CB["vermil"], lw=0.8, ls=":", alpha=0.28, zorder=1)
    ax.axhline(0.5, color="#DDDDDD", lw=0.8, zorder=0)
    ax.text(1.03, 0, f"{m:.3f}  [{cl:.3f}, {ch:.3f}]",
            transform=tr, fontsize=10, va="center", ha="left",
            fontweight="bold", color=CB["vermil"], clip_on=False)
    ax.text(1.38, 0, "100%", transform=tr, fontsize=10, va="center", ha="left",
            fontweight="bold", color=CB["vermil"], clip_on=False)
    fe_m = expit(res["fe_mu"]); fe_cl = expit(res["fe_ci_lo"]); fe_ch = expit(res["fe_ci_hi"])
    ax.fill([fe_cl, fe_m, fe_ch, fe_m], [-1, -0.58, -1, -1.42], color=CB["blue"], zorder=4, alpha=0.80)
    ax.axhline(-0.5, color="#CCCCCC", lw=0.7, zorder=0, ls=":")
    ax.text(1.03, -1, f"{fe_m:.3f}  [{fe_cl:.3f}, {fe_ch:.3f}]",
            transform=tr, fontsize=10, va="center", ha="left",
            fontweight="bold", color=CB["blue"], clip_on=False)
    ax.text(1.38, -1, "100%", transform=tr, fontsize=10, va="center", ha="left",
            fontweight="bold", color=CB["blue"], clip_on=False)

    ax.set_yticks(list(range(n, 0, -1)) + [0, -1])
    ax.set_yticklabels(list(df["Study_ID"]) + ["RE (pooled)", "FE (pooled)"], fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(-2.2, n+2.5)
    ax.set_xlabel("Proportion (95% CI)", fontsize=12, labelpad=5)
    ax.set_title("Forest Plot — Overall", fontsize=14, fontweight="bold", pad=10)
    _iq_box(ax, res)
    ax.legend(handles=[
        Line2D([0],[0], marker="s", color="w", mfc=CB["blue"], ms=9,
               label="Study (size ∝ weight)"),
        mpatches.Patch(facecolor=CB["vermil"], alpha=0.93, label="RE pooled estimate"),
        mpatches.Patch(facecolor=CB["blue"],   alpha=0.80, label="FE pooled estimate"),
        Line2D([0],[0], color=CB["vermil"], lw=1.6, ls="--", alpha=0.6,
               label="95% Prediction interval (RE)"),
    ], fontsize=9, loc="upper right", framealpha=0.88, edgecolor="#CCC")
    return fig


def forest_subgroup(df, res_overall, group_col):
    groups = sorted(df[group_col].astype(str).unique())
    g_color = {g: PAL[i % len(PAL)] for i, g in enumerate(groups)}

    rows = []
    for g in groups:
        sub = df[df[group_col].astype(str) == g].reset_index(drop=True)
        if sub.empty: continue
        r = run_meta(sub)
        k = len(sub); ev = int(sub["Cases"].sum()); n_t = int(sub["Sample"].sum())
        pq = f"p={r['pval_Q']:.3f}" if k > 1 else "p=N/A"
        hdr = (f"{g}   "
               f"[k={k}  n={n_t}  events={ev}  "
               f"I2={r['I2']:.0f}%  Q={r['Q']:.1f}  {pq}]")
        rows.append({"t":"hdr", "txt":hdr, "g":g})
        for _, s in sub.iterrows():
            yi_i, vi_i = logit_vi([s["Cases"]], [s["Sample"]])
            rows.append({"t":"study", "lbl":s["Study_ID"],
                         "yi":yi_i[0], "vi":vi_i[0], "g":g})
        rows.append({"t":"pool", "lbl":f"{g} (pooled)", "res":r, "g":g})
        rows.append({"t":"gap"})
    rows.append({"t":"overall", "lbl":"Overall (pooled)", "res":res_overall})

    nr = len(rows)
    fig_h = max(6.5, nr*0.42 + 3.0)
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.subplots_adjust(left=0.26, right=0.66, top=0.95, bottom=0.12)
    tr = blended_transform_factory(ax.transAxes, ax.transData)

    y = nr + 1; yticks = []; ylabels = []
    ax.text(1.03, y-0.3, "Proportion  [95% CI]",
            transform=tr, fontsize=10, fontweight="bold",
            va="center", ha="left", clip_on=False)

    for row in rows:
        y -= 1; rt = row["t"]
        if rt == "gap":
            ax.axhline(y+0.5, color="#DDDDDD", lw=0.7, zorder=0)
            continue
        elif rt == "hdr":
            col = g_color[row["g"]]
            ax.text(0.0, y, row["txt"],
                    transform=blended_transform_factory(ax.transAxes, ax.transData),
                    fontsize=9.5, fontweight="bold", color="#1A2744",
                    va="center", ha="left", clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.30",
                              fc=col+"1A", ec=col, lw=0.9, alpha=1.0))
            continue
        elif rt == "study":
            p  = expit(row["yi"]); pl = expit(row["yi"] - 1.96*np.sqrt(row["vi"]))
            ph = expit(row["yi"] + 1.96*np.sqrt(row["vi"]))
            col = g_color[row["g"]]
            ax.plot([pl, ph], [y, y], color=CB["grey"], lw=1.2,
                    solid_capstyle="round", zorder=2)
            ax.plot(p, y, "s", color=col, ms=6, mec=CB["black"], mew=0.35, zorder=3)
            ax.text(1.03, y, f"{p:.{st.session_state.get('global_prec',3)}f}  [{pl:.{st.session_state.get('global_prec',3)}f}, {ph:.{st.session_state.get('global_prec',3)}f}]",
                    transform=tr, fontsize=9.5, va="center", ha="left",
                    clip_on=False, color="#1A2744")
            yticks.append(y); ylabels.append(f"  {row['lbl']}")
        elif rt == "pool":
            r = row["res"]; col = g_color[row["g"]]
            m = expit(r["mu"]); cl = expit(r["ci_lo"]); ch = expit(r["ci_hi"])
            ax.fill([cl, m, ch, m], [y, y+0.36, y, y-0.36], color=col, zorder=4, alpha=0.87)
            ax.text(1.03, y, f"{m:.{st.session_state.get('global_prec',3)}f}  [{cl:.{st.session_state.get('global_prec',3)}f}, {ch:.{st.session_state.get('global_prec',3)}f}]",
                    transform=tr, fontsize=10, va="center", ha="left",
                    fontweight="bold", color=col, clip_on=False)
            yticks.append(y); ylabels.append(f"  {row['lbl']}")
        elif rt == "overall":
            r = row["res"]
            m = expit(r["mu"]); cl = expit(r["ci_lo"]); ch = expit(r["ci_hi"])
            ax.fill([cl, m, ch, m], [y, y+0.44, y, y-0.44],
                    color=CB["vermil"], zorder=5, alpha=0.95)
            ax.axvline(m, color=CB["vermil"], lw=0.8, ls=":", alpha=0.25, zorder=0)
            ax.text(1.03, y, f"{m:.{st.session_state.get('global_prec',3)}f}  [{cl:.{st.session_state.get('global_prec',3)}f}, {ch:.{st.session_state.get('global_prec',3)}f}]",
                    transform=tr, fontsize=10.5, va="center", ha="left",
                    fontweight="bold", color=CB["vermil"], clip_on=False)
            yticks.append(y); ylabels.append(f"  {row['lbl']}")

    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, nr+2.2)
    ax.set_xlabel("Proportion (95% CI)", fontsize=12, labelpad=5)
    ax.set_title(f"Forest Plot — Grouped by: {group_col}",
                 fontsize=14, fontweight="bold", pad=10)
    _iq_box(ax, res_overall)
    return fig

# =============================================================================
# FUNNEL PLOT
# =============================================================================

def funnel_plot(res, df, group_col=None, title="Funnel Plot"):
    yi = res["yi"]; vi = res["vi"]; mu = res["mu"]
    se = np.sqrt(vi); max_se = se.max()*1.18
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.12)
    se_r = np.linspace(0, max_se, 300)
    ax.fill_betweenx(se_r, mu - 1.96*se_r, mu + 1.96*se_r, alpha=0.10, color=CB["grey"])
    ax.plot(mu - 1.96*se_r, se_r, color=CB["grey"], lw=1.0, ls="--", label="95% pseudo-CI")
    ax.plot(mu + 1.96*se_r, se_r, color=CB["grey"], lw=1.0, ls="--")
    ax.axvline(mu, color=CB["vermil"], lw=1.5, alpha=0.7, label="Pooled estimate")

    if group_col and group_col in df.columns:
        groups = sorted(df[group_col].astype(str).unique())
        for i, g in enumerate(groups):
            mask = df[group_col].astype(str).values == g
            ax.scatter(yi[mask], se[mask], color=PAL[i % len(PAL)],
                       s=62, alpha=0.85, edgecolors=CB["black"],
                       linewidths=0.4, zorder=3, label=g)
    else:
        ax.scatter(yi, se, color=CB["blue"], s=62, alpha=0.85,
                   edgecolors=CB["black"], linewidths=0.4, zorder=3)

    ax.set_ylim(max_se, -max_se*0.02)
    ax.set_xlabel("Logit proportion (effect size)", fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.85)
    return fig

# =============================================================================
# PUBLICATION BIAS TESTS
# =============================================================================

def egger_test(yi, vi):
    sei = np.sqrt(vi); prec = 1.0/sei; z_std = yi/sei
    X = np.column_stack([np.ones_like(prec), prec])
    try:
        b = np.linalg.solve(X.T @ X, X.T @ z_std)
    except np.linalg.LinAlgError:
        return None
    resid = z_std - X @ b
    s2 = np.sum(resid**2)/max(1, len(yi)-2)
    se_b = np.sqrt(s2*np.diag(np.linalg.inv(X.T @ X)))
    t_int = b[0]/se_b[0]
    pval = float(2*stats.t.sf(abs(t_int), df=len(yi)-2))
    return dict(intercept=float(b[0]), se_int=float(se_b[0]), t=float(t_int), pval=pval)

def begg_test(yi, vi):
    tau, pval = stats.kendalltau(yi, np.sqrt(vi))
    return dict(tau=float(tau), pval=float(pval))

def trim_and_fill(yi, vi, estimator="L0", maxiter=100):
    """
    Duval & Tweedie (2000) trim-and-fill correction.
    estimator: 'L0' or 'R0'
    Returns dict: k0 (studies imputed), filled_yi, filled_vi,
                  mu_adj, ci_lo, ci_hi on logit scale; p_adj, p_lo, p_hi on proportion scale.
    """
    k = len(yi)

    def _re_pool(y, v):
        tau2 = max(0.0, dl_heterogeneity(y, v)[0])
        w = 1.0 / (v + tau2)
        mu = np.dot(w, y) / w.sum()
        return mu, tau2

    mu, _ = _re_pool(yi, vi)
    k0_prev, k0 = -1, 0

    for _ in range(maxiter):
        yi_c = yi - mu
        ranks = stats.rankdata(np.abs(yi_c))
        if estimator == "R0":
            T_n = float(np.sum(ranks[yi_c > 0]))
            k0 = max(0, int(np.round((4 * T_n) / (k + 1) - 0.5)))
        else:  # L0
            n_pos = int(np.sum(yi_c > 0))
            T_n = float(np.sum(ranks[yi_c > 0])) - n_pos * (n_pos + 1) / 2.0
            denom = k - T_n / max(k, 1)
            k0 = max(0, int(np.round(T_n / denom))) if denom > 0 else 0

        k0 = min(k0, k - 2)          # keep at least 2 studies
        if k0 == k0_prev:
            break
        k0_prev = k0

        # Trim k0 most extreme studies on the positive (right) side
        order = np.argsort(yi)[::-1]  # largest first
        keep = np.sort(order[k0:])
        if len(keep) < 2:
            k0 = 0; break
        mu, _ = _re_pool(yi[keep], vi[keep])

    # Fill: add mirror images of the trimmed studies
    if k0 > 0:
        order = np.argsort(yi)[::-1]
        trimmed_idx = order[:k0]
        filled_yi = np.concatenate([yi, 2 * mu - yi[trimmed_idx]])
        filled_vi = np.concatenate([vi, vi[trimmed_idx]])
    else:
        filled_yi = yi.copy()
        filled_vi = vi.copy()

    # Final pooled estimate using all (observed + filled) studies
    mu_adj, tau2_adj = _re_pool(filled_yi, filled_vi)
    w_adj = 1.0 / (filled_vi + tau2_adj)
    k_all = len(filled_yi)
    se_adj = np.sqrt(1.0 / w_adj.sum())
    t_crit = stats.t.ppf(0.975, df=max(1, k_all - 1))
    ci_lo = mu_adj - t_crit * se_adj
    ci_hi = mu_adj + t_crit * se_adj

    return dict(
        k0=k0,
        filled_yi=filled_yi,
        filled_vi=filled_vi,
        mu_adj=mu_adj,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        tau2_adj=tau2_adj,
        p_adj=float(expit(mu_adj)),
        p_lo=float(expit(ci_lo)),
        p_hi=float(expit(ci_hi)),
    )

def funnel_plot_trimfill(res, tf, title="Funnel Plot — Trim-and-Fill"):
    """
    Funnel plot showing observed studies (filled circles) and
    imputed (trim-and-fill) studies (open circles).
    tf: result dict from trim_and_fill().
    """
    yi_obs = res["yi"]; vi_obs = res["vi"]
    mu_orig = res["mu"]
    mu_adj  = tf["mu_adj"]
    k_obs   = len(yi_obs)

    all_yi = tf["filled_yi"]; all_vi = tf["filled_vi"]
    all_se = np.sqrt(all_vi)
    max_se = all_se.max() * 1.18

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.subplots_adjust(left=0.13, right=0.95, top=0.90, bottom=0.12)
    se_r = np.linspace(0, max_se, 300)

    # Pseudo-CI cone around adjusted estimate
    ax.fill_betweenx(se_r, mu_adj - 1.96*se_r, mu_adj + 1.96*se_r,
                     alpha=0.10, color=CB["grey"])
    ax.plot(mu_adj - 1.96*se_r, se_r, color=CB["grey"], lw=1.0, ls="--", label="95% pseudo-CI (adjusted)")
    ax.plot(mu_adj + 1.96*se_r, se_r, color=CB["grey"], lw=1.0, ls="--")

    # Original pooled estimate line
    ax.axvline(mu_orig, color=CB["vermil"], lw=1.5, alpha=0.65, ls="--", label=f"Original estimate ({expit(mu_orig):.3f})")
    # Adjusted pooled estimate line
    ax.axvline(mu_adj,  color=CB["blue"],   lw=1.8, alpha=0.85, label=f"Adjusted estimate ({expit(mu_adj):.3f})")

    # Observed studies
    se_obs = np.sqrt(vi_obs)
    ax.scatter(yi_obs, se_obs, color=CB["blue"], s=62, alpha=0.85,
               edgecolors=CB["black"], linewidths=0.5, zorder=4, label="Observed studies")

    # Imputed studies
    if tf["k0"] > 0:
        yi_imp = all_yi[k_obs:]; se_imp = all_se[k_obs:]
        ax.scatter(yi_imp, se_imp, color="white", s=62, alpha=0.95,
                   edgecolors=CB["vermil"], linewidths=1.5, zorder=4,
                   marker="o", label=f"Imputed studies (k₀={tf['k0']})")

    ax.set_ylim(max_se, -max_se * 0.02)
    ax.set_xlabel("Logit proportion (effect size)", fontsize=12)
    ax.set_ylabel("Standard Error", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
    ax.legend(fontsize=9, framealpha=0.90)
    return fig

def subgroup_funnel_grid(df, group_col):
    groups = sorted(df[group_col].astype(str).unique())
    nc = min(3, len(groups)); nr = (len(groups) + nc - 1)//nc
    fig, axes = plt.subplots(nr, nc, figsize=(6*nc, 5*nr), squeeze=False)
    for idx, g in enumerate(groups):
        ax = axes[idx//nc][idx%nc]
        sub = df[df[group_col].astype(str) == g]
        yi_s, vi_s = logit_vi(sub["Cases"].values, sub["Sample"].values)
        mu_s = np.dot(1/vi_s, yi_s)/(1/vi_s).sum()
        se_s = np.sqrt(vi_s); max_se = se_s.max()*1.18
        sr = np.linspace(0, max_se, 200); col = PAL[idx % len(PAL)]
        ax.fill_betweenx(sr, mu_s - 1.96*sr, mu_s + 1.96*sr, alpha=0.10, color=CB["grey"])
        ax.plot(mu_s - 1.96*sr, sr, color=CB["grey"], lw=1, ls="--")
        ax.plot(mu_s + 1.96*sr, sr, color=CB["grey"], lw=1, ls="--")
        ax.axvline(mu_s, color=col, lw=1.5, alpha=0.70)
        ax.scatter(yi_s, se_s, color=col, s=55, alpha=0.85,
                   edgecolors=CB["black"], linewidths=0.4, zorder=3)
        subtitle = ""
        if len(yi_s) >= 3:
            eg = egger_test(yi_s, vi_s)
            if eg: subtitle = f"\nEgger p = {eg['pval']:.3f}"
        ax.set_title(f"{g}{subtitle}", fontsize=10, fontweight="bold")
        ax.set_ylim(max_se, -max_se*0.02)
        ax.set_xlabel("Logit proportion", fontsize=9); ax.set_ylabel("SE", fontsize=9)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for idx in range(len(groups), nr*nc):
        axes[idx//nc][idx%nc].set_visible(False)
    fig.suptitle(f"Subgroup Funnel Plots — grouped by: {group_col}",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig

# =============================================================================
# META-REGRESSION
# =============================================================================

def meta_regression(df, res, moderator):
    yi, vi, tau2 = res["yi"], res["vi"], res["tau2"]
    W = 1.0/(vi + tau2)
    col = df[moderator]
    if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
    col = col.reset_index(drop=True).squeeze()
    is_num = pd.to_numeric(col, errors="coerce").notna().all()
    if is_num:
        x_raw = pd.to_numeric(col).values.astype(float)
        X = np.column_stack([np.ones(len(yi)), x_raw])
        x_label = [moderator]
    else:
        cats = sorted(col.astype(str).unique())
        dummies = [(col.astype(str) == c).astype(float).values for c in cats[1:]]
        x_raw = col.values
        X = np.column_stack([np.ones(len(yi))] + dummies) if dummies else np.ones((len(yi),1))
        x_label = [f"{moderator}={c}" for c in cats[1:]]
    k, p = len(yi), X.shape[1]
    XtWX = X.T @ np.diag(W) @ X
    try:
        XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        return None, None, None
    beta = XtWX_inv @ X.T @ np.diag(W) @ yi
    resid = yi - X @ beta
    df_r = max(1, k - p)
    Q_res = float(np.sum(W*resid**2))
    C2 = W.sum() - np.sum(W**2)/W.sum()
    tau2_res = max(0.0, (Q_res - df_r)/C2) if df_r > 0 else 0.0
    se_b = np.sqrt(np.diag(XtWX_inv))
    t_val = beta/se_b
    pvals = [float(2*stats.t.sf(abs(t), df=df_r)) for t in t_val]
    R2 = max(0.0, 100.0*(1 - tau2_res/tau2)) if tau2 > 0 else float("nan")
    coef_table = pd.DataFrame({
        "Coefficient": ["Intercept"] + x_label,
        "Estimate": np.round(beta, 4), "SE": np.round(se_b, 4),
        "t": np.round(t_val, 3),
        "p-value": [f"{pv:.4f}" for pv in pvals],
    })
    stats_d = dict(Q_res=Q_res, df_res=df_r,
                   pval_Qres=float(1 - stats.chi2.cdf(Q_res, df_r)),
                   tau2_res=tau2_res, R2=R2)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.90, bottom=0.12)
    if is_num:
        sizes = 30 + 500*(W/W.sum())
        ax.scatter(x_raw, expit(yi), s=sizes, color=CB["blue"], alpha=0.75,
                   edgecolors=CB["black"], linewidths=0.4, zorder=3)
        xfit = np.linspace(x_raw.min(), x_raw.max(), 300)
        yfit = expit(beta[0] + beta[1]*xfit)
        ax.plot(xfit, yfit, color=CB["vermil"], lw=2.0, label="Regression line")
        ax.axhline(expit(res["mu"]), color=CB["grey"], lw=1.0, ls="--",
                   alpha=0.6, label="Overall pooled")
        ax.set_xlabel(moderator, fontsize=12); ax.set_ylabel("Proportion", fontsize=12)
        ax.legend(fontsize=9)
    else:
        cat_list = sorted(col.astype(str).unique())
        means = []
        for c in cat_list:
            m = col.astype(str) == c
            yi_c, _ = logit_vi(df.loc[m.values, "Cases"].values,
                                df.loc[m.values, "Sample"].values)
            means.append(float(expit(yi_c.mean())))
        xs = range(len(cat_list))
        ax.bar(xs, means, color=PAL[:len(cat_list)], alpha=0.75, edgecolor=CB["black"], lw=0.5)
        ax.set_xticks(list(xs)); ax.set_xticklabels(cat_list, rotation=25, ha="right", fontsize=10)
        ax.axhline(expit(res["mu"]), color=CB["vermil"], lw=1.5, ls="--", label="Overall pooled")
        ax.set_xlabel(moderator, fontsize=12); ax.set_ylabel("Mean proportion", fontsize=12)
        ax.legend(fontsize=9)
    ax.set_title(f"Meta-Regression: Proportion ~ {moderator}", fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return coef_table, stats_d, fig

# =============================================================================
# CORRELATIONS
# =============================================================================

def correlation_heatmap(df, cols, method="pearson"):
    sub = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 3: return None
    corr = sub.corr(method=method)
    n = len(cols)
    fig, ax = plt.subplots(figsize=(max(5.5, n*1.2), max(4.5, n*1.0)))
    im = ax.imshow(corr.values, cmap="PuOr", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=10)
    ax.set_yticks(range(n)); ax.set_yticklabels(cols, fontsize=10)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            tc = "#FFFFFF" if abs(v) > 0.65 else "#1A2744"
            bbox_kw = dict(boxstyle="round,pad=0.15", fc="#1A2744", ec="none", alpha=0.75) if abs(v) > 0.65 else None
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9.5, color=tc,
                    fontweight="bold" if i == j else "normal", bbox=bbox_kw)
    ax.set_title(f"{method.capitalize()} Correlation Matrix", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig

# =============================================================================
# ZIP GENERATION
# =============================================================================

def generate_zip(analysis_df, excluded_df, res, group_cols, sample_col, cases_col):
    """
    Build a ZIP in memory with structured folders:

    {sample_col}__{cases_col}/
    ├── data/
    │   ├── cleaned_data.csv
    │   └── excluded_data.csv
    ├── overall/
    │   ├── summary_stats.csv
    │   ├── forest_overall.png
    │   ├── funnel_overall.png
    │   └── publication_bias.csv
    └── subgroups/
        └── {group_col}/
            ├── group_summary.csv
            ├── forest_subgroup.png
            ├── funnel_subgroups.png
            └── {group_value}/
                ├── forest.png
                └── stats.csv
    """
    buf = io.BytesIO()
    root = sanitize(f"{sample_col}__{cases_col}")

    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:

        # ── data/ ──────────────────────────────────────────────────────────
        zf.writestr(f"{root}/data/cleaned_data.csv", analysis_df.to_csv(index=False))
        if not excluded_df.empty:
            zf.writestr(f"{root}/data/excluded_data.csv", excluded_df.to_csv(index=False))

        # ── overall/ ───────────────────────────────────────────────────────
        m_p  = expit(res["mu"]);    ci_l = expit(res["ci_lo"]); ci_h = expit(res["ci_hi"])
        pi_l = expit(res["pi_lo"]); pi_h = expit(res["pi_hi"])
        fe_p = expit(res["fe_mu"]); fe_l = expit(res["fe_ci_lo"]); fe_h = expit(res["fe_ci_hi"])

        stats_rows = [
            {"Model": "RE (DL+HKSJ)", "k": res["k"],
             "Pooled": f"{m_p:.4f}", "CI_lo": f"{ci_l:.4f}", "CI_hi": f"{ci_h:.4f}",
             "PI_lo": f"{pi_l:.4f}", "PI_hi": f"{pi_h:.4f}",
             "I2": f"{res['I2']:.1f}%", "tau2": f"{res['tau2']:.4f}",
             "Q": f"{res['Q']:.2f}", "p_het": f"{res['pval_Q']:.4f}"},
            {"Model": "FE (Inv-Var)", "k": res["k"],
             "Pooled": f"{fe_p:.4f}", "CI_lo": f"{fe_l:.4f}", "CI_hi": f"{fe_h:.4f}",
             "PI_lo": "", "PI_hi": "", "I2": "", "tau2": "", "Q": "", "p_het": ""},
        ]
        zf.writestr(f"{root}/overall/summary_stats.csv",
                    pd.DataFrame(stats_rows).to_csv(index=False))

        # Per-study results
        study_rows = []
        for i, row in enumerate(analysis_df.itertuples(index=False)):
            p_s  = expit(res["yi"][i])
            pl_s = expit(res["yi"][i] - 1.96*np.sqrt(res["vi"][i]))
            ph_s = expit(res["yi"][i] + 1.96*np.sqrt(res["vi"][i]))
            wt   = 100.0*res["W"][i]/res["W"].sum()
            study_rows.append({
                "Study": row.Study_ID,
                "n": int(row.Sample), "Events": int(row.Cases),
                "Proportion": f"{p_s:.4f}",
                "CI_lo": f"{pl_s:.4f}", "CI_hi": f"{ph_s:.4f}",
                "Weight_%": f"{wt:.2f}",
            })
        zf.writestr(f"{root}/overall/per_study_results.csv",
                    pd.DataFrame(study_rows).to_csv(index=False))

        # Publication bias
        if len(res["yi"]) >= 3:
            eg = egger_test(res["yi"], res["vi"])
            bg = begg_test(res["yi"], res["vi"])
            if eg:
                pb_rows = [
                    {"Test": "Egger", "Statistic": "Intercept",   "Value": f"{eg['intercept']:.4f}"},
                    {"Test": "Egger", "Statistic": "SE",           "Value": f"{eg['se_int']:.4f}"},
                    {"Test": "Egger", "Statistic": "t",            "Value": f"{eg['t']:.3f}"},
                    {"Test": "Egger", "Statistic": "p-value",      "Value": f"{eg['pval']:.4f}"},
                    {"Test": "Begg",  "Statistic": "Kendall tau",  "Value": f"{bg['tau']:.4f}"},
                    {"Test": "Begg",  "Statistic": "p-value",      "Value": f"{bg['pval']:.4f}"},
                ]
                # Trim-and-fill (always run for export; flag significance)
                tf_zip = trim_and_fill(res["yi"], res["vi"], estimator="L0")
                orig_p  = float(expit(res["mu"]))
                orig_lo = float(expit(res["ci_lo"]))
                orig_hi = float(expit(res["ci_hi"]))
                bias_sig = (eg["pval"] < 0.10) or (bg["pval"] < 0.10)
                pb_rows += [
                    {"Test": "Trim-and-Fill", "Statistic": "Estimator",                "Value": "L0"},
                    {"Test": "Trim-and-Fill", "Statistic": "Studies imputed (k0)",     "Value": str(tf_zip["k0"])},
                    {"Test": "Trim-and-Fill", "Statistic": "Bias confirmed (p<0.10)",  "Value": str(bias_sig)},
                    {"Test": "Trim-and-Fill", "Statistic": "Original proportion",      "Value": f"{orig_p:.4f}"},
                    {"Test": "Trim-and-Fill", "Statistic": "Original CI lower",        "Value": f"{orig_lo:.4f}"},
                    {"Test": "Trim-and-Fill", "Statistic": "Original CI upper",        "Value": f"{orig_hi:.4f}"},
                    {"Test": "Trim-and-Fill", "Statistic": "Adjusted proportion",      "Value": f"{tf_zip['p_adj']:.4f}"},
                    {"Test": "Trim-and-Fill", "Statistic": "Adjusted CI lower",        "Value": f"{tf_zip['p_lo']:.4f}"},
                    {"Test": "Trim-and-Fill", "Statistic": "Adjusted CI upper",        "Value": f"{tf_zip['p_hi']:.4f}"},
                ]
                zf.writestr(f"{root}/overall/publication_bias.csv",
                            pd.DataFrame(pb_rows).to_csv(index=False))
                # Save adjusted funnel plot
                fig_tf_zip = funnel_plot_trimfill(res, tf_zip,
                                                  title="Funnel Plot — Trim-and-Fill Adjusted")
                zf.writestr(f"{root}/overall/funnel_trimfill.png", fig_png(fig_tf_zip))
                plt.close(fig_tf_zip)

        # Forest overall
        fig_fo = forest_overall(analysis_df, res)
        zf.writestr(f"{root}/overall/forest_overall.png", fig_png(fig_fo))
        plt.close(fig_fo)

        # Funnel overall
        fig_fun = funnel_plot(res, analysis_df, title="Funnel Plot — Overall")
        zf.writestr(f"{root}/overall/funnel_overall.png", fig_png(fig_fun))
        plt.close(fig_fun)

        # ── subgroups/ ─────────────────────────────────────────────────────
        for gc in group_cols:
            gc_safe = sanitize(gc)
            groups = sorted(analysis_df[gc].astype(str).unique())

            if len(groups) < 2:
                continue

            # Group summary CSV
            sg_rows = []
            for g in groups:
                sub = analysis_df[analysis_df[gc].astype(str) == g]
                r_  = run_meta(sub)
                sg_rows.append({
                    gc: g, "k": r_["k"],
                    "Events": int(sub["Cases"].sum()), "N": int(sub["Sample"].sum()),
                    "Proportion": f"{expit(r_['mu']):.4f}",
                    "CI_lo": f"{expit(r_['ci_lo']):.4f}",
                    "CI_hi": f"{expit(r_['ci_hi']):.4f}",
                    "I2": f"{r_['I2']:.1f}%", "tau2": f"{r_['tau2']:.4f}",
                    "Q": f"{r_['Q']:.2f}", "p_het": f"{r_['pval_Q']:.4f}",
                })
            zf.writestr(f"{root}/subgroups/{gc_safe}/group_summary.csv",
                        pd.DataFrame(sg_rows).to_csv(index=False))

            # Subgroup forest
            fig_sg = forest_subgroup(analysis_df, res, gc)
            zf.writestr(f"{root}/subgroups/{gc_safe}/forest_subgroup.png", fig_png(fig_sg))
            plt.close(fig_sg)

            # Subgroup funnel grid
            fig_sfg = subgroup_funnel_grid(analysis_df, gc)
            zf.writestr(f"{root}/subgroups/{gc_safe}/funnel_subgroups.png", fig_png(fig_sfg))
            plt.close(fig_sfg)

            # Per-group-value subfolders
            for g in groups:
                g_safe = sanitize(g)
                sub = analysis_df[analysis_df[gc].astype(str) == g].reset_index(drop=True)
                if len(sub) < 2:
                    continue
                r_sub = run_meta(sub)

                # Per-subgroup stats CSV
                sub_stats = [{
                    "Group": g, "k": r_sub["k"],
                    "Events": int(sub["Cases"].sum()), "N": int(sub["Sample"].sum()),
                    "Proportion": f"{expit(r_sub['mu']):.4f}",
                    "CI_lo": f"{expit(r_sub['ci_lo']):.4f}",
                    "CI_hi": f"{expit(r_sub['ci_hi']):.4f}",
                    "I2": f"{r_sub['I2']:.1f}%", "tau2": f"{r_sub['tau2']:.4f}",
                    "Q": f"{r_sub['Q']:.2f}", "p_het": f"{r_sub['pval_Q']:.4f}",
                }]
                zf.writestr(
                    f"{root}/subgroups/{gc_safe}/{g_safe}/stats.csv",
                    pd.DataFrame(sub_stats).to_csv(index=False)
                )

                # Per-subgroup forest plot
                fig_sub = forest_overall(sub, r_sub)
                fig_sub.axes[0].set_title(
                    f"Forest Plot — {gc}: {g}",
                    fontsize=14, fontweight="bold", pad=10
                )
                zf.writestr(
                    f"{root}/subgroups/{gc_safe}/{g_safe}/forest.png",
                    fig_png(fig_sub)
                )
                plt.close(fig_sub)

    buf.seek(0)
    return buf.getvalue()

# =============================================================================
# DUMMY DATA
# =============================================================================

DUMMY_CSV = (
    "Study_ID,Sample,Cases,Risk_Category,Year,Region,Mean_Age,Pct_Male,"
    "Follow_up_months,Study_Design,Quality_Score,Country_Income\n"
    "Smith 2010,450,85,High risk,2010,North America,52.3,48.2,12,Cohort,7,High\n"
    "Jones 2011,320,62,Low risk,2011,Europe,45.1,55.0,24,Case-Control,6,High\n"
    "Chen 2012,680,110,High risk,2012,Asia,58.7,41.3,18,Cohort,8,Upper-middle\n"
    "Kumar 2013,290,45,Low risk,2013,South Asia,49.2,60.1,6,Cross-sectional,5,Lower-middle\n"
    "Muller 2014,520,95,High risk,2014,Europe,63.4,38.7,36,Cohort,9,High\n"
    "Osei 2014,180,38,Low risk,2014,Africa,42.8,52.3,12,Case-Control,6,Low\n"
    "Park 2015,410,72,High risk,2015,Asia,55.9,45.6,24,Cohort,7,Upper-middle\n"
    "Santos 2015,260,41,Low risk,2015,Latin America,47.3,58.9,18,Case-Control,5,Upper-middle\n"
    "Williams 2016,590,102,High risk,2016,North America,61.2,42.1,36,Cohort,8,High\n"
    "Ahmed 2016,340,58,Low risk,2016,Middle East,50.4,63.7,12,Cohort,7,Upper-middle\n"
    "Li 2017,720,125,High risk,2017,Asia,57.6,39.4,24,Cohort,9,Upper-middle\n"
    "Patel 2017,230,39,Low risk,2017,South Asia,44.7,56.2,6,Cross-sectional,6,Lower-middle\n"
    "Rossi 2018,480,88,High risk,2018,Europe,64.1,37.8,30,Cohort,8,High\n"
    "Tanaka 2018,370,63,Low risk,2018,Asia,51.3,49.5,18,Case-Control,7,High\n"
    "Brown 2019,560,97,High risk,2019,North America,59.8,43.2,24,Cohort,8,High\n"
    "Diallo 2019,210,36,Low risk,2019,Africa,41.5,54.8,12,Cross-sectional,5,Low\n"
    "Garcia 2020,440,79,High risk,2020,Latin America,56.4,40.9,18,Cohort,7,Upper-middle\n"
    "Ibrahim 2020,300,51,Low risk,2020,Middle East,48.6,61.3,6,Case-Control,6,Upper-middle\n"
    "Kim 2021,650,115,High risk,2021,Asia,60.7,38.1,36,Cohort,9,High\n"
    "Okonkwo 2021,190,33,Low risk,2021,Africa,43.2,53.6,12,Cross-sectional,5,Low\n"
    "Meyer 2022,510,91,High risk,2022,Europe,62.5,41.7,24,Cohort,8,High\n"
    "Sharma 2022,270,46,Low risk,2022,South Asia,46.9,59.4,18,Case-Control,6,Lower-middle\n"
    "Thompson 2023,600,108,High risk,2023,North America,58.3,44.5,12,Cohort,9,High\n"
    "Fernandez 2023,350,60,Low risk,2023,Latin America,50.1,57.2,6,Cross-sectional,6,Upper-middle\n"
    "Wang 2024,470,84,High risk,2024,Asia,57.1,40.3,24,Cohort,8,Upper-middle\n"
    "Nguyen 2018,200,NR,High risk,2018,Asia,53.4,47.8,12,Cohort,7,Lower-middle\n"
    "Jackson 2021,150,200,Low risk,2021,North America,48.9,52.1,18,Case-Control,6,High\n"
    "Hassan 2023,ABC,45,High risk,2023,Middle East,55.7,43.2,24,Cohort,7,Upper-middle\n"
    "Petrov 2022,,38,Low risk,2022,Europe,49.3,56.8,12,Cross-sectional,5,High\n"
    "Lee 2020,380,67,NR,2020,Asia,54.6,46.1,18,Cohort,8,Upper-middle\n"
)

# =============================================================================
# DATA CLEANING
# =============================================================================

def load_and_clean(raw, sc, nc, cc):
    df = raw.copy()
    for target, source in [("Study_ID", sc), ("Sample", nc), ("Cases", cc)]:
        if source != target and target in df.columns:
            df = df.drop(columns=[target])
    df = df.rename(columns={sc: "Study_ID", nc: "Sample", cc: "Cases"})
    reasons = []
    for _, r in df.iterrows():
        err = []
        try:
            n_ = float(r["Sample"])
            if np.isnan(n_): err.append("Sample missing")
        except Exception: err.append("Sample non-numeric")
        try:
            c_ = float(r["Cases"])
            if np.isnan(c_): err.append("Cases missing")
        except Exception: err.append("Cases non-numeric")
        if not err:
            n_, c_ = float(r["Sample"]), float(r["Cases"])
            if n_ <= 0: err.append("Sample <= 0")
            if c_ < 0:  err.append("Cases < 0")
            if c_ > n_: err.append("Cases > Sample")
        reasons.append("; ".join(err))
    df["Exclusion Reason"] = reasons
    excl  = df[df["Exclusion Reason"] != ""].reset_index(drop=True)
    clean = df[df["Exclusion Reason"] == ""].drop(
        columns=["Exclusion Reason"]).reset_index(drop=True)
    clean["Sample"] = pd.to_numeric(clean["Sample"])
    clean["Cases"]  = pd.to_numeric(clean["Cases"])
    return clean, excl

# =============================================================================
# APP
# =============================================================================

st.title("Meta-Analysis  —  Proportion Pooling")
st.caption("Random-effects (DerSimonian-Laird tau2)  |  HKSJ correction  |  Logit transformation  |  Okabe-Ito palette")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Setup")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if not uploaded:
        st.download_button("Download demo CSV", DUMMY_CSV, "demo_meta_data.csv", "text/csv")

    raw_df = pd.read_csv(uploaded) if uploaded else pd.read_csv(io.StringIO(DUMMY_CSV))
    all_cols = raw_df.columns.tolist()

    def guess(kws, fallback=None):
        for kw in kws:
            for c in all_cols:
                if kw.lower() in c.lower(): return c
        return fallback or all_cols[0]

    st.markdown("### Column mapping")
    st.caption("Map your CSV columns to the required fields for analysis.")
    def safe_index(col):
        return all_cols.index(col) if col in all_cols else 0
    study_col  = st.selectbox("Study ID",       all_cols, index=safe_index(guess(["study","id","author"])))
    sample_col = st.selectbox("Sample (n)",     all_cols, index=safe_index(guess(["sample","total","n"],"Sample")))
    cases_col  = st.selectbox("Cases / Events", all_cols, index=safe_index(guess(["case","event","count"],"Cases")))

    cleaned_df, excluded_df = load_and_clean(raw_df, study_col, sample_col, cases_col)

    # ── Group by  (multi-select) ───────────────────────────────────────────
    st.markdown("### Group by")
    st.caption("Select one or more columns to define subgroups. Each gets its own analysis and folder in the saved output.")
    group_candidates = [c for c in cleaned_df.columns if c not in {"Study_ID","Sample","Cases"}]
    group_cols = st.multiselect(
        "Group by … (select multiple)",
        group_candidates,
        default=[group_candidates[0]] if group_candidates else [],
    )
    # Primary group col — used where a single column is required (focus filter, funnel colouring)
    group_col = group_cols[0] if group_cols else None

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("### Filters")
    manual_excl = st.multiselect("Manually exclude studies", cleaned_df["Study_ID"].tolist())

    if group_col:
        sg_opts  = ["All groups"] + sorted(cleaned_df[group_col].astype(str).unique())
        sg_focus = st.selectbox(f"Focus on {group_col}", sg_opts,
                                help="Restricts the entire analysis to this subgroup value. "
                                     "Only applies to the primary (first) Group by column.")
    else:
        sg_focus = "All groups"

    analysis_df = cleaned_df.copy()
    if manual_excl:
        analysis_df = analysis_df[~analysis_df["Study_ID"].isin(manual_excl)].reset_index(drop=True)
    if sg_focus != "All groups" and group_col:
        analysis_df = analysis_df[analysis_df[group_col].astype(str) == sg_focus].reset_index(drop=True)

    # ── Display precision ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Display precision")
    PREC = st.slider(
        "Decimal places",
        min_value=1, max_value=6, value=4, step=1,
        help="Controls how many decimal places appear in all numeric outputs (tables, metrics, forest plot labels).",
        key="global_prec",
    )
    PCT_PREC = st.slider(
        "Proportion display",
        min_value=0, max_value=4, value=2, step=1,
        help="Decimal places for back-transformed proportions shown as percentages (e.g. 2 → 63.47%).",
        key="pct_prec",
    )

    st.markdown("---")
    st.markdown(f"Studies in analysis: **{len(analysis_df)}**")
    st.markdown(f"Excluded: **{len(excluded_df) + len(manual_excl)}**")
    for gc in group_cols:
        n_g = analysis_df[gc].astype(str).nunique()
        st.markdown(f"Groups ({gc}): **{n_g}**")

    # ── SAVE ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Save All Outputs")
    st.caption(
        "Generates a ZIP with structured folders:\n"
        f"`{sanitize(sample_col)}__{sanitize(cases_col)}/`\n"
        "  `data/`  `overall/`  `subgroups/{group}/{value}/`"
    )

    if st.button("Generate ZIP", type="primary", use_container_width=True):
        if len(analysis_df) >= 2:
            with st.spinner("Building outputs…"):
                zip_bytes = generate_zip(
                    analysis_df, excluded_df, res if "res" in dir() else run_meta(analysis_df),
                    group_cols, sample_col, cases_col
                )
                st.session_state["zip_bytes"]    = zip_bytes
                st.session_state["zip_filename"] = (
                    f"{sanitize(sample_col)}__{sanitize(cases_col)}_meta_analysis.zip"
                )
        else:
            st.error("Need ≥ 2 studies to generate outputs.")

    if "zip_bytes" in st.session_state:
        st.download_button(
            "⬇️ Download ZIP  (Save As…)",
            data=st.session_state["zip_bytes"],
            file_name=st.session_state["zip_filename"],
            mime="application/zip",
            use_container_width=True,
        )
        st.caption("Your browser's Save As dialog will open.")

# ── Guard ──────────────────────────────────────────────────────────────────────
if len(analysis_df) < 2:
    st.error("Need >= 2 valid studies. Check column mapping or filters.")
    st.stop()

res = run_meta(analysis_df)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Cleaned Data", "Excluded Data", "Summary Stats",
    "Forest (Overall)", "Forest (Subgroup)",
    "Funnel Plot", "Publication Bias",
    "Meta-Regression", "Correlations",
])

# ── 0 Cleaned Data ─────────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("Cleaned Data")
    st.caption(f"{len(cleaned_df)} studies passed all quality checks")
    st.dataframe(cleaned_df, use_container_width=True, hide_index=True)
    st.download_button("Download cleaned CSV", cleaned_df.to_csv(index=False), "cleaned.csv")

# ── 1 Excluded Data ─────────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("Excluded Data")
    if excluded_df.empty:
        st.success("No rows excluded during data cleaning.")
    else:
        st.caption(f"{len(excluded_df)} rows excluded — reasons shown in last column")
        st.dataframe(excluded_df, use_container_width=True, hide_index=True)
        st.download_button("Download excluded CSV", excluded_df.to_csv(index=False), "excluded.csv")

# ── 2 Summary Stats ─────────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("Summary Statistics")
    m_p  = expit(res["mu"]);  ci_l = expit(res["ci_lo"]); ci_h = expit(res["ci_hi"])
    pi_l = expit(res["pi_lo"]); pi_h = expit(res["pi_hi"])
    fe_p = expit(res["fe_mu"]); fe_l = expit(res["fe_ci_lo"]); fe_h = expit(res["fe_ci_hi"])
    _p = PREC  # local alias for readability
    st.markdown("##### Random-Effects Model (DerSimonian-Laird + HKSJ)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Studies (k)",        res["k"])
    c2.metric("RE Pooled",          f"{m_p:.{_p}f}")
    c3.metric("95% CI (RE)",        f"[{ci_l:.{_p}f}, {ci_h:.{_p}f}]")
    c4.metric("95% PI",             f"[{pi_l:.{_p}f}, {pi_h:.{_p}f}]")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("I²",                 f"{res['I2']:.1f}%")
    c6.metric("τ²",                 f"{res['tau2']:.{_p}f}")
    c7.metric("Q (df={})".format(res['df_Q']), f"{res['Q']:.{_p}f}")
    c8.metric("p(heterogeneity)",   f"{res['pval_Q']:.{_p}f}")
    st.markdown("##### Fixed-Effects Model (Inverse-Variance)")
    cf1, cf2, cf3 = st.columns(3)
    cf1.metric("FE Pooled",         f"{fe_p:.{_p}f}")
    cf2.metric("95% CI (FE)",       f"[{fe_l:.{_p}f}, {fe_h:.{_p}f}]")
    cf3.metric("Note", "Assumes homogeneity")

    st.markdown("---")
    st.subheader("Per-Study Results")
    study_rows = []
    for i, row in enumerate(analysis_df.itertuples(index=False)):
        p  = expit(res["yi"][i])
        pl = expit(res["yi"][i] - 1.96*np.sqrt(res["vi"][i]))
        ph = expit(res["yi"][i] + 1.96*np.sqrt(res["vi"][i]))
        wt = 100.0*res["W"][i]/res["W"].sum()
        r_dict = {"Study": row.Study_ID, "n": int(row.Sample), "Events": int(row.Cases),
                  "Weight_pct_RE": f"{wt:.1f}%"}
        for gc in group_cols:
            r_dict[gc] = getattr(row, gc, "")
        study_rows.append(r_dict)
    st.dataframe(pd.DataFrame(study_rows), use_container_width=True, hide_index=True)

    # Group summaries for every selected group col
    for gc in group_cols:
        st.markdown("---")
        st.subheader(f"Group Summary  (grouped by: {gc})")
        sg_rows = []
        for g in sorted(analysis_df[gc].astype(str).unique()):
            sub = analysis_df[analysis_df[gc].astype(str) == g]
            r_  = run_meta(sub)
            m_s = expit(r_["mu"])
            sg_rows.append({
                gc: g, "k": r_["k"],
                "Events": int(sub["Cases"].sum()), "N": int(sub["Sample"].sum()),
                "Proportion": f"{m_s:.{PREC}f}",
                "95% CI": f"[{expit(r_['ci_lo']):.{PREC}f}, {expit(r_['ci_hi']):.{PREC}f}]",
                "I2": f"{r_['I2']:.1f}%", "tau2": f"{r_['tau2']:.{PREC}f}",
                "Q": f"{r_['Q']:.{PREC}f}", "p(het)": f"{r_['pval_Q']:.{PREC}f}",
            })
        st.dataframe(pd.DataFrame(sg_rows), use_container_width=True, hide_index=True)

# ── 3 Forest (Overall) ──────────────────────────────────────────────────────────
with tabs[3]:
    st.subheader("Forest Plot — Overall")
    fig = forest_overall(analysis_df, res)
    st.pyplot(fig, use_container_width=False)
    st.download_button("Download forest (overall)", fig_png(fig), "forest_overall.png", "image/png")
    plt.close(fig)

# ── 4 Forest (Subgroup) ─────────────────────────────────────────────────────────
with tabs[4]:
    st.subheader("Forest Plot — Subgroup")
    if not group_cols:
        st.info("Select at least one Group by column in the sidebar.")
    else:
        for gc in group_cols:
            with st.expander(f"Grouped by: **{gc}**", expanded=(gc == group_cols[0])):
                n_groups_sg = analysis_df[gc].astype(str).nunique()
                if n_groups_sg < 2:
                    st.info(f"'{gc}' has only {n_groups_sg} unique value — choose a column with 2+ groups.")
                else:
                    fig_sg = forest_subgroup(analysis_df, res, gc)
                    st.pyplot(fig_sg, use_container_width=False)
                    st.download_button(
                        f"Download forest ({gc})",
                        fig_png(fig_sg),
                        f"forest_subgroup_{sanitize(gc)}.png",
                        "image/png",
                        key=f"dl_forest_{gc}",
                    )
                    plt.close(fig_sg)

# ── 5 Funnel Plot ───────────────────────────────────────────────────────────────
with tabs[5]:
    st.subheader("Funnel Plot")
    color_options = ["None (plain)"] + group_cols
    col_opt = st.radio("Color points by", color_options, horizontal=True, key="funnel_color")
    gc_funnel = col_opt if col_opt != "None (plain)" else None
    fig = funnel_plot(res, analysis_df, group_col=gc_funnel)
    st.pyplot(fig, use_container_width=False)
    st.download_button("Download funnel plot", fig_png(fig), "funnel.png", "image/png")
    plt.close(fig)

# ── 6 Publication Bias ──────────────────────────────────────────────────────────
with tabs[6]:
    st.subheader("Publication Bias Assessment")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("#### Egger's Test (Overall)")
        if len(res["yi"]) >= 3:
            eg = egger_test(res["yi"], res["vi"])
            if eg:
                st.dataframe(pd.DataFrame({
                    "Statistic": ["Intercept","SE (intercept)","t-value","p-value"],
                    "Value": [f"{eg['intercept']:.{PREC}f}", f"{eg['se_int']:.{PREC}f}",
                              f"{eg['t']:.{PREC}f}", f"{eg['pval']:.{PREC}f}"],
                }), use_container_width=True, hide_index=True)
                if eg["pval"] < 0.10:
                    st.warning("Egger p < 0.10 — possible funnel asymmetry.")
                else:
                    st.success("Egger p >= 0.10 — no significant asymmetry.")
        else:
            st.info("Egger's test requires >= 3 studies.")
    with col_r:
        st.markdown("#### Begg-Mazumdar Rank Correlation")
        if len(res["yi"]) >= 3:
            bg = begg_test(res["yi"], res["vi"])
            st.dataframe(pd.DataFrame({
                "Statistic": ["Kendall's tau","p-value"],
                "Value": [f"{bg['tau']:.{PREC}f}", f"{bg['pval']:.{PREC}f}"],
            }), use_container_width=True, hide_index=True)
            if bg["pval"] < 0.10:
                st.warning("Begg p < 0.10 — significant rank correlation.")
            else:
                st.success("Begg p >= 0.10 — no significant rank correlation.")
        else:
            st.info("Begg's test requires >= 3 studies.")

    # ── Trim-and-Fill correction ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Trim-and-Fill Correction (Duval & Tweedie, 2000)")

    bias_confirmed = False
    if len(res["yi"]) >= 3:
        _eg = egger_test(res["yi"], res["vi"])
        _bg = begg_test(res["yi"], res["vi"])
        egger_sig = _eg is not None and _eg["pval"] < 0.10
        begg_sig  = _bg["pval"] < 0.10
        bias_confirmed = egger_sig or begg_sig

    if len(res["yi"]) < 3:
        st.info("Trim-and-fill requires >= 3 studies.")
    elif not bias_confirmed:
        st.success(
            "Neither Egger's nor Begg's test detected significant asymmetry (p >= 0.10). "
            "Trim-and-fill not indicated; original pooled estimate is unchanged."
        )
        tf_estimator = st.selectbox("Run anyway with estimator", ["L0", "R0"],
                                    key="tf_est_optional")
        if st.button("Run Trim-and-Fill (optional)", key="tf_btn_optional"):
            tf = trim_and_fill(res["yi"], res["vi"], estimator=tf_estimator)
            st.session_state["tf_result"] = tf
    else:
        which = []
        if egger_sig: which.append("Egger's")
        if begg_sig:  which.append("Begg's")
        st.warning(
            f"Statistically significant publication bias detected ({' and '.join(which)} p < 0.10). "
            "Trim-and-fill correction applied below."
        )
        tf_estimator = st.selectbox("Estimator", ["L0", "R0"], key="tf_est")
        tf = trim_and_fill(res["yi"], res["vi"], estimator=tf_estimator)
        st.session_state["tf_result"] = tf

    tf = st.session_state.get("tf_result")
    if tf is not None:
        k0 = tf["k0"]
        if k0 == 0:
            st.info("Trim-and-fill found no asymmetry to correct (k0 = 0). Estimates unchanged.")
        else:
            st.markdown(f"**Studies imputed (k0):** {k0}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Adjusted proportion",
                  f"{tf['p_adj']:.{PREC}f}",
                  delta=f"{tf['p_adj'] - expit(res['mu']):.{PREC}f} vs. original")
        c2.metric("95% CI (lower)", f"{tf['p_lo']:.{PREC}f}")
        c3.metric("95% CI (upper)", f"{tf['p_hi']:.{PREC}f}")

        orig_p = expit(res["mu"])
        orig_lo = expit(res["ci_lo"]); orig_hi = expit(res["ci_hi"])
        st.dataframe(pd.DataFrame({
            "Estimate":       ["Original (RE)",           "Trim-and-Fill adjusted"],
            "Proportion":     [f"{orig_p:.{PREC}f}",     f"{tf['p_adj']:.{PREC}f}"],
            "95% CI lower":   [f"{orig_lo:.{PREC}f}",    f"{tf['p_lo']:.{PREC}f}"],
            "95% CI upper":   [f"{orig_hi:.{PREC}f}",    f"{tf['p_hi']:.{PREC}f}"],
            "Studies (k)":    [str(len(res["yi"])),       str(len(res["yi"]) + k0)],
        }), use_container_width=True, hide_index=True)

        fig_tf = funnel_plot_trimfill(res, tf, title="Funnel Plot — Trim-and-Fill Adjusted")
        st.pyplot(fig_tf, use_container_width=False)
        st.download_button("Download adjusted funnel plot", fig_png(fig_tf),
                           "funnel_trimfill.png", "image/png", key="dl_tf_funnel")
        plt.close(fig_tf)

    st.markdown("---")
    st.subheader("Funnel Plot — Overall")
    fig = funnel_plot(res, analysis_df,
                      group_col=group_col if group_col else None,
                      title="Funnel Plot (Overall)")
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    if group_cols:
        st.markdown("---")
        for gc in group_cols:
            st.subheader(f"Funnel Plots — By Group  ({gc})")
            n_g = analysis_df[gc].astype(str).nunique()
            if n_g >= 2:
                fig_sf = subgroup_funnel_grid(analysis_df, gc)
                st.pyplot(fig_sf, use_container_width=False)
                st.download_button(
                    f"Download subgroup funnels ({gc})",
                    fig_png(fig_sf),
                    f"funnel_subgroups_{sanitize(gc)}.png",
                    "image/png",
                    key=f"dl_funnel_{gc}",
                )
                plt.close(fig_sf)
            else:
                st.info(f"Need >= 2 groups. '{gc}' has only {n_g}.")

# ── 7 Meta-Regression ───────────────────────────────────────────────────────────────────
with tabs[7]:
    st.subheader("Meta-Regression")
    skip = {"Study_ID","Sample","Cases"}
    mod_cols = [c for c in analysis_df.columns if c not in skip]
    if not mod_cols:
        st.info("No moderator columns available.")
    else:
        moderator = st.selectbox("Moderator variable", mod_cols)
        if st.button("Run Meta-Regression", type="primary"):
            ct, sd, fig_mr = meta_regression(analysis_df, res, moderator)
            if ct is None:
                st.error("Regression failed — collinearity or insufficient data.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Residual Q",      f"{sd['Q_res']:.2f}")
                c2.metric("p(residual het)", f"{sd['pval_Qres']:.4f}")
                r2_str = f"{sd['R2']:.1f}%" if not np.isnan(sd["R2"]) else "N/A"
                c3.metric("R2 analog", r2_str)
                st.markdown("#### Coefficients (logit scale)")
                st.dataframe(ct, use_container_width=True, hide_index=True)
                st.caption("Estimates on logit scale. Positive = higher proportion.")
                st.pyplot(fig_mr, use_container_width=False)
                st.download_button("Download plot", fig_png(fig_mr),
                                   "meta_regression.png", "image/png")
                plt.close(fig_mr)

# ── 8 Correlations ──────────────────────────────────────────────────────────────────────────
with tabs[8]:
    st.subheader("Correlation Analysis")
    num_cols = [c for c in analysis_df.columns
                if c not in {"Study_ID"}
                and pd.to_numeric(analysis_df[c], errors="coerce").notna().sum() >= 3]
    if len(num_cols) < 2:
        st.info("Need >= 2 numeric columns.")
    else:
        sel_cols = st.multiselect("Select columns", num_cols,
                                   default=num_cols[:min(6, len(num_cols))])
        method = st.radio("Method", ["pearson","spearman"], horizontal=True)
        if len(sel_cols) >= 2:
            fig_c = correlation_heatmap(analysis_df, sel_cols, method)
            if fig_c:
                st.pyplot(fig_c, use_container_width=False)
                st.download_button("Download heatmap", fig_png(fig_c),
                                   "correlations.png", "image/png")
                plt.close(fig_c)
            st.markdown("#### Pairwise Scatter Plots")
            pairs = [(sel_cols[i], sel_cols[j])
                     for i in range(len(sel_cols)) for j in range(i+1, len(sel_cols))]
            if 1 <= len(pairs) <= 15:
                nc_s = min(3, len(pairs)); nr_s = (len(pairs)+nc_s-1)//nc_s
                fig_s, axes_s = plt.subplots(nr_s, nc_s, figsize=(5.5*nc_s, 4.2*nr_s), squeeze=False)
                for idx, (cx, cy) in enumerate(pairs):
                    ax = axes_s[idx//nc_s][idx%nc_s]
                    xv = pd.to_numeric(analysis_df[cx], errors="coerce")
                    yv = pd.to_numeric(analysis_df[cy], errors="coerce")
                    ok = xv.notna() & yv.notna()
                    ax.scatter(xv[ok], yv[ok], color=CB["blue"], s=48, alpha=0.75,
                               edgecolors=CB["black"], lw=0.3)
                    if ok.sum() >= 2:
                        m_, b_ = np.polyfit(xv[ok], yv[ok], 1)
                        xf = np.linspace(xv[ok].min(), xv[ok].max(), 200)
                        ax.plot(xf, m_*xf+b_, color=CB["vermil"], lw=1.5)
                        r_, p_ = stats.pearsonr(xv[ok], yv[ok])
                        ax.set_title(f"{cx} vs {cy}\nr={r_:.2f}  p={p_:.3f}", fontsize=9.5, pad=4)
                    ax.set_xlabel(cx, fontsize=9); ax.set_ylabel(cy, fontsize=9)
                    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                for idx in range(len(pairs), nr_s*nc_s):
                    axes_s[idx//nc_s][idx%nc_s].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=False)
                plt.close(fig_s)
        else:
            st.info("Select at least 2 columns.")
