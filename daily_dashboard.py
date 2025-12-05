# streamlit run daily_dashboard.py
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import glob, os, math
import logging


st.set_page_config(page_title="月別ドリルダウン（日次アノマリー）", layout="wide")

# =========================
# Utilities
# =========================
WEEKDAY_MAP = {0:"月",1:"火",2:"水",3:"木",4:"金",5:"土",6:"日"}
MONTH_MAP   = {i: f"{i}月" for i in range(1,13)}

def normalize_columns(df):
    cols_lower = {c.lower().strip(): c for c in df.columns}
    rename = {}
    for key in ["time","open","high","low","close"]:
        if key in cols_lower:
            rename[cols_lower[key]] = key
    return df.rename(columns=rename)

def winsorize(s, pct=0.01):
    if s.empty or pct<=0: 
        return s
    low = s.quantile(pct)
    high = s.quantile(1-pct)
    return s.clip(lower=low, upper=high)

@st.cache_data
def load_data(pattern):
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df = normalize_columns(df)
            need = {"time","open","high","low","close"}
            if not need.issubset(set(df.columns)):
                continue
            df["time"] = pd.to_datetime(df["time"])
            df["source"] = os.path.basename(f)
            dfs.append(df[["time","open","high","low","close","source"]])
        except Exception:
            continue
    if not dfs:
        return pd.DataFrame()
    data = pd.concat(dfs, ignore_index=True)
    data = data.sort_values(["source","time"])
    # Base returns
    data["ret_oc"] = data["close"]/data["open"] - 1.0
    # Close-to-close (by file/source)
    data["prev_close"] = data.groupby("source")["close"].shift(1)
    data["ret_cc"] = data["close"]/data["prev_close"] - 1.0
    data["year"] = data["time"].dt.year
    data["month"] = data["time"].dt.month
    data["day"] = data["time"].dt.day
    data["weekday"] = data["time"].dt.weekday
    # Range %
    data["range_pct"] = (data["high"] - data["low"]) / data["open"]
    return data

def add_trading_day_index(df_month):
    df_month = df_month.copy().sort_values("time")
    df_month["ym"] = df_month["time"].dt.to_period("M")
    df_month["tdi"] = df_month.groupby("ym").cumcount()+1
    return df_month

def lag1_autocorr(s):
    s = s.dropna()
    if len(s) < 2: return np.nan
    return pd.Series(s).autocorr(lag=1)

def ci95_mean(mean, std, n):
    if n<=1 or math.isnan(std): return (np.nan, np.nan)
    se = std / math.sqrt(n)
    m = 1.96 * se
    return (mean - m, mean + m)

def compute_month_rank_context(data, ret_col, winsor_pct):
    use = data.copy()
    use["ret"] = use[ret_col]
    if winsor_pct>0:
        use["ret"] = use.groupby(["source","month"])["ret"].transform(lambda s: winsorize(s, winsor_pct))
    g = use.groupby("month")["ret"].agg(
        win_rate = lambda x: (x>0).mean()*100.0,
        avg = lambda x: x.mean()*100.0,
        std = lambda x: x.std(ddof=1)*100.0,
        avg_up = lambda x: (x[x>0].mean()*100.0) if (x>0).any() else np.nan,
        avg_down = lambda x: (x[x<0].mean()*100.0) if (x<0).any() else np.nan,
    ).reset_index()
    g["pl_ratio"] = g["avg_up"] / g["avg_down"].abs()
    ranks = {
        "win_rate_rank": g["win_rate"].rank(ascending=False, method="min"),
        "avg_rank": g["avg"].rank(ascending=False, method="min"),
        "pl_ratio_rank": g["pl_ratio"].rank(ascending=False, method="min"),
        "std_rank": g["std"].rank(ascending=True, method="min"),
    }
    for k,v in ranks.items():
        g[k]=v
    g["month_name"] = g["month"].map(MONTH_MAP)
    return g

def detect_streaks_in_month(df_month, ret_col):
    rows = []
    for y, grp in df_month.groupby("year"):
        r = grp.sort_values("time")[["time", ret_col]].rename(columns={ret_col:"ret"}).reset_index(drop=True)
        prev_dir = 0; start_idx=None; length=0
        for i, rv in enumerate(r["ret"].tolist()):
            d = 1 if rv>0 else (-1 if rv<0 else 0)
            if d==0:
                if length>0 and prev_dir!=0:
                    seg = r.iloc[start_idx:i]
                    cum = (1.0+seg["ret"]).prod()-1.0
                    rows.append({"year":y, "direction":"up" if prev_dir==1 else "down", "length":length,
                                 "start":seg.iloc[0]["time"],"end":seg.iloc[-1]["time"],"cum_ret_pct":cum*100.0})
                prev_dir=0; start_idx=None; length=0
                continue
            if d==prev_dir:
                length+=1
            else:
                if length>0 and prev_dir!=0:
                    seg = r.iloc[start_idx:i]
                    cum = (1.0+seg["ret"]).prod()-1.0
                    rows.append({"year":y,"direction":"up" if prev_dir==1 else "down","length":length,
                                 "start":seg.iloc[0]["time"],"end":seg.iloc[-1]["time"],"cum_ret_pct":cum*100.0})
                prev_dir=d; start_idx=i; length=1
        if length>0 and prev_dir!=0:
            seg = r.iloc[start_idx:len(r)]
            cum = (1.0+seg["ret"]).prod()-1.0
            rows.append({"year":y,"direction":"up" if prev_dir==1 else "down","length":length,
                         "start":seg.iloc[0]["time"],"end":seg.iloc[-1]["time"],"cum_ret_pct":cum*100.0})
    return pd.DataFrame(rows)

def add_zero_axes(fig, xs, ys):
    # 0%線（x,y両方）
    if len(xs)==0 or len(ys)==0: 
        return fig
    x_min, x_max = float(np.nanmin(xs)), float(np.nanmax(xs))
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    fig.add_shape(type="line", x0=0, x1=0, y0=y_min, y1=y_max, line=dict(color="gray", width=1))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=0, y1=0, line=dict(color="gray", width=1))
    return fig

def plot_grid(figs, cols=2):
    if not figs: 
        return
    rows = math.ceil(len(figs)/cols)
    idx = 0
    for _ in range(rows):
        col_objs = st.columns(cols)
        for j in range(cols):
            if idx < len(figs):
                with col_objs[j]:
                    st.plotly_chart(figs[idx], use_container_width=True)
                    idx += 1

# =========================
# UI Controls
# =========================
from pathlib import Path

st.title("📊 月別ドリルダウン（日次アノマリー）")
DATA_FILENAME = Path(__file__).parent/'data_day/*.csv'

pattern = st.text_input("CSVのパス（ワイルドカード可）", value=DATA_FILENAME)

data = load_data(pattern)
if data.empty:
    st.warning("CSVが見つからないか、列名が不足しています（必要: time, open, high, low, close）")
    st.stop()

symbols = sorted(data["source"].unique().tolist())
c1, c2, c3, c4 = st.columns([2,1,2,2])
with c1:
    symbol = st.selectbox("ファイル（銘柄/指数）", symbols)
    df_all = data[data["source"]==symbol].copy().sort_values("time")  # 銘柄固定
with c2:
    month_sel = st.selectbox("対象月", options=list(range(1,13)), format_func=lambda m: f"{m}月")
    years = sorted(df_all["year"].dropna().unique().tolist())
with c3:
    year_min, year_max = min(years), max(years)
    year_range = st.slider("年レンジ", min_value=int(year_min), max_value=int(year_max), value=(int(year_min), int(year_max)), step=1)
with c4:
    ret_type = st.radio("リターン種別", options=["OC","CC"], index=0, horizontal=True)

wins = st.selectbox("Winsorize（外れ値抑制）", options=["なし","1%","2.5%"], index=0)
wins_pct = 0.0 if wins=="なし" else (0.01 if wins=="1%" else 0.025)

# Prepare ret column
ret_col = "ret_oc" if ret_type=="OC" else "ret_cc"

# Filter by year range first
df_all = df_all[(df_all["year"]>=year_range[0]) & (df_all["year"]<=year_range[1])].copy()

# Rank context vs other months (mini cards)
rank_ctx = compute_month_rank_context(df_all, ret_col, wins_pct)

# Now filter to selected calendar month
dfm = df_all[df_all["month"]==month_sel].copy().sort_values("time")
dfm["ret"] = dfm[ret_col]
if wins_pct>0:
    dfm["ret"] = winsorize(dfm["ret"], wins_pct)

# Add trading-day index within each year-month of selected month
dfm = add_trading_day_index(dfm)

# =========================
# Section A: KPI Cards
# =========================
st.markdown("### A. サマリーKPI（選択月・日次ベース）")
total_days = len(dfm)
up_days = int((dfm["ret"]>0).sum())
down_days = int((dfm["ret"]<0).sum())
win_rate = (up_days/total_days*100.0) if total_days>0 else np.nan

avg_ret = dfm["ret"].mean()*100.0 if total_days>0 else np.nan
std_ret = dfm["ret"].std(ddof=1)*100.0 if total_days>1 else np.nan
median_ret = dfm["ret"].median()*100.0 if total_days>0 else np.nan
avg_up = (dfm.loc[dfm["ret"]>0,"ret"].mean()*100.0) if up_days>0 else np.nan
avg_down = (dfm.loc[dfm["ret"]<0,"ret"].mean()*100.0) if down_days>0 else np.nan
pl_ratio = (avg_up/abs(avg_down)) if (not np.isnan(avg_up) and not np.isnan(avg_down) and avg_down!=0) else np.nan
pos1 = (dfm["ret"]>0.01).mean()*100.0 if total_days>0 else np.nan
neg1 = (dfm["ret"]<-0.01).mean()*100.0 if total_days>0 else np.nan
lag1 = lag1_autocorr(dfm["ret"])

# Ranks from context
this_ctx = rank_ctx[rank_ctx["month"]==month_sel]
wr_rank = int(this_ctx["win_rate_rank"].iloc[0]) if not this_ctx.empty else np.nan
avg_rank = int(this_ctx["avg_rank"].iloc[0]) if not this_ctx.empty else np.nan
pl_rank  = int(this_ctx["pl_ratio_rank"].iloc[0]) if not this_ctx.empty else np.nan
std_rank = int(this_ctx["std_rank"].iloc[0]) if not this_ctx.empty else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("勝率（%）", f"{win_rate:.2f}" if not np.isnan(win_rate) else "—", help="上昇日数/総日数×100")
c2.metric("平均（%）", f"{avg_ret:.2f}" if not np.isnan(avg_ret) else "—", help="平均日次リターン")
c3.metric("上昇平均（%）", f"{avg_up:.2f}" if not np.isnan(avg_up) else "—")
c4.metric("下落平均（%）", f"{avg_down:.2f}" if not np.isnan(avg_down) else "—")
c5.metric("損益比率", f"{pl_ratio:.2f}" if not np.isnan(pl_ratio) else "—", help="上昇平均 ÷ |下落平均|")
c6.metric("標準偏差（%）", f"{std_ret:.2f}" if not np.isnan(std_ret) else "—")

c7, c8, c9, c10 = st.columns(4)
c7.metric(">+1% 率", f"{pos1:.2f}%" if not np.isnan(pos1) else "—")
c8.metric("<-1% 率", f"{neg1:.2f}%" if not np.isnan(neg1) else "—")
c9.metric("lag-1 相関", f"{lag1:.3f}" if not np.isnan(lag1) else "—")
c10.metric("（参考）勝率順位", f"{wr_rank}/12" if not np.isnan(wr_rank) else "—")

st.divider()

# =========================
# A-2: 月別「上昇平均 vs 下落平均」（横並び）
# =========================
st.markdown("### A-2. 月別 上昇平均/下落平均（%）— 横にずらして表示")
if not rank_ctx.empty:
    g = rank_ctx.sort_values("month")
    fig_updown = go.Figure()
    fig_updown.add_trace(go.Bar(x=g["month_name"], y=g["avg_up"],   name="avg_up_return",   offsetgroup="up"))
    fig_updown.add_trace(go.Bar(x=g["month_name"], y=g["avg_down"], name="avg_down_return", offsetgroup="down"))
    fig_updown.update_yaxes(zeroline=True, zerolinewidth=2, title="リターン（%）")
    fig_updown.update_xaxes(title="月")
    fig_updown.update_layout(barmode="group", bargap=0.15, legend_title="種類")
    st.plotly_chart(fig_updown, use_container_width=True)
else:
    st.info("十分なデータがありません。")
st.divider()

# =========================
# Section B: Distribution & CI
# =========================
st.markdown("### B. 分布・信頼区間")
colL, colR = st.columns(2)
with colL:
    fig = px.histogram((dfm["ret"]*100.0).dropna(), nbins=40, title=f"{MONTH_MAP[month_sel]} 日次リターン分布（%）")
    st.plotly_chart(fig, use_container_width=True)
with colR:
    fig = px.box((dfm["ret"]*100.0).dropna(), points="outliers", title=f"{MONTH_MAP[month_sel]} 箱ひげ（%）")
    st.plotly_chart(fig, use_container_width=True)

mean = dfm["ret"].mean()*100.0 if total_days else np.nan
std = dfm["ret"].std(ddof=1)*100.0 if total_days>1 else np.nan
ci_lo, ci_hi = ci95_mean(mean, std, total_days) if total_days>1 else (np.nan, np.nan)
quantiles = dfm["ret"].quantile([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99])*100.0 if total_days else pd.Series(dtype=float)

stats_df = pd.DataFrame({
    "件数":[total_days],
    "平均%":[mean],
    "標準偏差%":[std],
    "95%CI%_下限":[ci_lo],
    "95%CI%_上限":[ci_hi],
    "中央値%":[median_ret],
    "歪度":[dfm['ret'].skew() if total_days>2 else np.nan],
    "尖度":[dfm['ret'].kurtosis() if total_days>3 else np.nan],
})
st.dataframe(stats_df.round(3), use_container_width=True)

qdf = quantiles.reset_index()
qdf.columns = ["分位点","%"]
st.dataframe(qdf.round(3), use_container_width=True)

st.divider()

# =========================
# Section C: Extremes (when)
# =========================
st.markdown("### C. 極端値（いつ起こったか）")
if total_days>0:
    idx_max = dfm["ret"].idxmax()
    idx_min = dfm["ret"].idxmin()
    row_max = dfm.loc[idx_max]
    row_min = dfm.loc[idx_min]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("最大上昇日")
        st.write(f"日付: {row_max['time'].date()}  リターン: {row_max['ret']*100:.2f}%")
        st.write(f"OHLC: O={row_max['open']}  H={row_max['high']}  L={row_max['low']}  C={row_max['close']}")
    with c2:
        st.subheader("最大下落日")
        st.write(f"日付: {row_min['time'].date()}  リターン: {row_min['ret']*100:.2f}%")
        st.write(f"OHLC: O={row_min['open']}  H={row_min['high']}  L={row_min['low']}  C={row_min['close']}")

    # Top/Bottom 5
    top5 = dfm.nlargest(5, "ret")[["time","open","high","low","close","ret","range_pct"]].copy()
    bot5 = dfm.nsmallest(5, "ret")[["time","open","high","low","close","ret","range_pct"]].copy()
    top5["date"] = top5["time"].dt.date
    bot5["date"] = bot5["time"].dt.date
    top5["ret%"] = top5["ret"]*100.0
    bot5["ret%"] = bot5["ret"]*100.0
    top5["range%"] = top5["range_pct"]*100.0
    bot5["range%"] = bot5["range_pct"]*100.0
    st.markdown("**Top 5 上昇日**")
    st.dataframe(top5[["date","ret%","open","high","low","close","range%"]].round(2), use_container_width=True)
    st.markdown("**Bottom 5 下落日**")
    st.dataframe(bot5[["date","ret%","open","high","low","close","range%"]].round(2), use_container_width=True)
else:
    st.info("データがありません。")

st.divider()

# =========================
# Section D: Streaks
# =========================
st.markdown("### D. 連続上昇/下落（ストリーク：月内のみ）")
if total_days>0:
    streaks = detect_streaks_in_month(dfm, "ret")
    if streaks.empty:
        st.info("ストリークが検出されませんでした。")
    else:
        ups = streaks[streaks["direction"]=="up"]
        downs = streaks[streaks["direction"]=="down"]
        c1, c2 = st.columns(2)
        with c1:
            if not ups.empty:
                up_len = ups["length"].max()
                cand = ups[ups["length"]==up_len].sort_values("cum_ret_pct", ascending=False).iloc[0]
                st.write(f"**最長連続上昇:** {int(cand['length'])}日  合計{cand['cum_ret_pct']:.2f}%  期間: {pd.to_datetime(cand['start']).date()} → {pd.to_datetime(cand['end']).date()}")
            else:
                st.write("上昇ストリークなし")
        with c2:
            if not downs.empty:
                down_len = downs["length"].max()
                cand = downs[downs["length"]==down_len].sort_values("cum_ret_pct").iloc[0]
                st.write(f"**最長連続下落:** {int(cand['length'])}日  合計{cand['cum_ret_pct']:.2f}%  期間: {pd.to_datetime(cand['start']).date()} → {pd.to_datetime(cand['end']).date()}")
            else:
                st.write("下落ストリークなし")

        st.markdown("**ストリーク長 × 平均合計リターン（%）／件数**")
        max_len = int(st.number_input("最大ストリーク長（分析対象）", min_value=2, max_value=20, value=5, step=1))
        lens = list(range(2, max_len+1))
        mode = st.radio("方向", ["上昇","下落"], index=0, horizontal=True)
        dir_key = "up" if mode=="上昇" else "down"
        s = streaks[streaks["direction"]==dir_key]
        s = s[s["length"].isin(lens)]
        if s.empty:
            st.info("該当ストリークがありません。")
        else:
            agg = s.groupby("length").agg(
                count=("cum_ret_pct","count"),
                avg_cum=("cum_ret_pct","mean"),
                med_cum=("cum_ret_pct","median")
            ).reset_index()
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(agg, x="length", y="avg_cum", title=f"{mode}：ストリーク長ごとの**平均合計リターン（%）**")
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.bar(agg, x="length", y="count", title=f"{mode}：ストリーク長ごとの**件数**")
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(agg.round(2), use_container_width=True)
else:
    st.info("データがありません。")

st.divider()

# =========================
# Section E: 月内の位置（取引日順・曜日）
# =========================
st.markdown("### E. 月内の位置（取引日順・曜日）")
if total_days>0:
    # Trading day index (tdi)
    g_tdi = dfm.groupby("tdi")["ret"].agg(
        days="count",
        win_rate=lambda x: (x>0).mean()*100.0,
        avg=lambda x: x.mean()*100.0,
        std=lambda x: x.std(ddof=1)*100.0,
        avg_up=lambda x: (x[x>0].mean()*100.0) if (x>0).any() else np.nan,
        avg_down=lambda x: (x[x<0].mean()*100.0) if (x<0).any() else np.nan,
    ).reset_index()

    # 第n営業日の平均リターン
    fig = px.bar(g_tdi, x="tdi", y="avg", title="第n営業日の平均リターン（%）")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(g_tdi.round(2), use_container_width=True)

    # 追加：第n営業日「上昇平均 vs 下落平均」(横並び)
    st.markdown("**第n営業日：上昇平均/下落平均（%）**")
    fig_tdi_updown = go.Figure()
    fig_tdi_updown.add_trace(go.Bar(x=g_tdi["tdi"], y=g_tdi["avg_up"],  name="avg_up_return",   offsetgroup="up"))
    fig_tdi_updown.add_trace(go.Bar(x=g_tdi["tdi"], y=g_tdi["avg_down"], name="avg_down_return", offsetgroup="down"))
    fig_tdi_updown.update_yaxes(zeroline=True, zerolinewidth=2, title="リターン（%）")
    fig_tdi_updown.update_xaxes(title="第n営業日")
    fig_tdi_updown.update_layout(barmode="group", bargap=0.15, legend_title="種類")
    st.plotly_chart(fig_tdi_updown, use_container_width=True)

    # Weekday table (only inside selected month)
    dfm["weekday_name"] = dfm["weekday"].map(WEEKDAY_MAP)
    g_wd = dfm.groupby("weekday_name")["ret"].agg(
        days="count",
        up_days=lambda x: int((x>0).sum() ),
        down_days=lambda x: int((x<0).sum() ),
        win_rate=lambda x: (x>0).mean()*100.0,
        avg=lambda x: x.mean()*100.0,
        avg_up=lambda x: (x[x>0].mean()*100.0) if (x>0).any() else np.nan,
        avg_down=lambda x: (x[x<0].mean()*100.0) if (x<0).any() else np.nan,
    ).reset_index()
    g_wd["pl_ratio"] = g_wd["avg_up"] / g_wd["avg_down"].abs()

    # 曜日別 勝率/平均
    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g_wd, x="weekday_name", y="win_rate", title="曜日別 勝率（%）")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(g_wd, x="weekday_name", y="avg", title="曜日別 平均リターン（%）")
        st.plotly_chart(fig, use_container_width=True)

    # 追加：曜日別「上昇平均 vs 下落平均」（横並び）
    st.markdown("**曜日別：上昇平均/下落平均（%）**")
    fig_wd_updown = go.Figure()
    fig_wd_updown.add_trace(go.Bar(x=g_wd["weekday_name"], y=g_wd["avg_up"],   name="avg_up_return",   offsetgroup="up"))
    fig_wd_updown.add_trace(go.Bar(x=g_wd["weekday_name"], y=g_wd["avg_down"], name="avg_down_return", offsetgroup="down"))
    fig_wd_updown.update_yaxes(zeroline=True, zerolinewidth=2, title="リターン（%）")
    fig_wd_updown.update_xaxes(title="曜日")
    fig_wd_updown.update_layout(barmode="group", bargap=0.15, legend_title="種類")
    st.plotly_chart(fig_wd_updown, use_container_width=True)

    st.dataframe(g_wd.round(2), use_container_width=True)

    # 前半/後半
    first = dfm[dfm["day"]<=10]["ret"]; second = dfm[dfm["day"]>=11]["ret"]
    def summarize_series(x):
        if len(x)==0: return pd.Series({"days":0,"win_rate%":np.nan,"avg%":np.nan,"std%":np.nan})
        return pd.Series({"days":len(x),"win_rate%":(x>0).mean()*100.0,"avg%":x.mean()*100.0,"std%":x.std(ddof=1)*100.0 if len(x)>1 else np.nan})
    comp = pd.DataFrame({"前半":summarize_series(first), "後半":summarize_series(second)})
    st.markdown("**前半（1〜10日） vs 後半（11日〜）**")
    st.dataframe(comp.round(2), use_container_width=True)
else:
    st.info("データがありません。")

st.divider()

# =========================
# Section F: Volatility & Range
# =========================
st.markdown("### F. ボラティリティ / レンジ")
if total_days>0:
    fig = px.histogram((dfm["range_pct"]*100.0).dropna(), nbins=40, title="レンジ分布（(H-L)/O %）")
    st.plotly_chart(fig, use_container_width=True)
    fig = px.scatter(x=dfm["range_pct"]*100.0, y=dfm["ret"]*100.0, labels={"x":"レンジ%","y":"リターン%"},
                     title="リターン vs レンジ（%）")
    st.plotly_chart(fig, use_container_width=True)

# =========================
# F-2: リターン相関（チェックで複数表示 + 追加分析）
# =========================
st.markdown("### F-2. リターン相関（チェックで表示を選択）")
if len(df_all) > 1:
    with st.expander("表示オプション", expanded=True):
        colA, colB, colC = st.columns(3)
        with colA:
            show_lag1   = st.checkbox("当日 vs 前日（日次ラグ1）", True)
            show_lag5   = st.checkbox("当日 vs 前週同曜日（ラグ5営業日）", False)
            show_roll   = st.checkbox("当日 vs 直近N日累積（N日ローリング）", False)
            k = st.slider("N（日）", 2, 30, 5, 1) if show_roll else 5
        with colB:
            show_prevW  = st.checkbox("当日 vs 前週の週次リターン（前週合計）", False)
            show_WoW    = st.checkbox("週次 vs 1週前の週次（WoW）", False)
            show_density= st.checkbox("lag1 密度ヒートマップ", False)
        with colC:
            show_acf    = st.checkbox("ACFバー（±1.96/√N）", False)
            acf_lags    = st.slider("ACFの最大ラグ", 1, 40, 20, 1) if show_acf else 20
            show_trans  = st.checkbox("上/下/ゼロ 遷移行列（ヒートマップ）", False)
            show_bins   = st.checkbox("前日リターン分位別の翌日平均（95%CI付）", False)
            nbins       = st.slider("分位の数", 3, 10, 5, 1) if show_bins else 5

    base_daily = df_all[["time","month",ret_col]].rename(columns={ret_col:"ret"}).sort_values("time").copy()
    figs = []

    if show_lag1:
        df = base_daily.copy()
        df["ret_prev"] = df["ret"].shift(1)
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","ret_prev"])
        xs = df["ret_prev"]*100; ys = df["ret"]*100
        fig = px.scatter(x=xs, y=ys, labels={"x":"前日リターン（%）", "y":"当日リターン（%）"},
                         title="当日リターン vs 前日リターン（%）")
        add_zero_axes(fig, xs, ys)
        fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        figs.append(fig)

    if show_lag5:
        df = base_daily.copy()

        df["ret_prev5"] = df["ret"].shift(5)
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","ret_prev5"])
        xs = df["ret_prev5"]*100; ys = df["ret"]*100
        fig = px.scatter(x=xs, y=ys, labels={"x":"1週間前（同曜日）リターン（%）", "y":"当日リターン（%）"},
                         title="当日リターン vs 1週間前（同曜日）リターン（%）")
        add_zero_axes(fig, xs, ys); fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        figs.append(fig)

    if show_roll:
        df = base_daily.copy()
        df["ret_k_prev"] = (1.0 + df["ret"]).rolling(k).apply(np.prod, raw=True).shift(1) - 1.0
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","ret_k_prev"])
        xs = df["ret_k_prev"]*100; ys = df["ret"]*100
        fig = px.scatter(x=xs, y=ys, labels={"x":f"直近{k}日累積（%）","y":"当日リターン（%）"},
                         title=f"当日リターン vs 直近{k}日累積（%）")
        add_zero_axes(fig, xs, ys); fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        figs.append(fig)


    if show_prevW:
        wk = base_daily.copy()
        logging.warning(wk) 

        wk["week"] = wk["time"].dt.to_period("W-MON")
        weekly = wk.groupby("week")["ret"].apply(lambda s: (1.0+s).prod()-1.0).reset_index(name="week_ret")
        prev_map = weekly.set_index("week")["week_ret"]
        df = wk[["time","ret","week"]].copy()
        df["prev_week"] = df["week"] - 1
        df["prev_week_ret"] = df["prev_week"].map(prev_map)
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","prev_week_ret"])
        xs = df["prev_week_ret"]*100; ys = df["ret"]*100
        fig = px.scatter(x=xs, y=ys, labels={"x":"前週の週次リターン（%）", "y":"当日リターン（%）"},
                         title="当日リターン vs 前週の週次リターン（%）")
        add_zero_axes(fig, xs, ys); fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        figs.append(fig)

    if show_WoW:
        wk = base_daily.copy()
        wk["week"] = wk["time"].dt.to_period("W-MON")
        weekly = wk.groupby("week")["ret"].apply(lambda s: (1.0+s).prod()-1.0).reset_index(name="week_ret")
        weekly = weekly.sort_values("week")
        weekly["prev_week_ret"] = weekly["week_ret"].shift(1)
        weekly["week_start"] = weekly["week"].dt.start_time
        weekly = weekly[weekly["week_start"].dt.month == month_sel].dropna(subset=["week_ret","prev_week_ret"])
        xs = weekly["prev_week_ret"]*100; ys = weekly["week_ret"]*100
        fig = px.scatter(x=xs, y=ys, labels={"x":"前週の週次リターン（%）", "y":"当週の週次リターン（%）"},
                         title="週次 vs 1週前の週次（%）")
        add_zero_axes(fig, xs, ys); fig.update_layout(margin=dict(l=0,r=0,t=40,b=0))
        figs.append(fig)

    if show_density:
        df = base_daily.copy()
        df["ret_prev"] = df["ret"].shift(1)
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","ret_prev"])
        fig = px.density_heatmap(df, x=df["ret_prev"]*100, y=df["ret"]*100,
                                 nbinsx=40, nbinsy=40,
                                 labels={"x":"前日（%）","y":"当日（%）"},
                                 title="当日 vs 前日：密度ヒートマップ")
        figs.append(fig)

    # ACF
    if show_acf:
        s = base_daily["ret"].dropna()
        N = len(s)
        acf_vals = [s.autocorr(lag=k) for k in range(1, acf_lags+1)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(range(1,acf_lags+1)), y=acf_vals, name="ACF"))
        bound = 1.96/np.sqrt(N) if N>0 else np.nan
        fig.add_hline(y= bound, line_dash="dot", line_color="gray")
        fig.add_hline(y=-bound, line_dash="dot", line_color="gray")
        fig.update_layout(title=f"ACF（1〜{acf_lags}）と±1.96/√N", xaxis_title="ラグ", yaxis_title="自己相関")
        figs.append(fig)

    # 遷移行列
    if show_trans:
        df = base_daily.copy()
        sign = np.where(df["ret"]>0,"Up", np.where(df["ret"]<0,"Down","Flat"))
        next_sign = pd.Series(sign).shift(-1)
        tm = pd.crosstab(sign, next_sign, normalize='index').fillna(0.0)*100.0
        fig = px.imshow(tm, text_auto=".1f", aspect="auto",
                        labels=dict(x="翌日", y="前日", color="確率（%）"),
                        title="上/下/ゼロの遷移行列（%）")
        figs.append(fig)

    # 分位別の翌日平均＋CI
    if show_bins:
        df = base_daily.copy()
        df["ret_prev"] = df["ret"].shift(1)
        df = df[df["time"].dt.month == month_sel].dropna(subset=["ret","ret_prev"])
        q = pd.qcut(df["ret_prev"], q=nbins, duplicates="drop")
        g = df.groupby(q)["ret"].agg(['mean','std','count']).reset_index()
        g["mean%"] = g["mean"]*100; g["std%"] = g["std"]*100
        g["ci"] = 1.96*(g["std"]/np.sqrt(g["count"])) * 100
        fig = go.Figure()
        fig.add_trace(go.Bar(x=g["ret_prev"].astype(str), y=g["mean%"],
                             error_y=dict(type='data', array=g["ci"], visible=True),
                             name="翌日平均（%）"))
        fig.update_layout(title="前日リターン分位別の翌日平均（%）と95%CI", xaxis_title="前日リターン分位", yaxis_title="翌日平均（%）")
        figs.append(fig)

    # グリッド表示
    plot_grid(figs, cols=2)
else:
    st.info("十分なデータがありません。")

st.divider()

# =========================
# Section G: 条件付き（この月×この曜日×この日）での累積リターン
# =========================
st.markdown("### G. 条件付き累積リターン（買い持ち vs 条件のみ投資）")
with st.expander("対象期間の選択（チェックで素早く選べます）", expanded=True):
    # 月チェック
    st.markdown("**月**")
    sel_all_m = st.checkbox("すべての月を選択", value=False, key="m_all")
    months_selected = []
    if sel_all_m:
        months_selected = list(range(1,13))
    else:
        rows = [st.columns(6), st.columns(6)]
        for i in range(1,13):
            with rows[0 if i<=6 else 1][(i-1)%6]:
                checked = st.checkbox(f"{i}月", value=(i==month_sel), key=f"m_{i}")
                if checked: months_selected.append(i)
    if not months_selected:
        st.warning("少なくとも1つの月を選択してください。")

    # 曜日チェック
    st.markdown("**曜日**")
    sel_all_w = st.checkbox("すべての曜日を選択", value=True, key="w_all")
    weekdays_selected = []
    if sel_all_w:
        weekdays_selected = list(range(7))
    else:
        cols = st.columns(7)
        for i in range(7):
            with cols[i]:
                checked = st.checkbox(WEEKDAY_MAP[i], value=True, key=f"wd_{i}")
                if checked: weekdays_selected.append(i)

    # 日付範囲（これはスライダのままが早いので据え置き）
    day_min, day_max = st.slider("対象日（1〜31）", 1, 31, (1,31), step=1, key="day_cond")

base_series = st.radio("累積に使うリターン", ["OC","CC"], index=0, horizontal=True, key="ret_cond_type")
ret_cond_col = "ret_oc" if base_series=="OC" else "ret_cc"

if df_all.empty or not months_selected or not weekdays_selected:
    st.info("条件付き集計に利用できるデータがありません。")
else:
    dfc = df_all.sort_values("time").copy()
    r = dfc[ret_cond_col].fillna(0.0)
    buy_hold = (1.0 + r).cumprod()

    cond_mask = (
        dfc["month"].isin(months_selected) &
        dfc["weekday"].isin(weekdays_selected) &
        dfc["day"].between(day_min, day_max)
    )

    # KPI（条件にヒットした日のみ）
    hit = r[cond_mask]
    hit_days = int(cond_mask.sum())
    hit_win = (hit>0).mean()*100.0 if hit_days>0 else np.nan
    hit_avg = hit.mean()*100.0 if hit_days>0 else np.nan
    hit_up  = hit[hit>0].mean()*100.0 if (hit>0).any() else np.nan
    hit_down= hit[hit<0].mean()*100.0 if (hit<0).any() else np.nan
    hit_pl  = (hit_up/abs(hit_down)) if (not np.isnan(hit_up) and not np.isnan(hit_down) and hit_down!=0) else np.nan

    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("条件一致 日数", f"{hit_days}")
    k2.metric("条件 勝率", f"{hit_win:.2f}%" if not np.isnan(hit_win) else "—")
    k3.metric("条件 平均", f"{hit_avg:.2f}%" if not np.isnan(hit_avg) else "—")
    k4.metric("条件 上昇平均", f"{hit_up:.2f}%" if not np.isnan(hit_up) else "—")
    k5.metric("条件 下落平均", f"{hit_down:.2f}%" if not np.isnan(hit_down) else "—")

    # ---- 累積の描画：分離タブ ----
    st.markdown("#### 累積リターン（系列別に表示）")
    tab1, tab2 = st.tabs(["買い持ちのみ（日付軸）", "条件のみ投資（ヒット回数軸）"])

    with tab1:
        cum_df_bh = pd.DataFrame({"time": dfc["time"], "買い持ち": buy_hold})
        fig_bh = px.line(cum_df_bh, x="time", y="買い持ち",
                         labels={"買い持ち":"累積倍率","time":"日付"},
                         title="買い持ち（Buy & Hold）")
        st.plotly_chart(fig_bh, use_container_width=True)

    with tab2:
        # ヒット日のみを連結・X軸はヒット回数
        dfc_cond = dfc.loc[cond_mask, ["time", ret_cond_col]].copy()
        if dfc_cond.empty:
            st.info("条件に一致する営業日がありません。")
        else:
            dfc_cond["hit_idx"] = np.arange(1, len(dfc_cond)+1)
            dfc_cond["cond_cum"] = (1.0 + dfc_cond[ret_cond_col].fillna(0.0)).cumprod()
            fig_cond = px.line(
                dfc_cond,
                x="hit_idx", y="cond_cum",
                markers=True,
                labels={"hit_idx":"ヒット回数", "cond_cum":"累積倍率"},
                title="条件のみ投資（ヒット日のみを連結／回数軸）"
            )
            fig_cond.update_traces(
                customdata=np.stack(
                    [dfc_cond["time"].dt.strftime("%Y-%m-%d"),
                     (dfc_cond[ret_cond_col]*100.0)], axis=-1),
                hovertemplate="回数=%{x}<br>累積=%{y:.4f}<br>日付=%{customdata[0]}<br>日次リターン=%{customdata[1]:.2f}%<extra></extra>"
            )
            st.plotly_chart(fig_cond, use_container_width=True)

    st.caption("※ 条件のみ投資は、選択した月・曜日・日付レンジに一致する**営業日**のみを連結し、対象外期間はX軸から除外（回数軸）します。")

st.caption("※ 列名は time, open, high, low, close が必要です。CCリターンは前日終値を使用。Winsorizeは選択月内のリターンに適用。ストリークは月内（年別）で検出します。")

# -------- 完 --------
