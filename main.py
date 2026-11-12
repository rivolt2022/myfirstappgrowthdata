# =============================================================
# 🌱 우리 학교 극지식물 개체별 생육 차이 분석 대시보드 (2026 2회차 예시)
#  - 1회차 실습 흐름(데이터 점검 → 환경 분석 → 개체별 차이 → 관계 분석
#    → 잘 자란/덜 자란 개체 비교 → 예측 모델 → 결론)을 그대로 웹으로 옮긴 예시
#  - 사이드바에서 CSV 2개(환경·생육결과)를 업로드 (업로드가 없으면 data/ 폴더의 예시 데이터 사용)
#  - 학교마다 열 구성이 달라도(질량·수온·CO2 추가, ph·ec 없음 등) 있는 열만 자동 분석
#  - requirements: streamlit, pandas, numpy, plotly (scipy·statsmodels 불필요)
# =============================================================
import math
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="극지식물 개체별 생육 차이 대시보드", layout="wide", page_icon="🌱")

# ─────────────────────────────────────────────
# 한글 폰트 (Streamlit + Plotly)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"], .stMarkdown, .stTable, .stMetric {
    font-family: 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
}
</style>
""",
    unsafe_allow_html=True,
)
PLOT_FONT = dict(family="Noto Sans KR, Malgun Gothic, Apple SD Gothic Neo, sans-serif")


def style_fig(fig, height=380, title=None):
    fig.update_layout(height=height, font=PLOT_FONT, margin=dict(t=50 if title else 30, b=40, l=40, r=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5))
    if title:
        fig.update_layout(title=title)
    return fig


# ─────────────────────────────────────────────
# 데이터 로딩: NFC 정규화 + "환경"/"생육" 키워드로 인식 (학교명 하드코딩 없음)
# ─────────────────────────────────────────────
def read_csv_safely(src):
    """경로 또는 바이트를 받아 인코딩(utf-8-sig / cp949 / euc-kr)을 자동으로 맞춰 읽기."""
    import io
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return pd.read_csv(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src, encoding="utf-8", encoding_errors="replace")


def school_from_name(name):
    """'OO고_환경데이터.csv' → 'OO고'. 파일명에 '_'와 '환경'/'생육'이 없으면 학교명을 추정하지 않음."""
    name = unicodedata.normalize("NFC", name)
    if "_" not in name or ("환경" not in name and "생육" not in name):
        return None
    prefix = name.split("_")[0].strip()
    return prefix if prefix and "환경" not in prefix and "생육" not in prefix and not prefix.lower().endswith(".csv") else None


@st.cache_data(show_spinner=False)
def read_uploaded(data: bytes):
    return read_csv_safely(data)


@st.cache_data(show_spinner=False)
def load_folder_data():
    """업로드가 없을 때 사용할 기본 예시 데이터(data/ 폴더, 있을 때만)."""
    data_dir = Path(__file__).parent / "data"
    env_df, growth_df, school = None, None, None
    if not data_dir.exists():
        return env_df, growth_df, school
    for f in sorted(data_dir.iterdir()):
        name = unicodedata.normalize("NFC", f.name)
        if not name.lower().endswith(".csv"):
            continue
        if "환경" in name:
            env_df = read_csv_safely(f)
        elif "생육" in name:
            growth_df = read_csv_safely(f)
        else:
            continue
        school = school_from_name(name) or school
    return env_df, growth_df, school


def norm(s):
    return unicodedata.normalize("NFC", str(s)).strip()


def find_column(df, keywords):
    if df is None:
        return None
    for col in df.columns:
        c = norm(col).lower().replace(" ", "")
        for kw in keywords:
            if kw.lower().replace(" ", "") in c:
                return col
    return None


def unit_of(col):
    m = re.search(r"\(([^)]*)\)", norm(col))
    return m.group(1) if m else ""


def label_of(col):
    """'잎 수(장)' → '잎 수', 'temperature' → '온도' 처럼 보기 좋은 이름."""
    base = re.sub(r"\s*\([^)]*\)", "", norm(col)).strip()
    env_names = {"temperature": "온도(℃)", "temp": "온도(℃)", "humidity": "습도(%)", "ph": "pH", "ec": "EC",
                 "water_temp": "수온(℃)", "co2": "CO2(ppm)", "light": "광량", "lux": "조도"}
    return env_names.get(base.lower(), base)


# ─────────────────────────────────────────────
# 통계 헬퍼 (numpy·math만 사용)
# ─────────────────────────────────────────────
def describe(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return dict(N=0, 평균=np.nan, 중앙값=np.nan, 표준편차=np.nan, 분산=np.nan, CV=np.nan, 최소=np.nan, 최대=np.nan)
    sd = s.std(ddof=1) if n > 1 else 0.0  # 1회차 보고서와 같은 표본표준편차
    mean = s.mean()
    return dict(N=n, 평균=mean, 중앙값=s.median(), 표준편차=sd, 분산=sd ** 2,
                CV=(sd / mean * 100) if mean else np.nan, 최소=s.min(), 최대=s.max())


def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = 1.0 / (d if abs(d) > FPMIN else FPMIN)
        c = 1.0 + aa / c; c = c if abs(c) > FPMIN else FPMIN
        de = d * c; h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x))
    return bt * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - bt * _betacf(b, a, 1 - x) / b


def t_pvalue(t, df):
    if df is None or df <= 0 or t is None or np.isnan(t):
        return np.nan
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def welch(a, b):
    a = pd.to_numeric(pd.Series(a), errors="coerce").dropna(); b = pd.to_numeric(pd.Series(b), errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return np.nan
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return t_pvalue(t, df)


def corr_p(x, y):
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    n = len(d)
    if n < 3 or d["x"].std() == 0 or d["y"].std() == 0:
        return np.nan, np.nan, n
    r = d["x"].corr(d["y"])
    if abs(r) >= 1:
        return r, 0.0, n
    return r, t_pvalue(r * math.sqrt((n - 2) / (1 - r * r)), n - 2), n


def iqr_bounds(series):
    s = pd.to_numeric(series, errors="coerce")
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def fit_linear(X, y):
    X1 = np.column_stack([np.asarray(X, dtype=float), np.ones(len(y))])
    coef, *_ = np.linalg.lstsq(X1, np.asarray(y, dtype=float), rcond=None)
    return coef


def predict(coef, X):
    return np.column_stack([np.asarray(X, dtype=float), np.ones(len(X))]) @ coef


def r2_mae(y, p):
    y = np.asarray(y, dtype=float); p = np.asarray(p, dtype=float)
    ss_tot = ((y - y.mean()) ** 2).sum()
    return (1 - ((y - p) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan, float(np.abs(y - p).mean())


def kfold_r2(X, y, k=5, seed=0):
    n = len(y)
    if n < 10:
        return np.nan
    idx = np.random.default_rng(seed).permutation(n)
    preds = np.empty(n)
    for fold in np.array_split(idx, k):
        tr = np.setdiff1d(idx, fold)
        preds[fold] = predict(fit_linear(X[tr], y[tr]), X[fold])
    return r2_mae(y, preds)[0]


def fmt_p(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt(v, d=2):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:,.{d}f}"


def csv_download(df, label, fname, key):
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=fname, mime="text/csv", key=key)


# ─────────────────────────────────────────────
# 데이터 준비
# ─────────────────────────────────────────────
# 1) 사이드바에서 우리 학교 CSV 2개를 업로드 (Streamlit Cloud에 data/ 폴더가 없어도 동작)
st.sidebar.title("📁 데이터 업로드")
st.sidebar.caption("우리 학교 CSV 2개를 올려 주세요. 파일 이름 예: OO고_환경데이터.csv, OO고_생육결과데이터.csv")
up_env = st.sidebar.file_uploader("🌡️ 환경 데이터 CSV", type=["csv"], key="up_env")
up_growth = st.sidebar.file_uploader("🌱 생육 결과 데이터 CSV", type=["csv"], key="up_growth")

with st.spinner("데이터를 불러오는 중..."):
    env_raw = read_uploaded(up_env.getvalue()) if up_env is not None else None
    growth_raw = read_uploaded(up_growth.getvalue()) if up_growth is not None else None
    school = None
    for up in (up_growth, up_env):
        if up is not None and school is None:
            school = school_from_name(up.name)
    data_source = "업로드한 파일"
    if env_raw is None and growth_raw is None:
        env_raw, growth_raw, school = load_folder_data()
        data_source = "기본 예시 데이터 (data/ 폴더)"

if env_raw is None and growth_raw is None:
    st.title("🌱 극지식물 개체별 생육 차이 분석")
    st.info("👈 왼쪽 사이드바에서 우리 학교 **환경 데이터 CSV**와 **생육 결과 데이터 CSV**를 업로드해 주세요. "
            "두 파일 중 하나만 올려도 해당 탭은 바로 볼 수 있습니다.")
    st.stop()
st.sidebar.caption(f"📌 사용 중인 데이터: {data_source}")

school_label = school or "우리 학교"

# 환경 데이터: time 파싱 + 숫자 열 자동 인식
env = None
time_col = None
env_cols = []
if env_raw is not None:
    env = env_raw.copy()
    time_col = find_column(env, ["time", "시간", "일시", "date", "날짜"])
    if time_col is not None:
        env["_dt"] = pd.to_datetime(env[time_col], errors="coerce")
        env = env.dropna(subset=["_dt"]).sort_values("_dt")
    for c in env_raw.columns:
        if c == time_col:
            continue
        num = pd.to_numeric(env[c], errors="coerce")
        if num.notna().sum() > 0:
            env[c] = num
            env_cols.append(c)

# 생육 데이터: 개체번호 + 숫자 생육 항목 자동 인식
growth = None
id_col = None
growth_cols = []
weight_col = None
if growth_raw is not None:
    growth = growth_raw.copy()
    id_col = find_column(growth, ["개체번호", "개체 번호", "번호", "id"])
    for c in growth.columns:
        if c == id_col:
            continue
        num = pd.to_numeric(growth[c], errors="coerce")
        if num.notna().sum() > 0:
            growth[c] = num
            growth_cols.append(c)
    weight_col = next((c for c in growth_cols if "생중량" in norm(c)), growth_cols[-1] if growth_cols else None)
    if id_col is None:
        growth.insert(0, "개체번호", range(1, len(growth) + 1)); id_col = "개체번호"
shoot_col = find_column(growth, ["지상부길이", "지상부 길이"]) if growth is not None else None
root_col = find_column(growth, ["지하부길이", "지하부 길이"]) if growth is not None else None

# ─────────────────────────────────────────────
# 사이드바
# ─────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.title("⚙️ 설정")
team_hypothesis = st.sidebar.text_area(
    "📝 우리 팀의 가설 (1회차 보고서에서 가져오기)",
    value="같은 스마트팜에서 재배해도 개체마다 생중량 차이는 클 것이고, 잎 수가 많은 개체일수록 생중량이 클 것이다.",
    height=110,
)
sel_growth = None
if growth_cols:
    sel_growth = st.sidebar.selectbox("📊 살펴볼 생육 항목", growth_cols, index=growth_cols.index(weight_col),
                                      format_func=label_of)
date_range = None
if env is not None and time_col is not None and not env.empty:
    dmin, dmax = env["_dt"].min().date(), env["_dt"].max().date()
    date_range = st.sidebar.date_input("📅 환경 데이터 기간", value=(dmin, dmax), min_value=dmin, max_value=dmax)

st.title(f"🌱 {school_label} 극지식물 개체별 생육 차이 분석")
st.caption("1회차 보고서(문제 정의 → 분석 방법 → 결과 → 논의)를 누구나 볼 수 있는 웹 대시보드로 만든 예시입니다.")

tabs = st.tabs(["📖 실험 개요", "🌡️ 환경 변화", "📊 개체별 생육", "🔗 관계 분석", "🏆 그룹 비교", "🔮 예측 모델", "📝 결론"])

# ═════════════════════════════════════════════
# 1. 실험 개요 + 데이터 점검  (1회차 (1) 우리 학교 데이터 파악하기)
# ═════════════════════════════════════════════
with tabs[0]:
    st.header("📖 실험 개요")
    st.markdown(
        "같은 에코스마트팜에서 극지식물을 재배한 뒤 개체별로 잎 수·길이·무게를 측정했습니다. "
        "<b>같은 스마트팜에서 자랐는데 왜 개체마다 다르게 자랐을까?</b>를 데이터로 확인합니다.", unsafe_allow_html=True
    )
    st.markdown("**우리 팀의 가설** (사이드바에서 수정)")
    st.success(team_hypothesis)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧪 개체 수", f"{len(growth)}개체" if growth is not None else "—")
    if weight_col is not None:
        c2.metric(f"⚖️ 평균 {label_of(weight_col)}", f"{fmt(growth[weight_col].mean())} {unit_of(weight_col)}")
    tcol = find_column(env, ["temperature", "온도"]) if env is not None else None
    hcol = find_column(env, ["humidity", "습도"]) if env is not None else None
    c3.metric("🌡️ 평균 온도", f"{fmt(env[tcol].mean(), 1)} ℃" if tcol else "—")
    c4.metric("💧 평균 습도", f"{fmt(env[hcol].mean(), 1)} %" if hcol else "—")

    st.subheader("🔍 데이터 점검")
    st.caption("분석 전에 데이터를 믿고 써도 되는지 확인합니다. 결측값은 빈 칸, 이상치는 IQR(가운데 50% 폭) 기준으로 너무 크거나 작은 값입니다. "
               "이상치는 틀린 값이 아닐 수 있으므로 지우지 않고 표시만 합니다.")
    rows = []
    if env is not None:
        period = f"{env['_dt'].min():%Y-%m-%d} ~ {env['_dt'].max():%Y-%m-%d}" if time_col else "—"
        gap = env["_dt"].diff().dropna() if time_col else pd.Series(dtype="timedelta64[ns]")
        interval = "—"
        if not gap.empty:
            mins = int(gap.mode().iloc[0].total_seconds() // 60)
            interval = f"{mins // 60}시간" if mins % 60 == 0 else f"{mins}분"
        rows.append({"데이터": "환경 데이터", "행 수": len(env_raw), "기간 / 간격": f"{period} / {interval}",
                     "측정 항목": ", ".join(label_of(c) for c in env_cols),
                     "결측값": ", ".join(f"{label_of(c)} {int(env[c].isna().sum())}" for c in env_cols if env[c].isna().sum()) or "없음"})
    if growth is not None:
        rows.append({"데이터": "생육 결과 데이터", "행 수": len(growth_raw), "기간 / 간격": "수확 시 1회 측정",
                     "측정 항목": ", ".join(label_of(c) for c in growth_cols),
                     "결측값": ", ".join(f"{label_of(c)} {int(growth[c].isna().sum())}" for c in growth_cols if growth[c].isna().sum()) or "없음"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    if growth is not None:
        ids = pd.to_numeric(growth[id_col], errors="coerce").dropna().astype(int)
        missing = sorted(set(range(ids.min(), ids.max() + 1)) - set(ids)) if not ids.empty else []
        if missing:
            st.caption(f"📌 빠진 개체번호: {', '.join(map(str, missing))}")
        outs = []
        for c in growth_cols:
            lo, hi = iqr_bounds(growth[c])
            who = growth.loc[(growth[c] < lo) | (growth[c] > hi), id_col].astype(str).tolist()
            if who:
                outs.append(f"{label_of(c)}: #{', #'.join(who)}")
        st.caption("📌 IQR 이상치 개체 — " + (" / ".join(outs) if outs else "없음"))
    st.caption("⚠️ 생육 결과 데이터에는 측정일·개체 위치 정보가 없어, 환경 변화나 위치가 개체별 차이에 준 영향은 직접 확인할 수 없습니다.")

# ═════════════════════════════════════════════
# 2. 환경 변화  (1회차 (2) 스마트팜 환경 분석하기)
# ═════════════════════════════════════════════
with tabs[1]:
    st.header("🌡️ 재배 기간 환경 변화")
    if env is None or not env_cols:
        st.info("환경 데이터가 없습니다.")
    else:
        fenv = env
        if time_col and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            fenv = env[(env["_dt"].dt.date >= date_range[0]) & (env["_dt"].dt.date <= date_range[1])]
        if fenv.empty:
            st.warning("선택한 기간에 데이터가 없습니다.")
        else:
            st.subheader("📋 환경 요약 통계")
            st.caption("표준편차가 클수록 값이 많이 흔들렸다는 뜻입니다. CV(변동계수)=표준편차÷평균×100은 평균이 큰 값(온도·pH)에서 작게 나오므로 CV만으로 '안정적'이라고 판단하지 않습니다.")
            def _n_out(c):
                lo, hi = iqr_bounds(fenv[c])
                return int(((fenv[c] < lo) | (fenv[c] > hi)).sum())
            st.dataframe(pd.DataFrame([{"항목": label_of(c), **{k: (v if k == "N" else round(v, 3)) for k, v in describe(fenv[c]).items()},
                                        "이상치(IQR) 개수": _n_out(c)}
                                       for c in env_cols]).rename(columns={"CV": "CV(%)"}), hide_index=True, width="stretch")
            st.caption("이상치(IQR)는 센서 오류일 수도, 실제로 특별한 순간일 수도 있습니다. 지우지 말고 원인을 먼저 생각해 보세요.")
            if any(fenv[c].isna().mean() > 0.05 for c in env_cols):
                st.caption("⚠️ 일부 항목은 측정이 빠진 기간이 있습니다(센서 미측정). 빈 기간은 그래프에서 끊겨 보입니다.")

            if time_col:
                show = st.multiselect("그래프로 볼 환경 항목", env_cols, default=env_cols[:2], format_func=label_of)
                if show:
                    daily = fenv.groupby(fenv["_dt"].dt.date)[show].mean()
                    fig = make_subplots(rows=len(show), cols=1, shared_xaxes=True,
                                        subplot_titles=[f"날짜별 평균 {label_of(c)}" for c in show])
                    for i, c in enumerate(show, 1):
                        fig.add_trace(go.Scatter(x=daily.index, y=daily[c], mode="lines+markers", name=label_of(c)), row=i, col=1)
                    st.plotly_chart(style_fig(fig, height=230 * len(show)).update_layout(showlegend=False), width="stretch")

                    hourly = fenv.groupby(fenv["_dt"].dt.hour)[show].mean()
                    fig_h = make_subplots(rows=1, cols=len(show), subplot_titles=[f"시간대별 평균 {label_of(c)}" for c in show])
                    for i, c in enumerate(show, 1):
                        fig_h.add_trace(go.Scatter(x=hourly.index, y=hourly[c], mode="lines+markers", name=label_of(c)), row=1, col=i)
                        fig_h.update_xaxes(title_text="시(0~23)", dtick=3, row=1, col=i)
                    st.plotly_chart(style_fig(fig_h, height=320).update_layout(showlegend=False), width="stretch")
                    if tcol in show and hourly[tcol].notna().any():
                        st.caption(f"📌 가장 더운 시간대 {int(hourly[tcol].idxmax())}시, 가장 추운 시간대 {int(hourly[tcol].idxmin())}시, "
                                   f"시간대 평균 기준 일교차 {hourly[tcol].max() - hourly[tcol].min():.2f}℃")

                st.subheader("⏳ 전반부 vs 후반부")
                half = len(fenv) // 2
                a, b = fenv.iloc[:half], fenv.iloc[half:]
                st.dataframe(pd.DataFrame([{"항목": label_of(c), "전반부 평균": round(a[c].mean(), 3), "전반부 표준편차": round(a[c].std(), 3),
                                            "후반부 평균": round(b[c].mean(), 3), "후반부 표준편차": round(b[c].std(), 3)} for c in env_cols]),
                             hide_index=True, width="stretch")
                st.caption("측정 시점 순서로 데이터를 반으로 나눠 비교합니다. 차이의 원인(계절·관리 등)은 측정하지 않았으므로 '가능성'으로만 해석합니다.")
            with st.expander("📄 환경 데이터 원본 + CSV 다운로드"):
                st.dataframe(fenv.drop(columns=["_dt"], errors="ignore"), width="stretch", height=280)
                csv_download(fenv.drop(columns=["_dt"], errors="ignore"), "📥 CSV 다운로드", "환경데이터.csv", "dl_env")

# ═════════════════════════════════════════════
# 3. 개체별 생육  (1회차 (3) 개체별 생육 차이 분석하기)
# ═════════════════════════════════════════════
with tabs[2]:
    st.header("📊 개체별 생육 차이")
    if growth is None or not growth_cols:
        st.info("생육 결과 데이터가 없습니다.")
    else:
        w = growth[[id_col, weight_col]].dropna()
        if not w.empty:
            mx, mn = w.loc[w[weight_col].idxmax()], w.loc[w[weight_col].idxmin()]
            k1, k2, k3 = st.columns(3)
            k1.metric(f"🔺 {label_of(weight_col)} 최대 개체", f"#{mx[id_col]:.0f} — {mx[weight_col]:.2f} {unit_of(weight_col)}")
            k2.metric(f"🔻 {label_of(weight_col)} 최소 개체", f"#{mn[id_col]:.0f} — {mn[weight_col]:.2f} {unit_of(weight_col)}")
            k3.metric("📏 최대/최소 배수", f"{mx[weight_col] / mn[weight_col]:.1f}배" if mn[weight_col] > 0 else "—")

        st.subheader("📋 항목별 기초 통계")
        st.caption("중앙값: 크기순 한가운데 값(튀는 값에 덜 흔들림). 분산·표준편차: 평균에서 퍼진 정도. CV: 단위가 달라도 개체 간 차이를 같은 잣대로 비교하는 값.")
        stat_df = pd.DataFrame([{"항목": label_of(c), "단위": unit_of(c), **describe(growth[c])} for c in growth_cols])
        stat_df["최대/최소"] = np.where(stat_df["최소"] > 0, stat_df["최대"] / stat_df["최소"].replace(0, np.nan), np.nan)
        show_df = stat_df.rename(columns={"CV": "CV(%)"}).round(2)
        show_df["최대/최소"] = show_df["최대/최소"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}")  # 최솟값 0이면 배수 계산 불가
        st.dataframe(show_df, hide_index=True, width="stretch")
        top_cv = stat_df.loc[stat_df["CV"].idxmax()]
        st.info(f"👉 개체 간 차이(CV)가 가장 큰 항목: **{top_cv['항목']}** (CV {top_cv['CV']:.1f}%)")

        st.subheader(f"📈 {label_of(sel_growth)} 분포")
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(style_fig(px.histogram(growth, x=sel_growth, nbins=15, color_discrete_sequence=["#2ecc71"])
                                      .update_layout(xaxis_title=label_of(sel_growth), yaxis_title="개체 수"), title="히스토그램"),
                            width="stretch")
        with h2:
            st.plotly_chart(style_fig(px.box(growth, y=sel_growth, points="all", hover_data=[id_col],
                                             color_discrete_sequence=["#3498db"]).update_layout(yaxis_title=label_of(sel_growth)),
                                      title="상자그림 (점 하나 = 개체 하나)"), width="stretch")
        st.caption("히스토그램은 구간별 개체 수, 상자그림은 중앙값·사분위 범위·이상치를 보여줍니다.")

        st.subheader(f"🔢 개체별 {label_of(sel_growth)} 순위")
        rank = growth[[id_col, sel_growth]].dropna().sort_values(sel_growth, ascending=False)
        fig_r = px.bar(rank, x=rank[id_col].astype(str), y=sel_growth, color_discrete_sequence=["#9b59b6"])
        fig_r.add_hline(y=rank[sel_growth].mean(), line_dash="dash", annotation_text="평균")
        st.plotly_chart(style_fig(fig_r.update_layout(xaxis_title="개체번호 (큰 순)", yaxis_title=label_of(sel_growth),
                                                      xaxis_type="category")), width="stretch")
        with st.expander("📄 생육 결과 원본 + CSV 다운로드"):
            st.dataframe(growth_raw, width="stretch", height=280)
            csv_download(growth_raw, "📥 CSV 다운로드", "생육결과데이터.csv", "dl_growth")

# ═════════════════════════════════════════════
# 4. 관계 분석  (1회차 (3) 상관계수)
# ═════════════════════════════════════════════
with tabs[3]:
    st.header("🔗 생육 항목 사이의 관계")
    st.caption("상관계수 r은 −1~1. 1에 가까우면 함께 커지고, 0 근처면 관계가 거의 없습니다. 관계가 있다는 것이 원인이라는 뜻은 아닙니다.")
    if growth is None or len(growth_cols) < 2:
        st.info("관계를 볼 생육 항목이 2개 이상 필요합니다.")
    else:
        corr = growth[growth_cols].corr()
        fig_c = px.imshow(corr.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                          x=[label_of(c) for c in growth_cols], y=[label_of(c) for c in growth_cols])
        st.plotly_chart(style_fig(fig_c, height=420, title="상관계수 히트맵"), width="stretch")

        rows = []
        for c in growth_cols:
            if c == weight_col:
                continue
            r, p, n = corr_p(growth[c], growth[weight_col])
            rows.append({"항목": label_of(c), f"{label_of(weight_col)}와의 r": round(r, 3) if not np.isnan(r) else None, "p값": fmt_p(p), "n": n})
        st.dataframe(pd.DataFrame(rows).sort_values(f"{label_of(weight_col)}와의 r", ascending=False, key=lambda s: s.abs()),
                     hide_index=True, width="stretch")
        st.caption("p값이 0.05보다 작으면 '관계가 없다는 가정에서는 관찰되기 어려운 결과'입니다.")

        x_opts = [c for c in growth_cols if c != weight_col]
        xs = st.selectbox("산점도 X축 항목", x_opts, format_func=label_of)
        d = growth[[id_col, xs, weight_col]].dropna()
        r, p, n = corr_p(d[xs], d[weight_col])
        fig_s = px.scatter(d, x=xs, y=weight_col, hover_data=[id_col], color_discrete_sequence=["#2ecc71"])
        if len(d) >= 3:
            sl, ic = np.polyfit(d[xs], d[weight_col], 1)
            xl = np.linspace(d[xs].min(), d[xs].max(), 50)
            fig_s.add_trace(go.Scatter(x=xl, y=sl * xl + ic, mode="lines", name="추세선", line=dict(dash="dash", color="#e67e22")))
        st.plotly_chart(style_fig(fig_s.update_layout(xaxis_title=label_of(xs), yaxis_title=label_of(weight_col)),
                                  title=f"{label_of(xs)} vs {label_of(weight_col)} (r = {fmt(r, 3)}, p = {fmt_p(p)})"), width="stretch")

# ═════════════════════════════════════════════
# 5. 그룹 비교  (1회차 (4) 잘 자란 개체와 덜 자란 개체 비교하기)
# ═════════════════════════════════════════════
with tabs[4]:
    st.header("🏆 잘 자란 개체 vs 덜 자란 개체")
    if growth is None or weight_col is None:
        st.info("생육 결과 데이터가 없습니다.")
    else:
        basis = st.selectbox("그룹을 나눌 기준 항목", growth_cols, index=growth_cols.index(weight_col), format_func=label_of)
        g = growth.dropna(subset=[basis]).copy()
        q25, q75 = g[basis].quantile(0.25), g[basis].quantile(0.75)
        top, bot = g[g[basis] >= q75], g[g[basis] <= q25]
        st.caption(f"{label_of(basis)} 상위 25%(≥ {q75:.2f}) = Top **{len(top)}개체**, 하위 25%(≤ {q25:.2f}) = Bottom **{len(bot)}개체**")
        if min(len(top), len(bot)) < 5:
            st.warning("그룹별 개체 수가 적어 결과가 불확실합니다. 참고용으로만 해석하세요.")
        comp_cols = [c for c in growth_cols if c != basis]
        rows = []
        for c in comp_cols:
            rows.append({"항목": label_of(c), "Top 평균": top[c].mean(), "Top 표준편차": top[c].std(),
                         "Bottom 평균": bot[c].mean(), "Bottom 표준편차": bot[c].std(),
                         "Top/Bottom 배수": top[c].mean() / bot[c].mean() if bot[c].mean() else np.nan,
                         "p값 (Welch t검정)": fmt_p(welch(top[c], bot[c]))})
        if shoot_col and root_col and basis not in (shoot_col, root_col):
            rt = (top[shoot_col] / top[root_col].where(top[root_col] > 0)).dropna()
            rb = (bot[shoot_col] / bot[root_col].where(bot[root_col] > 0)).dropna()
            rows.append({"항목": "지상부÷지하부 길이 비율", "Top 평균": rt.mean(), "Top 표준편차": rt.std(),
                         "Bottom 평균": rb.mean(), "Bottom 표준편차": rb.std(),
                         "Top/Bottom 배수": rt.mean() / rb.mean() if rb.mean() else np.nan, "p값 (Welch t검정)": fmt_p(welch(rt, rb))})
        tb = pd.DataFrame(rows)
        st.dataframe(tb.round(2), hide_index=True, width="stretch")
        st.caption("p값: 두 그룹 차이가 없다고 가정할 때 이런 차이가 우연히 나올 가능성(0.05 미만이면 우연으로 보기 어려움). "
                   "그룹을 나눈 기준 항목은 검정하지 않습니다. 비율이 1보다 크면 지상부가 지하부보다 길다는 뜻입니다.")
        if not tb.empty:
            fig_tb = make_subplots(rows=1, cols=len(tb), subplot_titles=list(tb["항목"]))
            for i, rr in enumerate(tb.itertuples(), 1):
                fig_tb.add_trace(go.Bar(x=["Top", "Bottom"], y=[rr[2], rr[4]], error_y=dict(array=[rr[3], rr[5]]),
                                        marker_color=["#2ecc71", "#e74c3c"], showlegend=False), row=1, col=i)
            st.plotly_chart(style_fig(fig_tb, height=340, title="그룹별 평균 ± 표준편차 (항목마다 단위가 달라 따로 표시)"), width="stretch")

# ═════════════════════════════════════════════
# 6. 예측 모델  (1회차 (5) 생중량 예측모델 만들기)
# ═════════════════════════════════════════════
with tabs[5]:
    st.header("🔮 예측 모델")
    st.caption("선형회귀: 데이터를 가장 잘 지나는 직선(식)으로 값을 예측. R²(결정계수)는 모델이 차이를 얼마나 설명하는지(0~1), "
               "MAE는 예측이 평균 얼마나 틀렸는지, 5겹 교차검증 R²는 처음 보는 데이터에도 맞는지를 봅니다.")
    if growth is None or len(growth_cols) < 2:
        st.info("예측 모델에는 생육 항목이 2개 이상 필요합니다.")
    else:
        target = st.selectbox("예측할 항목", growth_cols, index=growth_cols.index(weight_col), format_func=label_of)
        cand = [c for c in growth_cols if c != target]
        default = [c for c in cand if "잎" in norm(c)][:1] or cand[:1]
        feats = st.multiselect("예측에 쓸 항목", cand, default=default, format_func=label_of)
        if feats:
            d = growth[feats + [target]].dropna()
            if len(d) < 5:
                st.warning("유효한 개체가 5개 미만이라 모델을 만들 수 없습니다.")
            else:
                X, y = d[feats].to_numpy(float), d[target].to_numpy(float)
                coef = fit_linear(X, y)
                pred = predict(coef, X)
                r2, mae = r2_mae(y, pred)
                cv = kfold_r2(X, y)
                eq = f"{label_of(target)} = " + " ".join(f"{coef[i]:+.4f}×{label_of(f)}" for i, f in enumerate(feats)) + f" {coef[-1]:+.4f}"
                st.code(eq, language=None)
                m1, m2, m3 = st.columns(3)
                m1.metric("R²", fmt(r2, 3)); m2.metric("MAE", f"{fmt(mae)} {unit_of(target)}")
                m3.metric("5겹 교차검증 R²", fmt(cv, 3) if not np.isnan(cv) else "개체 10개 이상 필요")
                st.caption("⚠️ R²만으로 좋은 모델이라고 판단하지 않습니다. 교차검증 R²가 많이 낮으면 일부 개체에 맞춰진 모델일 수 있습니다. "
                           "회귀계수는 관계일 뿐 원인을 뜻하지 않습니다.")
                fig_p = px.scatter(x=y, y=pred, labels={"x": f"실제 {label_of(target)}", "y": f"예측 {label_of(target)}"},
                                   color_discrete_sequence=["#3498db"])
                lim = [float(min(y.min(), pred.min())), float(max(y.max(), pred.max()))]
                fig_p.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="예측 = 실제", line=dict(dash="dash", color="#7f8c8d")))
                st.plotly_chart(style_fig(fig_p, title="실제값 vs 예측값 (점선에 가까울수록 정확)"), width="stretch")

                st.subheader("🧮 직접 넣어 예측해보기 (실제 데이터 범위 안에서만)")
                inputs = []
                cols_in = st.columns(len(feats))
                for i, f in enumerate(feats):
                    lo, hi = float(d[f].min()), float(d[f].max())
                    inputs.append(cols_in[i].slider(label_of(f), lo, hi, float(d[f].median())))
                est = float(predict(coef, np.array([inputs]))[0])
                st.metric(f"예상 {label_of(target)}", f"{est:.2f} {unit_of(target)}")
                if est < 0:
                    st.caption("⚠️ 예측값이 0보다 작습니다. 직선 모델이 작은 개체를 잘 설명하지 못한다는 한계입니다.")

# ═════════════════════════════════════════════
# 7. 결론  (1회차 보고서 '4. 논의'·'결론')
# ═════════════════════════════════════════════
with tabs[6]:
    st.header("📝 결론: 가설 검토와 한계")
    st.markdown("**우리 팀의 가설**")
    st.success(team_hypothesis)
    if growth is not None and weight_col is not None:
        s = describe(growth[weight_col])
        facts = [f"{label_of(weight_col)} 평균 **{s['평균']:.2f}{unit_of(weight_col)}**, 표준편차 **{s['표준편차']:.2f}**, "
                 f"CV **{s['CV']:.1f}%** (개체 {s['N']}개)"]
        if s["최소"] and s["최소"] > 0:
            facts.append(f"가장 무거운 개체는 가장 가벼운 개체의 **{s['최대'] / s['최소']:.1f}배**")
        best = None
        for c in growth_cols:
            if c == weight_col:
                continue
            r, p, n = corr_p(growth[c], growth[weight_col])
            if not np.isnan(r) and (best is None or abs(r) > abs(best[1])):
                best = (c, r, p)
        if best:
            facts.append(f"{label_of(weight_col)}와 관계가 가장 큰 항목은 **{label_of(best[0])}** (r = {best[1]:.2f}, p = {fmt_p(best[2])})")
        st.markdown("**데이터로 확인한 사실**")
        for f in facts:
            st.markdown(f"- {f}")
        st.caption("가설이 맞는지는 위 숫자를 근거로 팀이 직접 판단해 적어 보세요. 가설이 틀려도 실패가 아니라 하나의 결과입니다.")
    st.markdown("**연구의 한계**")
    st.markdown(
        "- 생육 측정일과 개체 위치 정보가 없어, 환경 변화나 위치(빛·바람)가 개체별 차이에 준 영향은 확인할 수 없습니다.\n"
        "- 개체 수가 적으면 결과가 몇 개체에 따라 크게 달라질 수 있습니다.\n"
        "- 상관관계·회귀계수는 관계를 보여줄 뿐, 원인과 결과를 뜻하지 않습니다."
    )
