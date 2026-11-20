import io
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import streamlit as st

# ── 한글 폰트 (Streamlit CSS) ──────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── 열 이름 표준화 ─────────────────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str)
                .str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df


# ── CSV 읽기 (utf-8-sig → cp949 이중 시도) ────────────────────────────────
def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()


# ── 학교명 추출 (파일명에서) ───────────────────────────────────────────────
def school_name(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    m = re.match(r"([^\d_]+)", name)
    return m.group(1).strip() if m else name


# ── @st.cache_data 로 데이터 로드 ──────────────────────────────────────────
@st.cache_data
def load_env(file):
    if file is None:
        return pd.DataFrame()
    return std_cols(read_csv(file))


@st.cache_data
def load_growth(file):
    if file is None:
        return pd.DataFrame()
    return std_cols(read_csv(file))


# ── 지표 카드 한 줄 ────────────────────────────────────────────────────────
def metric_row(items):
    """items: list of (label, value) 또는 (label, value, delta)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        if len(item) == 3:
            col.metric(label=item[0], value=item[1], delta=item[2])
        else:
            col.metric(label=item[0], value=item[1])


# ── 산점도 추세선 추가 (np.polyfit, go.Scatter) ───────────────────────────
def add_trendline(fig, x, y, color="#e74c3c", dash="dash"):
    df_xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(df_xy) < 2:
        return
    m, b = np.polyfit(df_xy["x"], df_xy["y"], 1)
    x_line = np.array([df_xy["x"].min(), df_xy["x"].max()])
    r = df_xy["x"].corr(df_xy["y"])
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=m * x_line + b,
            mode="lines",
            line=dict(color=color, dash=dash, width=2),
            name=f"추세선 (r={r:.2f})",
            hoverinfo="skip",
        )
    )


# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")
st.title("🔬 연송고 극지식물, 무게 차이의 비밀")

# ── 사이드바: CSV 업로드 + 측정 항목 선택 ────────────────────────────────
st.sidebar.header("📁 데이터 업로드")
env_file = st.sidebar.file_uploader(
    "환경 CSV (온도·습도·pH·EC 등)", type="csv"
)
growth_file = st.sidebar.file_uploader(
    "생육 결과 CSV (개체별 측정값)", type="csv"
)

st.sidebar.header("📏 생육 측정 항목")
growth_cols = []
if growth_file is not None:
    _tmp = load_growth(growth_file)
    avail = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in _tmp.columns]
    growth_cols = st.sidebar.multiselect("비교할 측정 항목", options=avail, default=avail)

with st.spinner("데이터를 불러오는 중이에요…"):
    env_df = load_env(env_file)
    growth_df = load_growth(growth_file)

school = ""
if env_file is not None:
    school = school_name(env_file.name)
elif growth_file is not None:
    school = school_name(growth_file.name)

if school:
    st.sidebar.success(f"✅ {school} 데이터 로드 완료")

# ── 탭 ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📋 실험 개요", "🌡️ 환경 데이터", "🌱 생육 결과"])

# ═══════════════════════════════════════════════════════════════════════════
# 탭 1: 실험 개요
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    try:
        st.markdown(
            """
**연구 배경**
극지식물은 추위와 건조에 강한 식물로, 우리 학교 과학실에서 키우고 있어요.
같은 조건에서 키워도 개체마다 무게와 크기에 큰 차이가 나는데, 그 이유를 알아보려고 합니다.

**연구 목적**
1. 개체별 생중량(무게) 차이를 **평균·표준편차·변동계수(CV)** 로 파악해요.
   - *평균*: 모든 값을 더해 개수로 나눈 대표값
   - *표준편차*: 값들이 평균에서 얼마나 퍼져 있는지 나타내는 값
   - *CV(변동계수)*: 표준편차를 평균으로 나눈 값(%) — 평균이 달라도 비교하기 좋아요
2. **지상부 길이**(줄기·잎)와 **지하부 길이**(뿌리) 중 생중량과 더 관련 깊은 쪽을 찾아요.
   - *상관계수(r)*: -1 ~ 1 사이 값, 1에 가까울수록 함께 증가, -1에 가까울수록 반대 방향
3. 생육이 가장 좋은 개체 **TOP 5**와 가장 나쁜 개체 **BOTTOM 5**를 비교해요.
"""
        )

        if growth_df.empty:
            st.info("📌 생육 결과 CSV를 업로드하면 연구 개요 요약이 표시됩니다.")
        else:
            n_individuals = len(growth_df)
            period_str = ""
            if not env_df.empty and "time" in env_df.columns:
                t = env_df["time"].dropna()
                if len(t) > 0:
                    period_str = f"{t.min().strftime('%Y-%m-%d')} ~ {t.max().strftime('%Y-%m-%d')}"

            summary_rows = []
            summary_rows.append({"항목": "학교명", "값": school if school else "파일에서 추출"})
            summary_rows.append({"항목": "개체 수", "값": f"{n_individuals}개체"})
            if period_str:
                summary_rows.append({"항목": "측정 기간", "값": period_str})
            if "잎 수" in growth_df.columns:
                summary_rows.append(
                    {"항목": "잎 수 (평균)", "값": f"{growth_df['잎 수'].dropna().mean():.1f}장"}
                )
            if "지상부 길이" in growth_df.columns:
                summary_rows.append(
                    {
                        "항목": "지상부 길이 (평균)",
                        "값": f"{growth_df['지상부 길이'].dropna().mean():.1f}mm",
                    }
                )
            if "지하부 길이" in growth_df.columns:
                summary_rows.append(
                    {
                        "항목": "지하부 길이 (평균)",
                        "값": f"{growth_df['지하부 길이'].dropna().mean():.1f}mm",
                    }
                )
            if "생중량" in growth_df.columns:
                gw = growth_df["생중량"].dropna()
                summary_rows.append({"항목": "생중량 (평균)", "값": f"{gw.mean():.2f}g"})
            summary_df = pd.DataFrame(summary_rows)
            st.dataframe(summary_df, width="stretch")

            st.subheader("📊 지표 카드")

            gw = growth_df["생중량"].dropna() if "생중량" in growth_df.columns else pd.Series()
            if len(gw) >= 2:
                mean_val = gw.mean()
                std_val = gw.std()
                cv_val = std_val / mean_val * 100 if mean_val != 0 else 0
                max_val = gw.max()
                min_val = gw.min()
                metric_row(
                    [
                        ("평균 생중량", f"{mean_val:.2f}g"),
                        ("표준편차", f"{std_val:.2f}g"),
                        ("CV (변동계수)", f"{cv_val:.1f}%"),
                        ("최대 ↔ 최소", f"{max_val:.2f}g ↔ {min_val:.2f}g"),
                    ]
                )
            else:
                st.info("생중량 데이터가 부족해 지표 카드를 표시할 수 없어요.")

            r_items = []
            if "생중량" in growth_df.columns:
                if "지상부 길이" in growth_df.columns:
                    sub = growth_df[["지상부 길이", "생중량"]].dropna()
                    if len(sub) >= 2:
                        r_above = sub["지상부 길이"].corr(sub["생중량"])
                        r_items.append(
                            ("지상부 길이 ↔ 생중량 (r)", f"{r_above:.3f}")
                        )
                if "지하부 길이" in growth_df.columns:
                    sub = growth_df[["지하부 길이", "생중량"]].dropna()
                    if len(sub) >= 2:
                        r_below = sub["지하부 길이"].corr(sub["생중량"])
                        r_items.append(
                            ("지하부 길이 ↔ 생중량 (r)", f"{r_below:.3f}")
                        )
            if r_items:
                metric_row(r_items)
            else:
                st.info("생중량과 비교할 길이 데이터가 없어요.")

        if env_df.empty and growth_df.empty:
            st.info("📂 사이드바에서 환경 CSV와 생육 CSV를 업로드하세요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 2: 환경 데이터
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    try:
        if env_df.empty:
            st.info("📌 환경 CSV를 업로드하면 온도·습도·pH·EC 그래프가 표시됩니다.")
        else:
            # ── 일별 평균 온도·습도 꺾은선 ──────────────────────────
            st.subheader("🌡️ 일별 평균 온도·습도 변화")
            if "time" in env_df.columns and "temperature" in env_df.columns:
                temp_cols = [c for c in ["temperature", "humidity"] if c in env_df.columns]
                if temp_cols:
                    env_daily = env_df.copy()
                    env_daily["date"] = env_daily["time"].dt.date
                    daily = env_daily.groupby("date")[temp_cols].mean().reset_index()
                    daily["date"] = pd.to_datetime(daily["date"])

                    fig_temp = px.line(
                        daily,
                        x="date",
                        y="temperature",
                        markers=True,
                        title="일별 평균 온도",
                        labels={"temperature": "온도 (°C)", "date": "날짜"},
                    )
                    fig_temp.update_traces(line=dict(color="#e74c3c", width=2))
                    fig_temp.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="날짜",
                        yaxis_title="온도 (°C)",
                        height=300,
                    )
                    st.plotly_chart(fig_temp, width="stretch", key="temp_line")

                    fig_hum = px.line(
                        daily,
                        x="date",
                        y="humidity",
                        markers=True,
                        title="일별 평균 습도",
                        labels={"humidity": "습도 (%)", "date": "날짜"},
                    )
                    fig_hum.update_traces(line=dict(color="#3498db", width=2))
                    fig_hum.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="날짜",
                        yaxis_title="습도 (%)",
                        height=300,
                    )
                    st.plotly_chart(fig_hum, width="stretch", key="hum_line")
                else:
                    st.info("온도·습도 열이 없어 그래프를 표시할 수 없어요.")
            else:
                st.info("시간(time) 또는 온도(temperature) 열이 없어 그래프를 표시할 수 없어요.")

            # ── pH · EC 서브플롯 ────────────────────────────────────
            st.subheader("🧪 pH · EC 변화 (서브플롯)")
            ph_ec_cols = [c for c in ["ph", "ec"] if c in env_df.columns]
            if "time" in env_df.columns and ph_ec_cols:
                env_sorted = env_df.dropna(subset=["time"]).sort_values("time")
                n_plots = len(ph_ec_cols)
                fig_sub = make_subplots(
                    rows=n_plots,
                    cols=1,
                    shared_xaxes=True,
                    subplot_titles=["pH 변화", "EC 변화"] if n_plots == 2 else ["pH·EC 변화"],
                    vertical_spacing=0.12,
                )
                colors = {"ph": "#27ae60", "ec": "#8e44ad"}
                for i, col in enumerate(ph_ec_cols, 1):
                    fig_sub.add_trace(
                        go.Scatter(
                            x=env_sorted["time"],
                            y=env_sorted[col],
                            mode="lines+markers",
                            name=col.upper(),
                            line=dict(color=colors.get(col, "#333"), width=1.5),
                            marker=dict(size=3),
                        ),
                        row=i,
                        col=1,
                    )
                fig_sub.update_layout(
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    height=220 * n_plots,
                    showlegend=False,
                )
                fig_sub.update_xaxes(title_text="시간")
                fig_sub.update_yaxes(title_text="pH", row=1, col=1)
                if n_plots >= 2:
                    fig_sub.update_yaxes(title_text="EC (mS/cm)", row=2, col=1)
                st.plotly_chart(fig_sub, width="stretch", key="ph_ec_sub")
            elif not ph_ec_cols:
                st.info("pH·EC 열이 없어 그래프를 표시할 수 없어요.")
            else:
                st.info("시간(time) 열이 없어 pH·EC 그래프를 표시할 수 없어요.")

        if env_file is None and growth_file is None:
            st.info("📂 사이드바에서 환경 CSV를 업로드하세요.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

# ═══════════════════════════════════════════════════════════════════════════
# 탭 3: 생육 결과
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    try:
        if growth_df.empty:
            st.info("📌 생육 결과 CSV를 업로드하면 개체별 생육 분석 그래프가 표시됩니다.")
        else:
            gw_col = "생중량" if "생중량" in growth_df.columns else None
            if gw_col is None:
                st.info("생중량 열이 없어 생육 분석 그래프를 표시할 수 없어요.")
            else:
                gw = growth_df[gw_col].dropna()
                if len(gw) < 2:
                    st.info("생중량 데이터가 2개 미만이라 그래프를 그릴 수 없어요.")
                else:
                    # ── TOP 5 / BOTTOM 5 막대그래프 ────────────────────
                    st.subheader("🏆 생중량 TOP 5 & BOTTOM 5")
                    ranked = growth_df.dropna(subset=[gw_col]).sort_values(
                        gw_col, ascending=False
                    )
                    top5 = ranked.head(5)
                    bot5 = ranked.tail(5)
                    top5_ids = top5["개체번호"].tolist()
                    bot5_ids = bot5["개체번호"].tolist()
                    top5_vals = top5[gw_col].tolist()
                    bot5_vals = bot5[gw_col].tolist()

                    fig_bar = go.Figure()
                    fig_bar.add_trace(
                        go.Bar(
                            x=[str(x) for x in top5_ids],
                            y=top5_vals.tolist(),
                            name="상위 5개체",
                            marker=dict(
                                color="#27ae60",
                                line=dict(width=1, color="white"),
                            ),
                            text=[f"{v:.2f}g" for v in top5_vals],
                            textposition="outside",
                        )
                    )
                    fig_bar.add_trace(
                        go.Bar(
                            x=[str(x) for x in bot5_ids],
                            y=bot5_vals.tolist(),
                            name="하위 5개체",
                            marker=dict(
                                color="#e74c3c",
                                line=dict(width=1, color="white"),
                            ),
                            text=[f"{v:.2f}g" for v in bot5_vals],
                            textposition="outside",
                        )
                    )
                    fig_bar.update_layout(
                        barmode="group",
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        xaxis_title="개체번호",
                        yaxis_title="생중량 (g)",
                        height=380,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    )
                    st.plotly_chart(fig_bar, width="stretch", key="topbot_bar")

                    # ── TOP 5 vs BOTTOM 5 비교 표 ──────────────────────
                    st.markdown("**TOP 5 vs BOTTOM 5 비교**")
                    comp_rows = []
                    for i in range(5):
                        row = {
                            "순위": f"TOP {i+1}",
                            "개체번호": str(top5_ids[i]),
                            "생중량 (g)": f"{top5_vals[i]:.2f}",
                        }
                        if "지상부 길이" in top5.columns:
                            row["지상부 길이 (mm)"] = f"{top5['지상부 길이'].iloc[i]:.1f}"
                        else:
                            row["지상부 길이 (mm)"] = "-"
                        if "지하부 길이" in top5.columns:
                            row["지하부 길이 (mm)"] = f"{top5['지하부 길이'].iloc[i]:.1f}"
                        else:
                            row["지하부 길이 (mm)"] = "-"
                        comp_rows.append(row)
                    for i in range(5):
                        bi = 4 - i
                        row = {
                            "순위": f"BOTTOM {i+1}",
                            "개체번호": str(bot5_ids[bi]),
                            "생중량 (g)": f"{bot5_vals[bi]:.2f}",
                        }
                        if "지상부 길이" in bot5.columns:
                            row["지상부 길이 (mm)"] = f"{bot5['지상부 길이'].iloc[bi]:.1f}"
                        else:
                            row["지상부 길이 (mm)"] = "-"
                        if "지하부 길이" in bot5.columns:
                            row["지하부 길이 (mm)"] = f"{bot5['지하부 길이'].iloc[bi]:.1f}"
                        else:
                            row["지하부 길이 (mm)"] = "-"
                        comp_rows.append(row)
                    comp_df = pd.DataFrame(comp_rows)
                    st.dataframe(comp_df, width="stretch")

                    # ── 생중량 박스플롯 ──────────────────────────────────
                    st.subheader("📦 생중량 박스플롯 (개체별 분포)")
                    fig_box = go.Figure()
                    fig_box.add_trace(
                        go.Box(
                            y=growth_df[gw_col].dropna().tolist(),
                            fillcolor="#8e44ad",
                            line=dict(color="white"),
                            name="전체 개체",
                        )
                    )
                    fig_box.update_layout(
                        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                        yaxis_title="생중량 (g)",
                        xaxis_title="",
                        height=280,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_box, width="stretch", key="boxplot")

                    # ── 지상부·지하부 길이 vs 생중량 산점도 + 추세선 ────
                    scatter_items = []
                    if "지상부 길이" in growth_df.columns:
                        scatter_items.append("지상부 길이")
                    if "지하부 길이" in growth_df.columns:
                        scatter_items.append("지하부 길이")

                    if scatter_items:
                        st.subheader("🔍 길이 vs 생중량 산점도 + 추세선")
                        cols_frame = st.columns(len(scatter_items))
                        for col, item_col in zip(cols_frame, scatter_items):
                            sub = growth_df[[item_col, gw_col]].dropna()
                            if len(sub) < 2:
                                col.info(f"{item_col} 데이터가 부족해 산점도를 그릴 수 없어요.")
                                continue
                            r_val = sub[item_col].corr(sub[gw_col])

                            fig_sc = go.Figure()
                            fig_sc.add_trace(
                                go.Scatter(
                                    x=sub[item_col].tolist(),
                                    y=sub[gw_col].tolist(),
                                    mode="markers",
                                    marker=dict(
                                        size=9,
                                        color="#2980b9",
                                        line=dict(width=1, color="white"),
                                        opacity=0.85,
                                    ),
                                    name="개체",
                                    hovertemplate=f"{item_col}: %{{x:.1f}}<br>생중량: %{{y:.2f}}g<extra></extra>",
                                )
                            )
                            add_trendline(fig_sc, sub[item_col], sub[gw_col])
                            fig_sc.update_layout(
                                font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                                xaxis_title=f"{item_col} (mm)",
                                yaxis_title="생중량 (g)",
                                height=320,
                                showlegend=True,
                                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                            )
                            col.plotly_chart(fig_sc, width="stretch")

                            direction = (
                                "강한 양의 상관"
                                if r_val > 0.7
                                else "중간 양의 상관"
                                if r_val > 0.3
                                else "약한 관계"
                                if abs(r_val) < 0.3
                                else "음의 상관"
                            )
                            col.caption(
                                f"📌 r = {r_val:.3f} — {direction}  "
                                f"(1에 가까울수록 함께 커짐, -1에 가까울수록 반대로 움직임)"
                            )
                    else:
                        st.info("지상부 길이·지하부 길이 열이 없어 산점도 그래프를 표시할 수 없어요.")

        if env_file is None and growth_file is None:
            st.info("📂 사이드바에서 생육 결과 CSV를 업로드하세요.")

        # ── CSV 다운로드 (한글 깨짐 방지: utf-8-sig) ──────────────
        if not growth_df.empty:
            csv_bytes = growth_df.to_csv(index=False).encode("utf-8-sig")
            fname = f"{school}_생육_분석용.csv" if school else "생육_분석용.csv"
            st.download_button(
                label="📥 표준화된 생육 CSV 다운로드",
                data=csv_bytes,
                file_name=fname,
                mime="text/csv",
            )
        if not env_df.empty:
            env_csv = env_df.to_csv(index=False).encode("utf-8-sig")
            fname = f"{school}_환경_분석용.csv" if school else "환경_분석용.csv"
            st.download_button(
                label="📥 표준화된 환경 CSV 다운로드",
                data=env_csv,
                file_name=fname,
                mime="text/csv",
            )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
