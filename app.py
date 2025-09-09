# app.py
import streamlit as st
import pandas as pd
import requests
import yfinance as yf
from datetime import datetime
import pytz

# ---------------------------
# 기본 설정
# ---------------------------
KST = pytz.timezone("Asia/Seoul")
st.set_page_config(page_title="투자 비교 대시보드 (O·예적금·BTC/ETH)", layout="wide")
st.title("📊 투자 비교 대시보드: 리얼티인컴 · 예·적금 · BTC/ETH (Finlife API 버전)")
st.caption("KRW 기준 비교. 환율/수수료 적용·미적용을 분리해 계산합니다. 예·적금 데이터는 금융감독원 Finlife 공식 API를 사용합니다.")

# ---------------------------
# 환율 (Frankfurter, 실패 시 기본값)
# ---------------------------
@st.cache_data(ttl=300)
def get_usdkrw():
    try:
        r = requests.get("https://api.frankfurter.app/latest?from=USD&to=KRW", timeout=15)
        r.raise_for_status()
        data = r.json()
        rate = float(data["rates"]["KRW"])
        ts = data.get("date", datetime.now().strftime("%Y-%m-%d"))
        return rate, ts
    except Exception:
        st.warning("환율 API 오류: 기본값 1350.0 사용 (사이드바에서 수정 가능)")
        return 1350.0, datetime.now().strftime("%Y-%m-%d")

# ---------------------------
# 리얼티인컴 가격/배당
# ---------------------------
@st.cache_data(ttl=900)
def get_realty_income_price_div(ticker="O"):
    t = yf.Ticker(ticker)
    hist = t.history(period="1mo", interval="1d")
    price_usd, price_date = None, None
    if not hist.empty:
        price_usd = float(hist["Close"].iloc[-1])
        try:
            price_date = hist.index[-1].tz_localize("UTC").astimezone(KST).strftime("%Y-%m-%d")
        except Exception:
            price_date = str(hist.index[-1].date())
    divs = t.dividends
    last_monthly_div = float(divs.iloc[-1]) if divs is not None and len(divs) > 0 else None
    return price_usd, price_date, last_monthly_div

# ---------------------------
# 업비트 일봉
# ---------------------------
@st.cache_data(ttl=300)
def get_upbit_daily(market="KRW-BTC", count=8):
    try:
        url = "https://api.upbit.com/v1/candles/days"
        res = requests.get(url, params={"market": market, "count": count}, timeout=15)
        res.raise_for_status()
        data = res.json()
        closes = [d["trade_price"] for d in reversed(data)]
        dates = [d["candle_date_time_kst"][:10] for d in reversed(data)]
        return pd.DataFrame({"date": dates, "close": closes})
    except Exception:
        return pd.DataFrame(columns=["date","close"])

def pct_change_over(df: pd.DataFrame, days: int):
    if df is None or df.empty or len(df) <= days:
        return None
    now = df["close"].iloc[-1]
    past = df["close"].iloc[-1 - days]
    return (now - past) / past * 100.0

# ---------------------------
# Finlife (금융상품 한눈에) 공식 API
# ---------------------------
FINLIFE_BASE = "https://finlife.fss.or.kr/finlifeapi"
TOP_BANKS = "020000"  # 은행(제1금융권)

@st.cache_data(ttl=600)
def finlife_fetch(api_key: str, endpoint: str, page_no: int = 1):
    """
    endpoint: 'depositProductsSearch.json' or 'savingProductsSearch.json'
    """
    if not api_key:
        return {}
    url = f"{FINLIFE_BASE}/{endpoint}"
    params = {"auth": api_key, "topFinGrpNo": TOP_BANKS, "pageNo": page_no}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _to_numeric(x):
    try:
        return float(x)
    except Exception:
        return None

def _filter_options(df_opt: pd.DataFrame, *, save_trm: int = 12, intr_rate_type: str = "S", rsrv_type: str | None = None):
    """
    save_trm: 만기(개월), e.g., 12
    intr_rate_type: 'S'(단리) or 'M'(복리)
    rsrv_type (적금만): 'F'(자유적립식) or 'S'(정액적립식)
    """
    if df_opt is None or df_opt.empty:
        return df_opt
    out = df_opt.copy()
    # 숫자화
    for c in ["save_trm", "intr_rate", "intr_rate2"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    # 필터
    out = out[(out.get("save_trm") == save_trm) & (out.get("intr_rate_type") == intr_rate_type)]
    if rsrv_type is not None and "rsrv_type" in out.columns:
        out = out[out["rsrv_type"] == rsrv_type]
    return out

def _merge_base_option(base: pd.DataFrame, opt: pd.DataFrame, *,
                       sort_by: str, top_n: int = 5, label_map: dict | None = None):
    """
    base: baseList (상품 기본정보)
    opt:  optionList (금리 옵션: intr_rate(기본), intr_rate2(우대))
    sort_by: 'intr_rate' or 'intr_rate2'
    """
    if base is None or base.empty or opt is None or opt.empty:
        return pd.DataFrame()

    # 옵션 테이블에서 같은 상품(fin_prdt_cd) 내 중복 → 최대값(해당 금리)으로 집계
    best = (
        opt.groupby("fin_prdt_cd")[[col for col in ["intr_rate", "intr_rate2"] if col in opt.columns]]
           .max()
           .reset_index()
    )
    merged = base.merge(best, on="fin_prdt_cd", how="inner")

    # 정렬
    if sort_by in merged.columns:
        merged = merged.sort_values(sort_by, ascending=False)
    else:
        # 기본금리/우대금리 둘 다 없다면 바로 빈 DF
        return pd.DataFrame()

    # 표 컬럼 구성
    out = pd.DataFrame()
    out["은행"] = merged.get("kor_co_nm", "")
    out["상품명"] = merged.get("fin_prdt_nm", "")
    if "intr_rate" in merged.columns:
        out["기본금리(%)"] = merged["intr_rate"].round(3)
    if "intr_rate2" in merged.columns:
        out["최고우대(%)"] = merged["intr_rate2"].round(3)

    # 레이블 치환(선택)
    if label_map:
        out["은행"] = out["은행"].replace(label_map)

    return out.head(top_n).reset_index(drop=True)

@st.cache_data(ttl=600)
def finlife_top5_deposit(api_key: str):
    """
    정기예금:
    - topFinGrpNo=020000(은행)
    - 옵션 필터: 단리(S), 12개월
    - 정렬: '기본금리(%)' (intr_rate) 내림차순
    """
    try:
        js = finlife_fetch(api_key, "depositProductsSearch.json", 1)
        base = pd.DataFrame(js.get("result", {}).get("baseList", []))
        opt  = pd.DataFrame(js.get("result", {}).get("optionList", []))
        if base.empty or opt.empty:
            return pd.DataFrame()
        opt_f = _filter_options(opt, save_trm=12, intr_rate_type="S")
        if opt_f.empty:
            return pd.DataFrame()
        out = _merge_base_option(base, opt_f, sort_by="intr_rate", top_n=5)
        return out
    except Exception as e:
        st.error(f"Finlife 예금 API 오류: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def finlife_top5_saving(api_key: str):
    """
    적금:
    - topFinGrpNo=020000(은행)
    - 옵션 필터: 자유적립식(F), 단리(S), 12개월
    - 정렬: '최고우대(%)' (intr_rate2) 내림차순 (동률 시 기본금리 보조)
    """
    try:
        js = finlife_fetch(api_key, "savingProductsSearch.json", 1)
        base = pd.DataFrame(js.get("result", {}).get("baseList", []))
        opt  = pd.DataFrame(js.get("result", {}).get("optionList", []))
        if base.empty or opt.empty:
            return pd.DataFrame()
        opt_f = _filter_options(opt, save_trm=12, intr_rate_type="S", rsrv_type="F")
        if opt_f.empty:
            return pd.DataFrame()
        # 우선 정렬 키 생성
        opt_f = opt_f.copy()
        opt_f["intr_rate2"] = pd.to_numeric(opt_f["intr_rate2"], errors="coerce")
        opt_f["intr_rate"]  = pd.to_numeric(opt_f["intr_rate"], errors="coerce")
        opt_f = opt_f.sort_values(["intr_rate2","intr_rate"], ascending=False)
        out = _merge_base_option(base, opt_f, sort_by="intr_rate2", top_n=5)
        return out
    except Exception as e:
        st.error(f"Finlife 적금 API 오류: {e}")
        return pd.DataFrame()

def add_simple_interest_krw(df: pd.DataFrame, principal_krw: float, rate_col: str):
    if df is None or df.empty or rate_col not in df.columns:
        return df
    out = df.copy()
    out["연이자(원, 단리)"] = (principal_krw * (pd.to_numeric(out[rate_col], errors="coerce")/100.0)).round(0).astype("Int64")
    return out

# ---------------------------
# 사이드바 (옵션)
# ---------------------------
with st.sidebar:
    st.header("⚙️ 옵션")
    base_amt_krw = st.number_input("기준 금액 (KRW)", min_value=10_000, step=10_000, value=5_000_000)
    apply_fx = st.checkbox("환율 적용", True)
    apply_fees = st.checkbox("수수료 적용", True)

    st.markdown("---")
    st.subheader("환율 (USD→KRW)")
    api_rate, api_date = get_usdkrw()
    usdkrw = st.number_input("USD/KRW 환율 (수동 조정 가능)", min_value=500.0, max_value=3000.0, value=float(api_rate), step=0.5, help=f"API 기준일: {api_date}")

    st.markdown("---")
    st.subheader("거래 수수료 (편집 가능)")
    upbit_fee_spot = st.number_input("업비트 KRW 마켓 수수료 (일반, %)", min_value=0.0, max_value=1.0, value=0.05, step=0.001)
    upbit_fee_reserved = st.number_input("업비트 KRW 마켓 수수료 (예약, %)", min_value=0.0, max_value=1.0, value=0.139, step=0.001)
    mirae_us_fee = st.number_input("미래에셋 미국주식 온라인 수수료 (%)", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

    st.markdown("---")
    st.subheader("Finlife 오픈 API 키")
    finlife_key = st.secrets.get("FINLIFE_API_KEY") or st.sidebar.text_input("API Key", type="password")

# ---------------------------
# 1) 리얼티인컴 (O): 종가·배당·수수료 + KRW 환산
# ---------------------------
st.header("1) 리얼티인컴 (O): 종가·배당·수수료")

price_usd, price_date, last_monthly_div = get_realty_income_price_div("O")
c1, c2, c3 = st.columns([1.1,1.6,1.1])

with c1:
    if price_usd:
        st.metric("최근 종가 (USD)", f"${price_usd:,.2f}", help=f"마지막 거래일: {price_date}")
        show_price = price_usd * (usdkrw if apply_fx else 1.0)
        st.metric(f"최근 종가 ({'KRW' if apply_fx else 'USD'})", (f"{show_price:,.0f} 원" if apply_fx else f"${price_usd:,.2f}"))
    else:
        st.error("가격 데이터를 불러오지 못했습니다. (yfinance)")

with c2:
    monthly_div_usd = last_monthly_div if last_monthly_div else 0.0
    annual_div_usd = monthly_div_usd * 12

    if price_usd:
        buy_fee = (mirae_us_fee/100.0) if apply_fees else 0.0
        usd_budget = (base_amt_krw / (usdkrw if apply_fx else 1.0))
        shares_no_fee = usd_budget / price_usd
        shares_fee    = usd_budget * (1 - buy_fee) / price_usd

        # USD 기준 배당
        m_div_usd_no_fee = shares_no_fee * monthly_div_usd
        y_div_usd_no_fee = m_div_usd_no_fee * 12
        m_div_usd_fee    = shares_fee * monthly_div_usd
        y_div_usd_fee    = m_div_usd_fee * 12

        # KRW 환산
        fx = (usdkrw if apply_fx else 1.0)
        m_div_krw_no_fee = m_div_usd_no_fee * fx
        y_div_krw_no_fee = y_div_usd_no_fee * fx
        m_div_krw_fee    = m_div_usd_fee * fx
        y_div_krw_fee    = y_div_usd_fee * fx

        st.write("**매수 가능 주식 수(기준 금액 기준)**")
        st.write(f"- 수수료 미적용: **{shares_no_fee:.4f}주**")
        st.write(f"- 수수료 적용(매수 {mirae_us_fee:.2f}%): **{shares_fee:.4f}주**")

        st.write("**배당 (USD → KRW 환산)**")
        st.write(f"- 월배당(미적용): **${m_div_usd_no_fee:,.2f} → {m_div_krw_no_fee:,.0f}원**")
        st.write(f"- 연배당(미적용): **${y_div_usd_no_fee:,.2f} → {y_div_krw_no_fee:,.0f}원**")
        st.write(f"- 월배당(수수료 적용): **${m_div_usd_fee:,.2f} → {m_div_krw_fee:,.0f}원**")
        st.write(f"- 연배당(수수료 적용): **${y_div_usd_fee:,.2f} → {y_div_krw_fee:,.0f}원**")

        if monthly_div_usd > 0:
            curr_yield = (annual_div_usd / price_usd) * 100
            st.metric("배당수익률(연, %)", f"{curr_yield:.2f}%")
    else:
        st.info("가격 정보가 없어서 배당 환산을 생략했습니다.")

with c3:
    if price_usd:
        fee_round = (mirae_us_fee/100.0) * (2 if apply_fees else 0)
        net_cost_krw = base_amt_krw * (1 + fee_round)
        st.write("**원금·수수료(참고)**")
        st.write(f"- 기준 금액: **{base_amt_krw:,.0f}원**")
        st.write(f"- 수수료 {'적용' if apply_fees else '미적용'} 왕복 가정 원금: **{net_cost_krw:,.0f}원**")
    else:
        st.write("—")

st.caption("배당/환율/수수료는 시점·이벤트에 따라 변동. 실제 주문 화면의 값이 우선합니다. (세금은 별도)")

# ---------------------------
# 2) (Finlife API) 예·적금 상위 5 + 연이자(KRW, 단리)
# ---------------------------
st.header("2) (Finlife) 예·적금 상위 5 + 연이자(KRW, 단리)")

if not finlife_key:
    st.warning("Finlife API Key가 없습니다. 사이드바에 키를 입력하면 실시간 예·적금 금리가 표시됩니다.")

deposit_df = finlife_top5_deposit(finlife_key) if finlife_key else pd.DataFrame()
saving_df  = finlife_top5_saving(finlife_key)  if finlife_key else pd.DataFrame()

# 연이자(원) 계산 칼럼 추가
deposit_show = add_simple_interest_krw(deposit_df, base_amt_krw, "기본금리(%)") if not deposit_df.empty else deposit_df
# 적금은 최고우대(%) 기준 수익 비교가 일반적이므로 그 기준으로 연이자 산출
saving_show  = add_simple_interest_krw(saving_df,  base_amt_krw, "최고우대(%)")  if not saving_df.empty else saving_df

colA, colB = st.columns(2)
with colA:
    st.subheader("예금 Top 5 (은행·단리·12개월 / 정렬: 기본금리)")
    if deposit_show.empty:
        st.info("예금 데이터가 비어 있습니다. (API 키, 네트워크, 혹은 조건에 맞는 옵션 미존재 가능성)")
    st.dataframe(deposit_show, use_container_width=True, hide_index=True)

with colB:
    st.subheader("적금 Top 5 (자유적립식·단리·12개월 / 정렬: 최고우대)")
    if saving_show.empty:
        st.info("적금 데이터가 비어 있습니다. (API 키, 네트워크, 혹은 조건에 맞는 옵션 미존재 가능성)")
    st.dataframe(saving_show, use_container_width=True, hide_index=True)

st.caption("출처: 금융감독원 ‘금융상품 한눈에’ 오픈API. 금리는 세전이며, 실제 가입조건/우대금리/기간에 따라 달라질 수 있습니다.")

# ---------------------------
# 3) 업비트 BTC·ETH 변동률 (1·3·5·7일)
# ---------------------------
st.header("3) 업비트 BTC·ETH 변동률 (1·3·5·7일)")
tabs = st.tabs(["BTC", "ETH"])
for market, tab in zip(["KRW-BTC","KRW-ETH"], tabs):
    with tab:
        df = get_upbit_daily(market, count=8)
        if df.empty:
            st.error("업비트 일별 데이터를 불러오지 못했습니다.")
            continue
        c1, c2 = st.columns([1,1])
        with c1:
            st.line_chart(df.set_index("date")["close"])
        with c2:
            c_1 = pct_change_over(df, 1)
            c_3 = pct_change_over(df, 3)
            c_5 = pct_change_over(df, 5)
            c_7 = pct_change_over(df, 7)
            fmt = lambda v: ("+" if v is not None and v>=0 else "") + (f"{v:.2f}%" if v is not None else "N/A")
            st.metric("1일 변화", fmt(c_1))
            st.metric("3일 변화", fmt(c_3))
            st.metric("5일 변화", fmt(c_5))
            st.metric("7일 변화", fmt(c_7))

        st.subheader("수수료 적용 비교 (예시 체결)")
        latest = df["close"].iloc[-1]
        fee = (upbit_fee_spot/100.0) if apply_fees else 0.0
        qty_no_fee = base_amt_krw / latest
        qty_fee    = (base_amt_krw * (1 - fee)) / latest
        gross_exit = qty_no_fee * latest
        net_exit_fee = qty_fee * latest * (1 - fee)
        st.write(f"- 현재가: **{latest:,.0f} KRW**")
        st.write(f"- 수수료 미적용 매수 수량: **{qty_no_fee:.8f}**")
        st.write(f"- 수수료 적용 매수 수량: **{qty_fee:.8f}** (매수 {upbit_fee_spot:.3f}%)")
        st.write(f"- 동가 매도 가정 왕복 수수료 효과: **{gross_exit - net_exit_fee:,.0f} KRW**")

st.caption("업비트 수수료 예시: KRW마켓 일반 0.05%, 예약 0.139% (부가세 포함, 이벤트 변동 가능). 출금 수수료 별도.")

# ---------------------------
# 4) 통합 비교 요약 (리얼티인컴 KRW 배당 포함)
# ---------------------------
st.header("4) 통합 비교 요약")
s1, s2, s3 = st.columns(3)

with s1:
    st.subheader("리얼티인컴")
    if price_usd:
        show_price = price_usd * (usdkrw if apply_fx else 1.0)
        st.write(f"- 종가: **{'$'+format(price_usd, ',.2f') if not apply_fx else format(show_price, ',.0f')+'원'}**")
        monthly_div_usd = last_monthly_div if last_monthly_div else 0.0
        if monthly_div_usd > 0:
            buy_fee = (mirae_us_fee/100.0) if apply_fees else 0.0
            usd_budget = (base_amt_krw / (usdkrw if apply_fx else 1.0))
            shares_fee = usd_budget * (1 - buy_fee) / price_usd
            m_div_usd_fee = shares_fee * monthly_div_usd
            y_div_usd_fee = m_div_usd_fee * 12
            fx = (usdkrw if apply_fx else 1.0)
            m_div_krw_fee = m_div_usd_fee * fx
            y_div_krw_fee = y_div_usd_fee * fx
            st.write(f"- 월배당(수수료 적용): **${m_div_usd_fee:,.2f} → {m_div_krw_fee:,.0f}원**")
            st.write(f"- 연배당(수수료 적용): **${y_div_usd_fee:,.2f} → {y_div_krw_fee:,.0f}원**")
            st.write(f"- 배당수익률(연/주당): **{(monthly_div_usd*12/price_usd*100):.2f}%**")
        else:
            st.write("- 최근 배당 데이터 없음(yfinance)")
        st.write(f"- 미래에셋 매수 수수료(한 방향): **{mirae_us_fee:.2f}%**")
    else:
        st.write("—")

with s2:
    st.subheader("예금 Top 5 (연이자 KRW 포함)")
    st.dataframe(deposit_show if not deposit_show.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

with s3:
    st.subheader("적금 Top 5 (연이자 KRW 포함)")
    st.dataframe(saving_show if not saving_show.empty else pd.DataFrame(), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown(
"""
**메모**  
- 예금: 제1금융권(topFinGrpNo=020000) / **단리(S)** / **12개월** 옵션만 필터하고, **기본금리(%)** 기준으로 정렬합니다.  
- 적금: 제1금융권 / **자유적립식(F)** / **단리(S)** / **12개월** 옵션만 필터하고, **최고우대(%)** 기준으로 정렬합니다.  
- 연이자(원)는 기준 금액 × (금리/100) **단리**로 계산했습니다(세전). 실제 우대조건/기간에 따라 달라질 수 있습니다.  
- 리얼티인컴 배당의 KRW 환산은 보유주식 수(수수료 반영) × 월배당(USD/주) × 환율로 계산합니다.  
"""
)

