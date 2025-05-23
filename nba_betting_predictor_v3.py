import pandas as pd
import numpy as np
import streamlit as st
import os
from datetime import datetime
import itertools

# CSV 파일 경로
SCHEDULE_PATH = "nba_schedule_2526.csv"
NEWS_PATH = "fake_nba_news_utf8.csv"

TEAM_OPTIONS = ["주요선수 이탈", "주요선수 복귀", "백투백 경기", "클럽하우스 불화", "승리요정", "패배요정"]

def apply_team_adjustment(prob, conditions):
    adj = 0
    for cond in conditions:
        if cond == "주요선수 이탈":
            adj -= 0.05
        elif cond == "주요선수 복귀":
            adj += 0.05
        elif cond == "백투백 경기":
            adj -= 0.02
        elif cond == "클럽하우스 불화":
            adj -= 0.03
        elif cond == "승리요정":
            adj += 0.05
        elif cond == "패배요정":
            adj -= 0.05
    return np.clip(prob + adj, 0.01, 0.99)

def sanitize_conditions(conds, team_key):
    if "승리요정" in conds and "패배요정" in conds:
        st.warning(f"⚠️ {team_key}에는 '승리요정'과 '패배요정'을 동시에 선택할 수 없습니다. 첫 번째 것만 유지됩니다.")
        conds = [c for c in conds if c not in ["승리요정", "패배요정"]] + [conds[-1]]
    return conds

@st.cache_data
def load_news():
    if os.path.exists(NEWS_PATH):
        df_news = pd.read_csv(NEWS_PATH)
        df_news["date"] = pd.to_datetime(df_news["date"])
        return df_news
    return pd.DataFrame(columns=["team", "title", "summary", "date"])

def extract_news_options(news_df, team, game_date):
    options = []
    relevant = news_df[(news_df["team"] == team) & (news_df["date"] <= game_date)]
    for text in (relevant["title"].fillna("") + " " + relevant["summary"].fillna("")):
        if "복귀" in text or "회복" in text:
            options.append("주요선수 복귀")
        if "부상" in text or "이탈" in text or "결장" in text:
            options.append("주요선수 이탈")
        if "불화" in text or "내분" in text:
            options.append("클럽하우스 불화")
    return list(set(options)), relevant[["date", "title", "summary"]]

def evaluate_betting_combinations(prob_components, bet_inputs, odds_inputs):
    n = len(prob_components)
    best_ev = -float("inf")
    best_combo = None
    best_win = 0

    for combo in itertools.product([1, 0], repeat=n):
        total_ev = 0
        total_win = 0
        for i in range(n):
            _, _, prob = prob_components[i]
            bet = bet_inputs[i]
            odds = odds_inputs[i]
            p = prob if combo[i] == 1 else 1 - prob
            expected_win = bet * odds
            ev = expected_win * p - bet * (1 - p)
            total_ev += ev
            total_win += expected_win * p

        if total_ev > best_ev:
            best_ev = total_ev
            best_combo = combo
            best_win = total_win

    return best_combo, best_ev, best_win
def run_app():
    st.set_page_config("NBA 베팅 예측 시스템", layout="wide")
    st.title("🏀 2025–26 NBA 시즌 예측 및 베팅 시뮬레이션")

    news_df = load_news()

    if not os.path.exists(SCHEDULE_PATH):
        st.error("경기 일정 파일이 존재하지 않습니다.")
        return

    df = pd.read_csv(SCHEDULE_PATH)
    df["date"] = pd.to_datetime(df["date"])

    menu = st.radio("탭 선택", ["경기 예측", "월별 일정", "팀별 일정"])

    if menu == "경기 예측":
        selected_date = st.date_input("예측할 날짜", value=df["date"].min().date())
        games = df[df["date"].dt.date == selected_date].reset_index(drop=True)

        if games.empty:
            st.warning("선택한 날짜에 경기가 없습니다.")
            return

        bet_inputs, odds_inputs = [], []
        user_preds, home_conditions, away_conditions = [], [], []
        prob_components, selected_games = [], []

        st.markdown("---")
        st.subheader("📋 경기 예측 및 베팅 입력")

        for idx, row in games.iterrows():
            st.markdown(f"### 🏀 {row['home']} vs {row['away']}")
            st.markdown(f"예측 확률: {row['home']} {row['home_win_prob']*100:.1f}% / {row['away']} {row['away_win_prob']*100:.1f}%")

            home_auto_opts, home_news = extract_news_options(news_df, row['home'], row['date'])
            away_auto_opts, away_news = extract_news_options(news_df, row['away'], row['date'])

            if not home_news.empty:
                with st.expander(f"📰 홈팀 {row['home']} 관련 뉴스"):
                    st.dataframe(home_news)

            if not away_news.empty:
                with st.expander(f"📰 원정팀 {row['away']} 관련 뉴스"):
                    st.dataframe(away_news)

            hc_raw = st.multiselect(f"홈팀 조건 - {row['home']}", TEAM_OPTIONS, default=home_auto_opts, key=f"hc_{idx}")
            ac_raw = st.multiselect(f"원정팀 조건 - {row['away']}", TEAM_OPTIONS, default=away_auto_opts, key=f"ac_{idx}")

            hc = sanitize_conditions(hc_raw, row['home'])
            ac = sanitize_conditions(ac_raw, row['away'])

            pred = st.radio(f"승리 예측 - {row['home']} vs {row['away']}", ["선택안함", "Home", "Away"], key=f"pred_{idx}", index=0)
            p_val = None if pred == "선택안함" else 1 if pred == "Home" else 0

            if p_val is not None:
                selected_games.append(row)
                user_preds.append(p_val)
                home_conditions.append(hc)
                away_conditions.append(ac)

                h = apply_team_adjustment(row["home_win_prob"], hc)
                a = apply_team_adjustment(row["away_win_prob"], ac)
                total = h + a
                h /= total
                a /= total
                prob = h if p_val == 1 else a
                prob_components.append((row['home'], row['away'], prob))

                match_key = f"{row['home']}_vs_{row['away']}"
                bet = st.number_input(f"💰 베팅 금액 - {match_key}", min_value=1000, value=10000, step=1000, key=f"bet_{idx}")
                odds = st.number_input(f"📈 배당률 - {match_key}", min_value=1.0, value=1.95, step=0.05, key=f"odds_{idx}")
                bet_inputs.append(bet)
                odds_inputs.append(odds)

        if st.button("📊 예측 결과 및 베팅 수익 계산") and selected_games:
            total_expected_win = 0
            total_expected_value = 0
            total_bet = 0

            st.markdown("---")
            st.subheader("📈 예측 결과 분석")

            for i, row in enumerate(selected_games):
                prob = prob_components[i][2]
                bet = bet_inputs[i]
                odds = odds_inputs[i]
                expected_win = bet * odds
                expected_val = expected_win * prob - bet * (1 - prob)

                total_expected_win += expected_win * prob
                total_expected_value += expected_val
                total_bet += bet

                st.markdown(f"### ✅ {row['home']} vs {row['away']}")
                st.markdown(f"🧮 예측 확률: {prob*100:.1f}%")
                st.markdown(f"💰 베팅 금액: ₩{bet:,.0f}")
                st.markdown(f"📈 기대 수익 (성공 시): ₩{expected_win:,.0f}")
                st.markdown(f"📉 손실 (실패 시): -₩{bet:,.0f}")
                st.markdown(f"📊 기대값(EV): ₩{expected_val:,.0f}")
                st.markdown("---")

            st.subheader("📊 전체 베팅 요약")
            st.markdown(f"💸 총 베팅 금액: ₩{total_bet:,.0f}")
            st.markdown(f"🎯 총 기대 수익: ₩{total_expected_win:,.0f}")
            st.markdown(f"📊 총 기대값(EV): ₩{total_expected_value:,.0f}")

            best_combo, best_ev, best_win = evaluate_betting_combinations(prob_components, bet_inputs, odds_inputs)

            st.subheader("🧠 최적 베팅 전략 (EV 기준)")
            for i, val in enumerate(best_combo):
                match = f"{prob_components[i][0]} vs {prob_components[i][1]}"
                pick = "Home" if val == 1 else "Away"
                st.markdown(f"- {match}: **{pick}** 선택")

            st.markdown(f"💡 최적 기대값(EV): ₩{best_ev:,.0f}")
            st.markdown(f"🎯 최적 기대 수익: ₩{best_win:,.0f}")

    elif menu == "월별 일정":
        df["month"] = df["date"].dt.strftime('%Y-%m')
        month = st.selectbox("월 선택", sorted(df["month"].unique()))
        for d in sorted(df[df["month"] == month]["date"].unique()):
            st.markdown(f"📆 {d}")
            for _, row in df[df["date"] == d].iterrows():
                st.markdown(f"- {row['home']} vs {row['away']}")

    elif menu == "팀별 일정":
        team = st.selectbox("팀 선택", sorted(df["home"].unique()))
        team_games = df[(df["home"] == team) | (df["away"] == team)]
        for _, row in team_games.iterrows():
            st.markdown(f"📅 {row['date']} — {row['home']} vs {row['away']}")

if __name__ == "__main__":
    run_app()