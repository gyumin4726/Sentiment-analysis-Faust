import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd

# 1️⃣ NLTK의 VADER 감성 분석기 다운로드 및 초기화
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text_lines):
    """
    각 대사에 대해 감성 점수를 계산하고 저장하는 함수
    """
    sentiment_data = []
    
    for idx, line in enumerate(text_lines):
        scores = sia.polarity_scores(line)  # VADER 감성 점수 계산
        sentiment_data.append({
            "index": idx,  # 소설 진행 순서
            "text": line.strip(),
            "positive": scores["pos"],
            "neutral": scores["neu"],
            "negative": scores["neg"],
            "compound": scores["compound"],  # 전체 감정 점수 (-1 ~ +1)
        })
    
    return sentiment_data

# 📌 수정된 파일 경로 (백슬래시 문제 해결: r"" 또는 os.path 사용)
faust_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\stop_words_default\faust_cleaned_no_stopwords.txt"
mephi_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\stop_words_default\mephi_cleaned_no_stopwords.txt"

with open(faust_file_path, "r", encoding="utf-8") as f:
    faust_text = f.readlines()

with open(mephi_file_path, "r", encoding="utf-8") as f:
    mephi_text = f.readlines()

# 3️⃣ 감성 분석 실행
faust_sentiment = analyze_sentiment(faust_text)
mephi_sentiment = analyze_sentiment(mephi_text)

# 4️⃣ 데이터프레임으로 변환 (시각화 준비)
faust_df = pd.DataFrame(faust_sentiment)
mephi_df = pd.DataFrame(mephi_sentiment)

# 5️⃣ 감정 변화 시각화 (시간 흐름에 따른 감정 변화)
plt.figure(figsize=(12, 6))
plt.plot(faust_df["index"], faust_df["compound"], label="Faust", linestyle="-", alpha=0.7)
plt.plot(mephi_df["index"], mephi_df["compound"], label="Mephistopheles", linestyle="--", alpha=0.7)

plt.axhline(y=0, color="black", linestyle="--", linewidth=0.5)  # 중립선
plt.xlabel("time steps")
plt.ylabel("sentimental score")
plt.title("Faust vs Mephistopheles")
plt.legend()
plt.show()
