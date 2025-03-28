from gensim.models import Word2Vec

# 파일 열기
with open(r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\THREE_NLTK_stop_words\faust_dialogues_final.txt", "r", encoding="utf-8") as f:
    faust_lines = f.readlines()
with open(r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\THREE_NLTK_stop_words\mephisto_dialogues_final.txt", "r", encoding="utf-8") as f:
    mephi_lines = f.readlines()

# ✅ 토큰화된 문장 리스트 (불용어 제거 + 전처리된 문장)
tokenized_lines_faust = [line.strip().split() for line in faust_lines if line.strip()]
# ✅ 토큰화된 문장 리스트 (불용어 제거 + 전처리된 문장)
tokenized_lines_mephi = [line.strip().split() for line in mephi_lines if line.strip()]


# ✅ Word2Vec 학습
model1 = Word2Vec(
    sentences=tokenized_lines_faust,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# ✅ Word2Vec 학습
model2 = Word2Vec(
    sentences=tokenized_lines_mephi,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# ✅ 연관 단어 출력
print("🔍 '파우스트의 love'와 유사한 단어:")
for word, score in model1.wv.most_similar('love', topn=10):
    print(f"{word:10} → {score:.4f}")

# ✅ 연관 단어 출력
print("🔍 '메피스토펠레스의 devil'과 유사한 단어:")
for word, score in model2.wv.most_similar('devil', topn=10):
    print(f"{word:10} → {score:.4f}")
