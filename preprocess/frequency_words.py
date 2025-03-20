from collections import Counter
import string
import os

def get_word_frequencies(text_lines):
    """
    텍스트에서 단어 빈도수를 계산하는 함수
    """
    all_words = []
    
    for line in text_lines:
        line = line.lower()  # 소문자로 변환
        line = line.translate(str.maketrans("", "", string.punctuation))  # 문장 부호 제거
        words = line.split()  # 단어 단위로 분할
        all_words.extend(words)  # 리스트에 추가
    
    word_counts = Counter(all_words)  # 단어 빈도수 계산
    return word_counts

# 📌 수정된 파일 경로 (백슬래시 문제 해결: r"" 또는 os.path 사용)
faust_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\stop_words_default\faust_cleaned_no_stopwords.txt"
mephi_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\stop_words_default\mephi_cleaned_no_stopwords.txt"

# 파일 읽기
with open(faust_file_path, "r", encoding="utf-8", errors="replace") as f:
    faust_text = f.readlines()

with open(mephi_file_path, "r", encoding="utf-8", errors="replace") as f:
    mephi_text = f.readlines()

# 단어 빈도수 계산
faust_word_counts = get_word_frequencies(faust_text)
mephi_word_counts = get_word_frequencies(mephi_text)

# 상위 20개 단어 출력
print("📌 파우스트 대사에서 가장 많이 나온 단어 20개:")
for word, count in faust_word_counts.most_common(20):
    print(f"{word}: {count}")

print("\n📌 메피스토펠레스 대사에서 가장 많이 나온 단어 20개:")
for word, count in mephi_word_counts.most_common(20):
    print(f"{word}: {count}") 
