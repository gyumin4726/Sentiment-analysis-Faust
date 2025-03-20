import re
import nltk
from nltk.corpus import stopwords

# 불용어 다운로드 (최초 실행 시 필요)
nltk.download("stopwords")

# 영어 불용어 리스트 가져오기
stop_words = set(stopwords.words("english"))

def remove_stopwords(text_lines):
    """
    대사에서 불용어를 제거하는 함수
    """
    cleaned_dialogues = []

    for line in text_lines:
        words = line.strip().split()  # 단어별로 분할
        filtered_words = [word for word in words if word.lower() not in stop_words]  # 불용어 제거
        cleaned_line = " ".join(filtered_words)  # 다시 문장으로 조합
        if cleaned_line:  # 빈 줄은 추가하지 않음
            cleaned_dialogues.append(cleaned_line)

    return cleaned_dialogues

# 📌 수정된 파일 경로 (백슬래시 문제 해결: r"" 또는 os.path 사용)
faust_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\faust_dialogues.txt"
mephi_file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\mephi_dialogues.txt"

with open(faust_file_path, "r", encoding="utf-8") as f:
    faust_text = f.readlines()

with open(mephi_file_path, "r", encoding="utf-8") as f:
    mephi_text = f.readlines()

# 불용어 제거 적용
cleaned_faust_no_stopwords = remove_stopwords(faust_text)
cleaned_mephi_no_stopwords = remove_stopwords(mephi_text)

# 결과 저장
faust_no_stopwords_path = "faust_cleaned_no_stopwords.txt"
mephi_no_stopwords_path = "mephi_cleaned_no_stopwords.txt"

with open(faust_no_stopwords_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_faust_no_stopwords))

with open(mephi_no_stopwords_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_mephi_no_stopwords))

print("✅ 불용어 제거 완료! 정리된 파일이 저장되었습니다.")
