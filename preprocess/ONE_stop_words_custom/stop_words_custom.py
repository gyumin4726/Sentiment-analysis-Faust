import re
from collections import Counter
import string

# 제거할 불필요한 단어 리스트 (고전 영어 + 일반 불용어)
custom_stopwords = [
    "illustration"
    ]


def clean_text(text):
    """
    1. [ ] 및 ( ) 안에 있는 내용을 제거
    2. 소문자로 변환
    3. 불용어 제거
    """
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)  # [ ]와 ( ) 안의 내용 제거
    text = text.lower()  # 소문자로 변환
    words = text.split()  # 단어 단위로 분할
    filtered_words = [word for word in words if word not in custom_stopwords]  # 불용어 제거
    return " ".join(filtered_words)  # 다시 문장으로 변환

def get_filtered_word_frequencies(text_lines):
    """
    텍스트에서 불용어를 제거하고 [ ], ( ) 안의 내용을 삭제한 후 단어 빈도수 계산
    """
    all_words = []
    
    for line in text_lines:
        cleaned_line = clean_text(line)
        words = cleaned_line.split()  # 단어 단위로 분할
        all_words.extend(words)  # 리스트에 추가
    
    word_counts = Counter(all_words)  # 단어 빈도수 계산
    return word_counts

# 수정된 파일 경로 (백슬래시 문제 해결: r"" 또는 os.path 사용)
file_path = "../../data/Faust [part 1]. Translated Into English in the Original Metres by Goethe.txt"

with open(file_path, "r", encoding="utf-8") as f:
    faust_text = f.readlines()

cleaned_no_stopwords = [clean_text(line) for line in faust_text]

# 불용어 제거 + 대사 정제 후 단어 빈도수 계산
filtered_faust_word_counts = get_filtered_word_frequencies(faust_text)

# 상위 20개 단어 출력
print("파우스트 대사에서 가장 많이 나온 단어 20개 (불용어 제거 + 정제 후):")
print(filtered_faust_word_counts.most_common(20))

# 결과 저장
faust_no_stopwords_path = "faust_cleaned_no_stopwords.txt"
mephi_no_stopwords_path = "mephi_cleaned_no_stopwords.txt"

with open(faust_no_stopwords_path, "w", encoding="utf-8") as f:
    f.write("\n".join(cleaned_no_stopwords))


print("불용어 제거 완료! 정리된 파일이 저장되었습니다.")