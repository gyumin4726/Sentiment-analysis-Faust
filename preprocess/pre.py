import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK 데이터 다운로드 (최초 1회 실행 필요)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# 파일 불러오기
file_path = "..\data\Faust [part 1]. Translated Into English in the Original Metres by Goethe.txt"  # 파일 경로 수정 필요
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# 1️⃣ 소문자 변환
text = text.lower()

# 2️⃣ 특수문자, 숫자 제거 (알파벳과 공백만 남김)
text = re.sub(r'[^a-zA-Z\s]', '', text)

# 3️⃣ 단어 토큰화
tokens = word_tokenize(text)

# 4️⃣ 불용어 제거 (stopwords)
stop_words = set(stopwords.words('english'))  # 영어 불용어 리스트
filtered_tokens = [word for word in tokens if word not in stop_words]

# 5️⃣ 전처리된 텍스트 저장
processed_text = " ".join(filtered_tokens)

# 결과 확인
print("✅ 전처리 완료! 일부 결과 확인:")
print(processed_text[:500])  # 앞부분 500자 출력

# 필요하면 전처리된 텍스트를 파일로 저장
with open("Faust_processed.txt", "w", encoding="utf-8") as output_file:
    output_file.write(processed_text)
