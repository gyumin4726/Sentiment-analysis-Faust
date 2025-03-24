import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# 1. NLTK 리소스 다운로드 (최초 1회만)
nltk.download('punkt')
nltk.download('stopwords')

# 2. 불용어 리스트
stop_words = set(stopwords.words('english'))

# 3. 불용어 제거 함수
def remove_stopwords(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered = [word for word in words if word not in stop_words and word not in string.punctuation]
    return ' '.join(filtered)

# 4. 처리할 파일 리스트 [(입력 경로, 출력 경로)]
file_pairs = [
    (r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\TWO_extract_lines\faust_dialogues.txt", "faust_dialogues_final.txt"),
    (r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\TWO_extract_lines\mephisto_dialogues.txt", "mephisto_dialogues_final.txt")
]

# 5. 각각 처리
for input_path, output_path in file_pairs:
    with open(input_path, "r", encoding="utf-8") as infile:
        lines = infile.readlines()

    cleaned_lines = [remove_stopwords(line) for line in lines]

    with open(output_path, "w", encoding="utf-8") as outfile:
        for line in cleaned_lines:
            outfile.write(line.strip() + "\n")

    print(f"✅ '{input_path}' → '{output_path}' 저장 완료!")
