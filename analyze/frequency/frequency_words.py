from collections import Counter
import string
import os

def get_word_frequencies(text_lines):
    """
    텍스트에서 단어 빈도수를 계산하는 함수
    """
    all_words = []
    
    for line in text_lines:
        line = line.translate(str.maketrans("", "", string.punctuation))  # 문장 부호 제거
        words = line.split()  # 단어 단위로 분할
        all_words.extend(words)  # 리스트에 추가
    
    word_counts = Counter(all_words)  # 단어 빈도수 계산
    return word_counts

# 수정된 파일 경로 (백슬래시 문제 해결: r"" 또는 os.path 사용)
faust_file_path = "../../preprocess/TWO_extract_lines/faust_dialogues.txt"
mephi_file_path = "../../preprocess/TWO_extract_lines/mephisto_dialogues.txt"


# 파일 읽기
with open(faust_file_path, "r", encoding="utf-8", errors="replace") as f:
    faust_text = f.readlines()

with open(mephi_file_path, "r", encoding="utf-8", errors="replace") as f:
    mephi_text = f.readlines()



import matplotlib.pyplot as plt

def plot_top_words(word_counts, title, color, filename):
    """
    단어 빈도 TOP 20등까지 막대그래프로 시각화
    """
    top_words = word_counts.most_common(20)  # 1등 제외
    words, counts = zip(*top_words)

    plt.figure(figsize=(10, 5))
    plt.bar(words, counts, color=color)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.xlabel("word")
    plt.ylabel("frequency")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # 고해상도 저장
    plt.show()

if __name__ == "__main__":
    # 단어 빈도수 계산
    faust_word_counts = get_word_frequencies(faust_text)
    mephi_word_counts = get_word_frequencies(mephi_text)

    # 상위 20개 단어 출력
    print("파우스트 대사에서 가장 많이 나온 단어 20개:")
    for word, count in faust_word_counts.most_common(20):
        print(f"{word}: {count}")

    print("\n메피스토펠레스 대사에서 가장 많이 나온 단어 20개:")
    for word, count in mephi_word_counts.most_common(20):
        print(f"{word}: {count}") 
        
    # 시각화 실행 + 저장
    plot_top_words(faust_word_counts, "Faust Dialogue Word Frequency", "skyblue", "faust_top_words.png")
    plot_top_words(mephi_word_counts, "Mephistopheles Dialogue Word Frequency", "salmon", "mephisto_top_words.png")
