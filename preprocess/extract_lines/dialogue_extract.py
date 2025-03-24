import re

file_path = r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-faust\preprocess\stop_words_custom\faust_cleaned_no_stopwords.txt"

mephistopheles_dialogues = []
current_speaker = None
current_lines = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()

        # 등장인물 이름이면 화자 전환 (공백 줄은 스킵)
        if stripped and re.fullmatch(r'[a-z][a-z\s]*', stripped) and len(stripped.split()) <= 3:
            # 직전 화자가 mephistopheles면 대사 저장
            if current_speaker == 'mephistopheles' and current_lines:
                mephistopheles_dialogues.append(' '.join(current_lines).strip())
            current_speaker = stripped
            current_lines = []
        else:
            # 현재 화자가 mephistopheles이면 줄 저장 (빈 줄 포함)
            if current_speaker == 'mephistopheles':
                current_lines.append(stripped)

# 마지막 블록도 저장
if current_speaker == 'mephistopheles' and current_lines:
    mephistopheles_dialogues.append(' '.join(current_lines).strip())

# 저장
output_path = "mephistopheles_dialogues.txt"
with open(output_path, "w", encoding="utf-8") as f:
    for dialogue in mephistopheles_dialogues:
        f.write(dialogue + "\n\n")

print("✅ 파우스트의 빈 줄 포함 전체 대사 완벽 추출 완료!")
