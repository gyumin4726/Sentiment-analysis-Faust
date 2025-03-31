import re

file_path = "../ONE_stop_words_custom/faust_cleaned_no_stopwords.txt"

# 저장할 대사 리스트 초기화
faust_dialogues = []
mephisto_dialogues = []

current_speaker = None
current_lines = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()

        # 등장인물 이름일 경우 (예: faust, mephistopheles 등)
        if stripped and re.fullmatch(r'[a-z][a-z\s]*', stripped) and len(stripped.split()) <= 3:
            # 이전 화자가 faust 혹은 mephistopheles면 저장
            if current_speaker == 'faust' and current_lines:
                faust_dialogues.append(' '.join(current_lines).strip())
            elif current_speaker == 'mephistopheles' and current_lines:
                mephisto_dialogues.append(' '.join(current_lines).strip())

            # 화자 갱신
            current_speaker = stripped
            current_lines = []
        else:
            if current_speaker in ['faust', 'mephistopheles']:
                current_lines.append(stripped)

# 마지막 대사 블록 저장
if current_speaker == 'faust' and current_lines:
    faust_dialogues.append(' '.join(current_lines).strip())
elif current_speaker == 'mephistopheles' and current_lines:
    mephisto_dialogues.append(' '.join(current_lines).strip())

# 파일 저장
with open("faust_dialogues.txt", "w", encoding="utf-8") as f:
    for dialogue in faust_dialogues:
        f.write(dialogue + "\n\n")

with open("mephisto_dialogues.txt", "w", encoding="utf-8") as f:
    for dialogue in mephisto_dialogues:
        f.write(dialogue + "\n\n")

print("✅ faust & mephistopheles 대사 각각 추출 및 저장 완료!")
