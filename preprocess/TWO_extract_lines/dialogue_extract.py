import re

file_path = "../ONE_stop_words_custom/faust_cleaned_no_stopwords.txt"

# 저장할 대사 리스트 초기화
faust_dialogues = []
mephisto_dialogues = []

# 문장 단위로 분리하는 함수
def split_into_sentences(text):
    # 문장 끝나는 지점에서 split (단, 약어 등에 대응은 미약함)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

current_speaker = None
current_lines = []

with open(file_path, "r", encoding="utf-8") as f:
    for line in f:
        stripped = line.strip()

        # 등장인물 이름일 경우
        if stripped and re.fullmatch(r'[a-z][a-z\s]*', stripped) and len(stripped.split()) <= 3:
            # 이전 화자 대사를 저장
            if current_speaker == 'faust' and current_lines:
                dialogue = ' '.join(current_lines).strip()
                faust_dialogues.extend(split_into_sentences(dialogue))
            elif current_speaker == 'mephistopheles' and current_lines:
                dialogue = ' '.join(current_lines).strip()
                mephisto_dialogues.extend(split_into_sentences(dialogue))

            # 화자 갱신
            current_speaker = stripped
            current_lines = []
        else:
            if current_speaker in ['faust', 'mephistopheles']:
                current_lines.append(stripped)

# 마지막 대사 블록 저장
if current_speaker == 'faust' and current_lines:
    dialogue = ' '.join(current_lines).strip()
    faust_dialogues.extend(split_into_sentences(dialogue))
elif current_speaker == 'mephistopheles' and current_lines:
    dialogue = ' '.join(current_lines).strip()
    mephisto_dialogues.extend(split_into_sentences(dialogue))

# 파일 저장
with open("faust_dialogues.txt", "w", encoding="utf-8") as f:
    for sentence in faust_dialogues:
        f.write(sentence + "\n")

with open("mephisto_dialogues.txt", "w", encoding="utf-8") as f:
    for sentence in mephisto_dialogues:
        f.write(sentence + "\n")

print("✅ faust & mephistopheles 대사 문장 단위로 분리하여 저장 완료!")
