import re

# 파일 읽기
file_path = "..\data\Faust [part 1]. Translated Into English in the Original Metres by Goethe.txt"  # 여기에 네 파일 경로 넣어
with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

# 파우스트의 대사만 추출
faust_pattern = r"MEPHISTOPHELES\n(.*?)(?=\n[A-Z]+\n|\Z)"  # FAUST 다음에 나오는 대사 찾기
faust_lines = re.findall(faust_pattern, text, re.DOTALL)

# 정리 후 저장
with open("mephi_dialogues.txt", "w", encoding="utf-8") as f:
    for dialogue in faust_lines:
        f.write(dialogue.strip() + "\n\n")

print("메피스토의 대사만 추출 완료!")
