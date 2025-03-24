from collections import defaultdict, Counter
import itertools
import matplotlib.pyplot as plt
import networkx as nx

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frequency.frequency_words import get_word_frequencies

def build_cooccurrence_matrix(text_lines, window_size=2):
    """
    각 줄마다 등장한 단어들의 co-occurrence를 계산 (window_size는 블록 단위)
    """
    cooccur = defaultdict(Counter)

    for line in text_lines:
        words = line.strip().split()
        for w1, w2 in itertools.combinations(set(words), 2):  # 순서 없이 쌍만 고려
            cooccur[w1][w2] += 1
            cooccur[w2][w1] += 1
    return cooccur

# 파일 열기
with open(r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\THREE_NLTK_stop_words\faust_dialogues_final.txt", "r", encoding="utf-8") as f:
    faust_lines = f.readlines()
with open(r"C:\Users\박규민\OneDrive - KookminUNIV\바탕 화면\빅데이터 최신기술\Sentiment-analysis-Faust\preprocess\THREE_NLTK_stop_words\mephisto_dialogues_final.txt", "r", encoding="utf-8") as f:
    mephi_lines = f.readlines()

# Co-occurrence 분석 수행
faust_cooccur = build_cooccurrence_matrix(faust_lines)
mephi_cooccur = build_cooccurrence_matrix(mephi_lines)

# 단어 빈도수 계산
faust_word_counts = get_word_frequencies(faust_lines)
mephi_word_counts = get_word_frequencies(mephi_lines)


def plot_cooccurrence_graph(cooccur_dict, target_word, top_n=10):
    G = nx.Graph()
    top_related = cooccur_dict[target_word].most_common(top_n)

    for word, freq in top_related:
        G.add_edge(target_word, word, weight=freq)

    pos = nx.spring_layout(G, seed=42)
    weights = [G[u][v]['weight'] for u,v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=weights, node_color='skyblue', edge_color='gray', font_size=10)
    plt.title(f"Co-occurrence Graph: '{target_word}'")
    plt.show()


def print_top_cooccurrences(word_counts, cooccur_dict, top_n=20, related_n=10, speaker="Faust"):
    """
    상위 top_n 단어 각각에 대해, 함께 자주 등장한 related_n개 단어 출력
    """
    top_words = word_counts.most_common(top_n)

    for word, count in top_words:
        if word not in cooccur_dict:
            continue  # 혹시 누락된 단어 있으면 스킵

        related = cooccur_dict[word].most_common(related_n)
        related_str = ', '.join([f"{w} ({c})" for w, c in related])

        print(f"📌 {speaker} 단어: '{word}' (빈도: {count})")
        print(f"   → 자주 같이 등장한 단어들: {related_str}")
        print("-" * 80)

# 예시 실행
print_top_cooccurrences(faust_word_counts, faust_cooccur, top_n=20, related_n=10, speaker="Faust")
print_top_cooccurrences(mephi_word_counts, mephi_cooccur, top_n=20, related_n=10, speaker="Mephistopheles")
