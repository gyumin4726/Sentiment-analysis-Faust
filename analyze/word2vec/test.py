from word2vec import model1
from word2vec import model2

from scipy.spatial.distance import cosine

word = "god"
if word in model1.wv and word in model2.wv:
    sim = 1 - cosine(model1.wv[word], model2.wv[word])
    print(f"💡 'god' 단어의 두 화자 간 의미 차이 (유사도): {sim:.4f}")
