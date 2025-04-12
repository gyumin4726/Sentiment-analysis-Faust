import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from RNN.model import RNN
from LSTM.model_lstm import LSTM
from Transformer.model_transformer import TransformerClassifier
from RNN.predict import predict_sentiment  # 모든 모델에 동일한 predict 사용

# -----------------------------
# 구성: 통합 테스트 실행 모듈화 (RNN, LSTM, Transformer)
# -----------------------------

def load_model(model_class, checkpoint_path, embedding_dim=100, hidden_dim=128, device='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab = checkpoint['vocab']
    label_names = checkpoint['label_names']
    input_dim = len(vocab)
    output_dim = len(label_names)
    pad_idx = vocab['<pad>']

    model = model_class(input_dim, embedding_dim, hidden_dim, output_dim, pad_idx)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, vocab, label_names


def evaluate_sentences(model, vocab, label_names, sentences, max_len=50, device='cpu', speaker="", model_name=""):
    print(f"\n========== [{model_name}] {speaker} 대표 문장 감정 예측 ==========")
    for i, sentence in enumerate(sentences, 1):
        label, confidence, _ = predict_sentiment(sentence, model, vocab, label_names, max_len=max_len, device=device)
        print(f"{i}. {sentence}")
        print(f"   → 예측 감정: {label} (확률: {confidence:.4f})\n")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 공통 문장 리스트
    faust_sentences = [
        "meanwhile, may not the treasure risen be, which there, behind, i glimmering see?",
        "if hopeful fancy once, in daring flight, her longings to the infinite expanded, yet now a narrow space contents her quite, since time’s wild wave so many a fortune stranded.",
        "why some inexplicable smart all movement of my life impedes?",
        "little sister, so good, laid my bones in the wood, in the damp moss and clay: then was i a beautiful bird o’ the wood; fly away!",
        "the wild desires no longer win us, the deeds of passion cease to chain; the love of man revives within us, the love of god revives again.",
        "yet, finally, the weary god is sinking; the new-born impulse fires my mind,— i hasten on, his beams eternal drinking, the day before me and the night behind, above me heaven unfurled, the floor of waves beneath me,— a glorious dream!",
        "there was a lion red, a wooer daring, within the lily’s tepid bath espoused, and both, tormented then by flame unsparing, by turns in either bridal chamber housed.",
        "i see thee, and the stings of pain diminish; i grasp thee, and my struggles slowly finish: my spirit’s flood-tide ebbeth more and more.",
        "my shrinking but lingers death more near.",
        "though i possessed the power to draw thee near me, the power to keep thee was denied my hand."
    ]

    mephisto_sentences = [
        "i lately gave therein a squint— saw splendid lion-dollars in ’t.",
        "under the old ribs of the rock retreating, hold fast, lest thou be hurled down the abysses there!",
        "he cried: “i find my conduct wholly hateful!",
        "all unprepared, the charm i spin: we’re here together, so begin!",
        "of god, the world and all that in it has a place, of man, and all that moves the being of his race, have you not terms and definitions given with brazen forehead, daring breast?",
        "thou hast no doubt about my noble blood: see, here’s the coat-of-arms that i am wearing!",
        "why such a noise?",
        "i like, at times, to hear the ancient’s word, and have a care to be most civil: it’s really kind of such a noble lord so humanly to gossip with the devil!",
        "i use no lengthened invocation: here rustles one that soon will work my liberation.",
        "the scholars are everywhere believers, but never succeed in being weavers."
    ]

    # RNN 실행
    model, vocab, label_names = load_model(RNN, "../../checkpoint/saved_rnn_model.pth", device=device)
    evaluate_sentences(model, vocab, label_names, faust_sentences, device=device, speaker="Faust", model_name="RNN")
    evaluate_sentences(model, vocab, label_names, mephisto_sentences, device=device, speaker="Mephistopheles", model_name="RNN")

    # LSTM 실행
    model, vocab, label_names = load_model(LSTM, "../../checkpoint/saved_lstm_model.pth", device=device)
    evaluate_sentences(model, vocab, label_names, faust_sentences, device=device, speaker="Faust", model_name="LSTM")
    evaluate_sentences(model, vocab, label_names, mephisto_sentences, device=device, speaker="Mephistopheles", model_name="LSTM")

    # Transformer 실행
    model, vocab, label_names = load_model(TransformerClassifier, "../../checkpoint/saved_transformer_model.pth", device=device)
    evaluate_sentences(model, vocab, label_names, faust_sentences, device=device, speaker="Faust", model_name="Transformer")
    evaluate_sentences(model, vocab, label_names, mephisto_sentences, device=device, speaker="Mephistopheles", model_name="Transformer")
