# Sentiment Analysis for Faust

파우스트(Faust)에 등장하는 주요 인물들(파우스트와 메피스토펠레스)의 대사를 분석하여 감정과 언어 사용 패턴을 파악하는 프로젝트입니다.

## 설치 방법

1. 프로젝트 클론
```bash
git clone https://github.com/your-username/Sentiment-analysis-Faust.git
cd Sentiment-analysis-Faust
```

2. 가상환경 생성 및 활성화
```bash
# Windows
python -m venv myenv
myenv\Scripts\activate

# Linux/Mac
python -m venv myenv
source myenv/bin/activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 프로젝트 구조
```
Sentiment-analysis-Faust/
├── preprocess/
│   ├── ONE_stop_words_custom/     # 불용어 처리
│   └── TWO_extract_lines/         # 대사 추출
├── analyze/
│   ├── frequency/                 # 단어 빈도 분석
│   ├── word2vec/                  # 단어 임베딩 분석
│   └── cooccurrence/             # 단어 공동 출현 분석
└── data/                         # 원본 텍스트 데이터
```

## 실행 방법

### 1. 전처리

1) 불용어 처리 및 텍스트 정제
```bash
cd preprocess/ONE_stop_words_custom
python stop_words_custom.py
```

2) 등장인물 대사 추출
```bash
cd ../TWO_extract_lines
python dialogue_extract.py
```

### 2. 분석

1) 단어 빈도 분석
```bash
cd ../../analyze/frequency
python frequency_words.py
```
- 결과: `faust_top_words.png`, `mephisto_top_words.png` 생성

2) Word2Vec 분석
```bash
cd ../word2vec
python word2vec.py
```
- 결과: `merged_w2v.model` 생성
- 파우스트와 메피스토펠레스의 대사에서 'love'와 'devil' 등 주요 단어의 유사어를 분석

3) 단어 공동 출현 분석
```bash
cd ../cooccurrence
python cooccurrence.py
```
- 결과: 주요 단어들의 공동 출현 네트워크 그래프 생성

## 주요 기능

1. **텍스트 전처리**
   - 불필요한 텍스트 제거 (괄호 내용, 특수문자 등)
   - 불용어 제거
   - 등장인물별 대사 추출

2. **단어 빈도 분석**
   - 각 등장인물이 자주 사용하는 단어 시각화
   - 막대 그래프를 통한 빈도 표현

3. **Word2Vec 분석**
   - 각 등장인물의 대사에 대한 Word2Vec 모델 학습
   - 주요 단어들의 문맥적 유사도 분석
   - 통합 모델을 통한 비교 분석

4. **공동 출현 분석**
   - 단어 간 동시 출현 빈도 계산
   - 네트워크 그래프를 통한 관계 시각화

## 결과물

1. **시각화 파일**
   - `faust_top_words.png`: 파우스트의 주요 사용 단어
   - `mephisto_top_words.png`: 메피스토펠레스의 주요 사용 단어
   - 공동 출현 네트워크 그래프

2. **모델 파일**
   - `merged_w2v.model`: 통합된 Word2Vec 모델

## 딥러닝 모델

프로젝트는 세 가지 딥러닝 모델을 제공합니다:

1. **LSTM 모델**
   - 2층 양방향 LSTM
   - 임베딩 + 평균/최대 풀링 결합
   - 드롭아웃과 레이어 정규화 적용

2. **RNN 모델**
   - 기본 RNN 구조
   - Word2Vec 사전 학습 임베딩 사용

3. **Transformer 모델**
   - 멀티헤드 어텐션 기반
   - 포지셔널 인코딩 적용

### 모델 훈련

1. **데이터 준비**
```bash
cd main/dataset
python load_dataset.py
```
- dair-ai/emotion 데이터셋 사용
- 텍스트 토큰화 및 패딩 처리
- Word2Vec 사전 학습 임베딩 적용

2. **LSTM 모델 훈련**
```bash
cd ../LSTM
python run_lstm.py
```
- 하이퍼파라미터:
  - embedding_dim: 100
  - hidden_dim: 128
  - num_epochs: 10
  - batch_size: 64
  - max_len: 50

3. **RNN 모델 훈련**
```bash
cd ../RNN
python run_rnn.py
```

4. **Transformer 모델 훈련**
```bash
cd ../Transformer
python run_transformer.py
```

### 모델 평가 및 예측

1. **LSTM 예측**
```bash
cd main/LSTM
python predict_lstm.py
```

2. **RNN 예측**
```bash
cd main/RNN
python predict.py
```

3. **Transformer 예측**
```bash
cd main/Transformer
python predict_transformer.py
```

### 모델 성능 비교

각 모델은 다음과 같은 평가 지표를 제공합니다:
- 정확도 (Accuracy)
- 손실 (Loss)
- 에포크별 학습 곡선

### 체크포인트

- 모든 모델의 체크포인트는 `main/checkpoint/` 디렉토리에 저장
- 저장 형식: 모델 가중치, 어휘 사전, 레이블 이름 포함

### 모델 최적화 팁

1. **데이터 전처리**
   - 문장 길이 조정 (max_len 파라미터)
   - 어휘 크기 조정 (min_freq 파라미터)
   - 배치 크기 조정

2. **모델 파라미터**
   - 임베딩 차원 (embedding_dim)
   - 은닉층 크기 (hidden_dim)
   - 드롭아웃 비율
   - 학습률

3. **학습 전략**
   - 사전 학습된 Word2Vec 임베딩 사용
   - 조기 종료 (Early Stopping)
   - 학습률 스케줄링

## 참고사항

- 원본 텍스트는 Project Gutenberg의 영문 번역본을 사용
- Python 3.8 이상 권장
- 대용량 텍스트 처리 시 충분한 메모리 필요
