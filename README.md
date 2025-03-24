# Sentiment-analysis-Faust

## 데이터
https://www.gutenberg.org/

## 전처리 과정

1. 전체 텍스트에서 [] 및 ( ) 안에 있는 내용들, 문자 부호, custom 불용어 제거
2. 주요 등장인물 (파우스트 및 메피스토펠레스) 대사 추출
3. 추출된 대사 텍스트에 대해 NLTK 기반의 토큰화(tokenization)와 불용어(stopwords) 제거를 수행하여, 감정 분석에 핵심적인 단어 중심으로 데이터를 정제

