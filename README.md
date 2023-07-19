

# 한국어 기반 MBTI 성격유형 분류 모델 정확도 향상을 위한 연구
> 2022-2학기 빅데이터/인공지능 융합전공 캡스톤 디자인 <br><br>
> 유튜브 댓글 크롤링으로 57,000개의 데이터를 수집, 전처리와 임베딩, 모델 설계, 학습을 통해 정확도 개선
<br>

## 1. 소개 
> 진행 기간: 2022. 09 ~ 2022. 12 <br>


<table>
  <tr>
    <td>서은효👑</td>
    <td>김정현</td>
    <td>박윤지</td>
  </tr>
  
  <tr>
    <td>모델 설계 및 training <br> </td>
    <td>크롤링 데이터 수집 <br>  </td>
    <td>데이터 전처리</td>
    

  </tr>

</table>

<br>

- 기존의 한국어 기반 MBTI 성격유형 분류 모델 연구는 33,000개의 데이터로 20.29%의 정확도를 도출했다. <br>
- 한국어 기반 MBTI 분류 모델의 정확도를 높이기 위하여 유튜브 댓글에서 크롤링을 통해 77,606개의 데이터를 수집하였고, 해당 데이터를 바탕으로 전처리, 임베딩, 모델 설계, 학습을 수행하였다. <br>
- 학습 결과 **약 57,000개의 데이터로 40%의 정확도를 도출했으며 정확도가 개선**되었음을 확인할 수 있었다.

<br>

<br>

## 2. 사용 기술 

#### Library
- `keras`
- `numpy`
- `sklearn`
- `pandas`
- `tensorflow`
- `selenium`
- `Beautiful Soup`
- `konlpy`
- `tqdm`
- `Counter`
- `os`
- `re`
- [비속어 사전 Open Source](https://github.com/organization/Gentleman)
#### Train
- `GPU server`

#### Language
- Anaconda3 기반 `Python 3.9.0`



<br>

## 3. 관련 연구

#### [1] [딥러닝 기반의 MBTI 성격유형 분류 연구](https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE11108463)

총 **33,000개의 데이터**로 **LSTM**을 이용해 **최초의 한국어 기반 MBTI 예측 모델**을 만들었다. **출력 layer의 activation 함수를 sigmoid로 설정해 0에 가까우면 E, N, F, P 유형, 1에 가까우면 I, S, T, J 유형**으로 결과를 나타냈으며 binary 방식으로 접근해 E/I, N/S, F/T, P/J 당 65.33%, 66.88%, 66.07%, 70.32%의 정확도를, **최종 전체 정확도는 20.29%** 를 산출하였다.
<br> 

#### [2] [유튜브 악성 댓글 탐지를 위한 LSTM 기반 기계학습 시스템 설계 및 구현](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002826308)

유튜브 콘텐츠에 대한 악성 댓글 판별 및 시각화를 위해 **LSTM** 기반의 자연어 처리를 이용했다. 크롤링을 이용한 댓글 수집, 데이터 전처리, LSTM 학습 모델 구축 등의 순서를 거쳐 **약 5만 개의 학습 데이터**로 LSTM 모델을 구축했다. 약 **92%의 정확도** 를 얻었으며 데이터를 전처리하는 과정에서 댓글에 특화된 감성 사전을 구축하여 감성 분석에 이용할 수 있음을 확인하는 결과를 도출했다.

<br><br>

## 4. 문제 해결 방법

📌 본 프로젝트는 multi-class 분류를 이용해 선행 연구[1] 대비 MBTI 분류 정확도를 높이는 것에 목적을 둠
<br>

#### 1. 크롤링(Crawling)
- 학습을 위한 데이터는 각 MBTI 유형과 관련된 유튜브 영상의 댓글을 크롤링
- 크롤링한 댓글들을 데이터 프레임의 형태로 각각 아이디와 댓글 내용으로 나누어 csv 파일로 저장
- 172개의 유튜브 영상에서 총 77,606개의 댓글 데이터들을 수집

<br> 

#### 2. 전처리 
- 라벨링 및 결측값 제거
	- 16가지의 MBTI 유형인 ENFJ, ENFP, ENTJ, ENTP, ESFJ, ESFP, ESTJ, ESTP, INFJ, INFP, INTJ, INTP, ISFJ, ISFP, ISTJ, ISTP를 각각 0 ~ 15로 라벨링 
	- pandas를 이용해 결측값을 제거(라벨링 과정으로 분류된 데이터 중 중복 값은 총 7개, 결측값은 총 48개로 확인 후 제거)
	- 댓글 길이가 12글자 이하인 댓글들은 학습에 유의미한 내용이 아니라고 판단하여 모두 제거
<br>

- 정규 표현식
  	- 완성형 한글의 범위 [^가-힣]를 제외한 문자들을 모두 제거
	- 영어와 숫자, 각종 특수문자를 제거
<br>

- 토큰화 및 표제어 추출
   	- KoNLPy의 Okt 형태소 분석기를 이용해 불용어(조사, 접속사, 초성 등)제거
   	- Embedding layer를 위해 토큰화된 단어들을 표제어로 변환
   	- morphs()를 이용하여 텍스트를 형태소 단위로 나누고 stem옵션을 이용하여 어간 추출

<br>

- 기타 불용어 제거
	- Github 오픈소스 비속어 사전을 이용하여 비속어를 제거
 	- 빈도수 분석을 통해 빈도수가 5 이하인 단어와 한 글자 단어 제거
  	- 특정 MBTI를 한국어로 나타내는 용어들(예: 엔프피, 엔프제, 인프제 등)은 별도 정의
  	- 0 ~ 15까지 부여했던 label은 one hot encoding을 통해 벡터로 구현
 <img width="356" alt="image" src="https://github.com/seoeunhyo/NLP-Project/assets/93567740/18736af4-9f75-4246-a03e-410499cf1606">

<br> 
<br>

#### 3. 모델(Model)
🧾 임베딩 벡터의 차원은 100, 에폭은 15, batch_size는 256, 훈련 데이터의 20%를 검증 데이터로 사용했으며 학습 도중 손실 함수가 정확도 대비 3번 이상 높게 반복될 경우 학습이 조기 종료되도록 설계

- CNN
- LSTM
- CNN-LSTM
- BiLSTM
<img width="113" alt="image" src="https://github.com/seoeunhyo/NLP-Project/assets/93567740/66d8f864-d5b0-4540-bd10-039034be2e5f">

<br>
<br>

#### 4. 분석 결과 
<img width="309" alt="image" src="https://github.com/seoeunhyo/NLP-Project/assets/93567740/d7902fc8-b62f-4dd0-a625-30d9541c6543">

정확도는 CNN, BiLSTM, LSTM, CNN-LSTM 순서로 높았으며 텍스트에서 특징 추출이 잘 이루어질 것이라 예상했던 CNN이 가장 높은 정확도를 보였다. 
<br>
<br>


#### 5. 결론 

- 기존 한국어 기반 MBTI 성격유형 분류 모델 연구의 정확도를 높이기 위해 유튜브 댓글에서 약 77,000개를 수집했고, 전처리 과정을 거쳐 57,000개의 데이터로 최대 46%의 정확도를 도출

- 약 26%의 정확도 향상
- 향후 학습 데이터를 유튜브 댓글로 국한하지 않고 카페나 커뮤니티 등 다양한 플랫폼에서 데이터를 수집해 결과를 도출해 볼 예정
