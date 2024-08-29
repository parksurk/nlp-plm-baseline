# NLP PLM Baseline Code 

## Text Classification - 상품리뷰에 에 대한 긍정/부정 분석(Positive Negative)

이 저장소에는 순환 신경망(LSTM)과 합성곱 신경망(CNN)을 사용한 단순한 텍스트 분류의 구현이 포함되어 있습니다([Kim 2014](http://arxiv.org/abs/1408.5882) 참조). 학습할 아키텍처를 지정해야 하며, 두 가지를 모두 선택할 수 있습니다. 두 아키텍처를 모두 선택하여 문장을 분류하면 단순 평균으로 앙상블 추론이 이루어집니다.


## 사전 요구 사항

- Python 3.6 이상
- PyTorch 1.6 이상
- PyTorch Ignite
- TorchText 0.5 이상
- [torch-optimizer 0.0.1a15](https://pypi.org/project/torch-optimizer/)
- 토크나이즈된 코퍼스(예: [Moses](https://www.nltk.org/_modules/nltk/tokenize/moses.html), Mecab, [Jieba](https://github.com/fxsjy/jieba))

BERT 파인튜닝을 사용하려면 다음도 필요할 수 있습니다.

- Huggingface

추가 요구사항
- SKlearn
- WandB

## 설치

### 로컬PC Python 가상환경 설정
conda 환경을 새로 생성한 후 필요한 라이브러리를 설치합니다.

```bash
conda create -n nlp-plm python=3.12
conda activate nlp-plm
conda install pytorch torchvision torchaudio torchtext -c pytorch
conda install -c pytorch ignite
conda install packaging
pip install torch_optimizer
conda install -c conda-forge transformers
pip install scikit-learn
pip install wandb
```
### Colab 설정
```bash

# PyTorch 및 관련 패키지 설치
!pip install torch torchvision torchaudio

# torch_optimizer 설치
!pip install torch_optimizer

# ignite 설치
!pip install pytorch-ignite

# Transformers 설치
!pip install transformers

# SKLearn
!pip install scikit-learn

# WandB 설치
!pip install wandb

```

## 사용 방법

### 준비

#### 형식

입력 파일은 클래스와 문장 두 개의 열로 구성되며, 이 열들은 탭으로 구분됩니다. 클래스는 숫자가 아니어도 되며, 공백 없이 단어로 작성될 수 있습니다. 아래는 예제 코퍼스입니다.

```bash
$ cat ./data/raw_corpus.txt | shuf | head
positive	나름 괜찬항요 막 엄청 좋은건 아님 그냥 그럭저럭임... 아직 까지 인생 디퓨져는 못찾은느낌
negative	재질은플라스틱부분이많고요...금방깨질거같아요..당장 물은나오게해야하기에..그냥설치했어요..지금도 조금은후회중.....
positive	평소 신던 신발보다 크긴하지만 운동화라 끈 조절해서 신으려구요 신발 이쁘고 편하네요
positive	두개사서 직장에 구비해두고 먹고있어요 양 많아서 오래쓸듯
positive	생일선물로 샀는데 받으시는 분도 만족하시구 배송도 빨라서 좋았네요
positive	아이가 너무 좋아합니다 크롱도 좋아라하지만 루피를 더..
negative	배송은 기다릴수 있었는데 8개나 주문했는데 샘플을 너무 적게보내주시네요ㅡㅡ;;
positive	너무귀여워요~~ㅎ아직사용은 못해? f지만 이젠 모기땜에 잠설치는일은 ? j겟죠
positive	13개월 아가 제일좋은 간식이네요
positive	지인추천으로 샀어요~ 싸고 가성비 좋다해서 낮기저귀로 써보려구요~
```

#### 토크나이징(선택 사항)

코퍼스의 문장을 토크나이징해야 할 수 있습니다. 언어에 따라 자신에게 맞는 토크나이저를 선택해야 합니다(예: 한국어의 경우 Mecab).

```bash
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $2 }' | mecab -O wakati > ./data/tmp.txt
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $1 }' > ./data/tmp_class.txt
$ paste ./data/tmp_class.txt ./data/tmp.txt > ./data/corpus.txt
$ rm ./data/tmp.txt ./data/tmp_class.txt
```

#### 셔플 및 학습/검증 세트 분할

적절한 형식화와 토크나이징 후에는 코퍼스를 학습 세트와 검증 세트로 분할해야 합니다.

```bash
$ wc -l ./data/corpus.txt
302680 ./data/corpus.txt
```

보시다시피, 코퍼스에는 260k개 이상의 샘플이 있습니다.

```bash
$ cat ./data/corpus.txt | shuf > ./data/corpus.shuf.txt
$ head -n 62680 ./data/corpus.shuf.txt > ./data/corpus.test.txt
$ tail -n 240000 ./data/corpus.shuf.txt > ./data/corpus.train.txt
```

이제 240,000개의 학습 세트 샘플과 62,680개의 검증 세트 샘플이 있습니다. MacOS를 사용하는 경우, 'shuf' 대신 'rl' 명령어를 사용할 수 있습니다.

### 학습

아래는 학습을 위한 예제 명령어입니다. 하이퍼파라미터 값은 인수 입력을 통해 자신만의 값을 선택할 수 있습니다.

#### 뉴럴네트워크가 RNN, CNN 일때 학습을 위한 예제 명령어

```bash
python train.py --config_path nlp-plm-ntc-config.xml 
```

#### PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어

```bash
python finetune_plm_native.py --config_path nlp-plm-ntc-config.xml 
```

#### PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어

```bash
python finetune_plm_hftrainer.py --config_path nlp-plm-ntc-config-hftrainer.xml 
```

학습을 위해 아키텍처를 지정해야 합니다. 앙상블 방법을 위해 rnn과 cnn을 모두 선택할 수 있습니다. 또한, 학습에 사용할 장치를 선택할 수 있습니다. CPU만 사용하려면 '--gpu_id' 인수에 기본값인 -1을 입력하면 됩니다.


nlp-plm-ntc-config.xml 에서 기본 하이퍼파라미터를 확인할 수 있습니다.

### 추론

아래와 같이 표준 입력을 추론 입력으로 사용할 수 있습니다. 예측 결과는 탭으로 구분된 두 개의 열(상위 k개의 클래스 및 입력 문장)로 구성됩니다. 결과는 표준 출력으로 표시됩니다.

#### 뉴럴네트워크가 RNN, CNN 일때 추론을 위한 예제 명령어

```bash
$ head ./data/review.sorted.uniq.refined.tok.shuf.test.tsv | awk -F'\t' '{ print $2 }' | python classify.py --config_path nlp-plm-ntc-config.xml 

positive	생각 보다 밝 아요 ㅎㅎ
negative	쓸 대 가 없 네요
positive	깔 금 해요 . 가벼워 요 . 설치 가 쉬워요 . 타 사이트 에 비해 가격 도 저렴 하 답니다 .
positive	크기 나 두께 가 딱 제 가 원 하 던 사이즈 네요 . 책상 의자 가 너무 딱딱 해서 쿠션 감 좋 은 방석 이 필요 하 던 차 에 좋 은 제품 만났 네요 . 냄새 얘기 하 시 는 분 도 더러 있 던데 별로 냄새 안 나 요 .
positive	빠르 고 괜찬 습니다 .
positive	유통 기한 도 넉넉 하 고 좋 아요
positive	좋 은 가격 에 좋 은 상품 잘 쓰 겠 습니다 .
negative	사이트 에서 늘 생리대 사 서 쓰 는데 오늘 처럼 이렇게 비닐 에 포장 되 어 받 아 본 건 처음 입니다 . 위생 용품 이 고 자체 도 비닐 포장 이 건만 소형 박스 에 라도 넣 어 보내 주 시 지 . ..
negative	연결 부분 이 많이 티 가 납니다 . 재질 구김 도 좀 있 습니다 .
positive	애기 태열 때문 에 구매 해서 잘 쓰 고 있 습니다 .
```

#### PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어

```bash
$ head ./data/review.sorted.uniq.refined.tok.shuf.test.tsv | awk -F'\t' '{ print $2 }' | python classify_plm.py --config_path nlp-plm-ntc-config.xml 

positive	생각 보다 밝 아요 ㅎㅎ
negative	쓸 대 가 없 네요
positive	깔 금 해요 . 가벼워 요 . 설치 가 쉬워요 . 타 사이트 에 비해 가격 도 저렴 하 답니다 .
positive	크기 나 두께 가 딱 제 가 원 하 던 사이즈 네요 . 책상 의자 가 너무 딱딱 해서 쿠션 감 좋 은 방석 이 필요 하 던 차 에 좋 은 제품 만났 네요 . 냄새 얘기 하 시 는 분 도 더러 있 던데 별로 냄새 안 나 요 .
positive	빠르 고 괜찬 습니다 .
positive	유통 기한 도 넉넉 하 고 좋 아요
positive	좋 은 가격 에 좋 은 상품 잘 쓰 겠 습니다 .
negative	사이트 에서 늘 생리대 사 서 쓰 는데 오늘 처럼 이렇게 비닐 에 포장 되 어 받 아 본 건 처음 입니다 . 위생 용품 이 고 자체 도 비닐 포장 이 건만 소형 박스 에 라도 넣 어 보내 주 시 지 . ..
negative	연결 부분 이 많이 티 가 납니다 . 재질 구김 도 좀 있 습니다 .
positive	애기 태열 때문 에 구매 해서 잘 쓰 고 있 습니다 .
```
#### PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어

```bash
상동
```



nlp-plm-ntc-config.xml 에서 기본 하이퍼파라미터를 확인할 수 있습니다.


## 평가

저는 코퍼스를 학습 세트와 검증 세트로 분할했습니다. 학습 세트는 240,000줄, 검증 세트는 62,680줄로 샘플링되었습니다. 아키텍처 스냅샷은 아래와 같습니다. 하이퍼파라미터 최적화를 통해 성능을 향상시킬 수 있습니다.

```bash
RNNClassifier(
  (emb): Embedding(35532, 128)
  (rnn): LSTM(128, 256, num_layers=4, batch_first=True, dropout=0.3, bidirectional=True)
  (generator): Linear(in_features=512, out_features=2, bias=True)


  (activation): LogSoftmax()
)
```

```bash
CNNClassifier(
  (emb): Embedding(35532, 256)
  (feature_extractors): ModuleList(
    (0): Sequential(
      (0): Conv2d(1, 100, kernel_size=(3, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
    (1): Sequential(
      (0): Conv2d(1, 100, kernel_size=(4, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
    (2): Sequential(
      (0): Conv2d(1, 100, kernel_size=(5, 256), stride=(1, 1))
      (1): ReLU()
      (2): Dropout(p=0.3, inplace=False)
    )
  )
  (generator): Linear(in_features=300, out_features=2, bias=True)
  (activation): LogSoftmax()
)
```

|아키텍처|테스트 정확도|
|-|-|
|Bi-LSTM|0.9035|
|CNN|0.9090|
|Bi-LSTM + CNN|0.9142|
|KcBERT|0.9598|

## Original 저자

|이름|김기현|
|-|-|
|이메일|pointzz.ki@gmail.com|
|깃허브|https://github.com/kh-kim/|
|링크드인|https://www.linkedin.com/in/ki-hyun-kim/|

## 참고 문헌

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)