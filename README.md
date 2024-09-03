# NLP PLM Baseline Code 

<!-- TOC -->

- 1. Text Classification - 상품리뷰에 대한 긍정/부정 분석(Positive Negative)
    - 1.1. 사전 요구 사항
    - 1.2. 설치
        - 1.2.1. 로컬PC Python 가상환경 설정
        - 1.2.2. Colab 설정
    - 1.3. 사용 방법
        - 1.3.1. 준비
            - 1.3.1.1. 형식
            - 1.3.1.2. 토크나이징(선택 사항)
            - 1.3.1.3. 셔플 및 학습/검증 세트 분할
        - 1.3.2. 학습
            - 1.3.2.1. 뉴럴네트워크가 RNN, CNN 일때 학습을 위한 예제 명령어
            - 1.3.2.2. PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어
            - 1.3.2.3. PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어
    - 1.4. 추론
        - 1.4.1. 뉴럴네트워크가 RNN, CNN 일때 추론을 위한 예제 명령어
        - 1.4.2. PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어
        - 1.4.3. PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어
    - 1.5. 평가
    - 1.6. Original 저자
    - 1.7. 참고 문헌
- 2. Chat Summarization
    - 2.1. 패키지 설치
    - 2.2. 설정 파일(config.yaml)
        - 2.2.1. `general` 섹션
        - 2.2.2. `inference` 섹션
        - 2.2.3. `tokenizer` 섹션
        - 2.2.4. `training` 섹션
        - 2.2.5. .
        - 2.2.6. 설정 파일을 통해 할 수 있는 일:
    - 2.3. dataset.py
        - 2.3.1. 주요 클래스 및 함수 설명
    - 2.4. 소스 및 디렉토리 구조
        - 2.4.1. Lightening 기반 학습/예측 관련 소스 
        - 2.4.2. Ingite 기반 학습/예측 관련 소스 
        - 2.4.3. Seq2SeqTrainer를 사용하지 않는 Ingite 기반 학습/예측 관련 소스 
    - 2.5. 학습 테스트 관련 커맨드 라인 명령어
        - 2.5.1. Lightening 기반 학습
        - 2.5.2. Lightening 기반 예측
        - 2.5.3. Ignite 기반 학습
        - 2.5.4. Ignite 기반 예측
        - 2.5.5. Seq2SeqTrainer를 사용하지 않는 Ignite 기반 학습
        - 2.5.6. Seq2SeqTrainer를 사용하지 않는 Ignite 기반 예측
    - 2.6. Lightening 대신 Ignite 적용 관련
        - 2.6.1. 학습 소스 변경
            - 2.6.1.1. 주요 변경 사항 설명
        - 2.6.2. 예측 소스 변경
            - 2.6.2.1. 주요 변경 사항 설명
    - 2.7. Ignite로 변환된 소스에 WandB 추가 설정 
        - 2.7.1. 추가된 기능 설명:
        - 2.7.2. 실행 시 `wandb` 대시보드에서 다음과 같은 정보를 확인할 수 있습니다:
            - 2.7.2.1. 추가된 `ROUGE` 메트릭 모니터링 기능 설명:
            - 2.7.2.2. 이 코드의 결과:
    - 2.8. Ignite로 변환된 소스에 체크포인트 관련 추가 코딩 
        - 2.8.1. Seq2SeqTrainer의 체크포인트 관리
    - 2.9. Stage server에서 실험 진행 순서
        - 2.9.1. 체크포인트 경로 확인
        - 2.9.2. 모델 로딩 오류 해결
        - 2.9.3. Lightening 기반 학습 및 추론 진행
        - 2.9.4. Ignite 기반 학습 및 추론 진행
        - 2.9.5. 결과 확인 및 로깅
        - 2.9.6. 성능 평가 및 튜닝
        - 2.9.7. 요약
    - 2.10. Early Stopping 관련
        - 2.10.1. Early Stopping의 동작 원리
            - 2.10.1.1. 주요 매개변수:
        - 2.10.2. Early Stopping이 동작하는 예시
        - 2.10.3. Early Stopping 결론
    - 2.11. Ignite를 활용한 커스마이징 범위 정리
        - 2.11.1. `ignite_trainer`와 `ignite_evaluator` 커스터마이징:
        - 2.11.2. 이벤트 핸들러를 통한 커스터마이징
            - 2.11.2.1. 에포크 완료 시 로직 (`EPOCH_COMPLETED` 이벤트)
            - 2.11.2.2. `iteration` 완료 시 로직 (`ITERATION_COMPLETED` 이벤트)
            - 2.11.2.3. 평가 완료 시 로직 (`COMPLETED` 이벤트)
        - 2.11.3. Ignite를 활용한 커스마이징 요약:
    - 2.12. Hugging Faces Transformers 가 제공하는 Seq2SeqTrainer 커스마징 범위
        - 2.12.1. 필요한 모듈 불러오기
        - 2.12.2. 모델 및 데이터 로드
        - 2.12.3. Ignite 기반 학습 및 평가 루프 구현
        - 2.12.4. 학습 과정 설정
        - 2.12.5. 핵심 커스터마이징 사항
    - 2.13. Hugging Faces Transformers 가 제공하는 Seq2SeqTrainer 없이 커스터마이지이 하기
        - 2.13.1. 모델과 토크나이저 로드
        - 2.13.2. 훈련 및 평가 엔진 생성
        - 2.13.3. 훈련 루프 및 체크포인트 저장
        - 2.13.4. 평가 및 ROUGE 점수 계산
        - 2.13.5. 학습 시작
        - 2.13.6. 요약

<!-- /TOC -->

## 1. Text Classification - 상품리뷰에 대한 긍정/부정 분석(Positive Negative)

이 저장소에는 순환 신경망(LSTM)과 합성곱 신경망(CNN)을 사용한 단순한 텍스트 분류의 구현이 포함되어 있습니다([Kim 2014](http://arxiv.org/abs/1408.5882) 참조). 학습할 아키텍처를 지정해야 하며, 두 가지를 모두 선택할 수 있습니다. 두 아키텍처를 모두 선택하여 문장을 분류하면 단순 평균으로 앙상블 추론이 이루어집니다.

### 1.1. 사전 요구 사항

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

### 1.2. 설치

#### 1.2.1. 로컬PC Python 가상환경 설정
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

#### 1.2.2. Colab 설정
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

### 1.3. 사용 방법

#### 1.3.1. 준비

##### 1.3.1.1. 형식

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

##### 1.3.1.2. 토크나이징(선택 사항)

코퍼스의 문장을 토크나이징해야 할 수 있습니다. 언어에 따라 자신에게 맞는 토크나이저를 선택해야 합니다(예: 한국어의 경우 Mecab).

```bash
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $2 }' | mecab -O wakati > ./data/tmp.txt
$ cat ./data/raw_corpus.txt | awk -F'\t' '{ print $1 }' > ./data/tmp_class.txt
$ paste ./data/tmp_class.txt ./data/tmp.txt > ./data/corpus.txt
$ rm ./data/tmp.txt ./data/tmp_class.txt
```

##### 1.3.1.3. 셔플 및 학습/검증 세트 분할

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

#### 1.3.2. 학습

아래는 학습을 위한 예제 명령어입니다. 하이퍼파라미터 값은 인수 입력을 통해 자신만의 값을 선택할 수 있습니다.

##### 1.3.2.1. 뉴럴네트워크가 RNN, CNN 일때 학습을 위한 예제 명령어

```bash
python train.py --config_path nlp-plm-ntc-config.xml 
```

##### 1.3.2.2. PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어

```bash
python finetune_plm_native.py --config_path nlp-plm-ntc-config.xml 
```

##### 1.3.2.3. PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어

```bash
python finetune_plm_hftrainer.py --config_path nlp-plm-ntc-config-hftrainer.xml 
```

학습을 위해 아키텍처를 지정해야 합니다. 앙상블 방법을 위해 rnn과 cnn을 모두 선택할 수 있습니다. 또한, 학습에 사용할 장치를 선택할 수 있습니다. CPU만 사용하려면 '--gpu_id' 인수에 기본값인 -1을 입력하면 됩니다.

nlp-plm-ntc-config.xml 에서 기본 하이퍼파라미터를 확인할 수 있습니다.

### 1.4. 추론

아래와 같이 표준 입력을 추론 입력으로 사용할 수 있습니다. 예측 결과는 탭으로 구분된 두 개의 열(상위 k개의 클래스 및 입력 문장)로 구성됩니다. 결과는 표준 출력으로 표시됩니다.

#### 1.4.1. 뉴럴네트워크가 RNN, CNN 일때 추론을 위한 예제 명령어

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
negative	연결 부분 이 많이 티 가

 납니다 . 재질 구김 도 좀 있 습니다 .
positive	애기 태열 때문 에 구매 해서 잘 쓰 고 있 습니다 .
```

#### 1.4.2. PLM 기반 Trainer를 커스터마이징할 수 있는 코드를 학습을 위한 예제 명령어

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

#### 1.4.3. PLM 기반 Trainer를 Hugging Face Transformer Trainer를 사용 하는 코드를 학습을 위한 예제 명령어

```bash
상동
```

nlp-plm-ntc-config.xml 에서 기본 하이퍼파라미터를 확인할 수 있습니다.

### 1.5. 평가

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

### 1.6. Original 저자

|이름|김기현|
|-|-|
|이메일|pointzz.ki@gmail.com|
|깃허브|https://github.com/kh-kim/|
|링크드인|https://www.linkedin.com/in/ki-hyun-kim/|

### 1.7. 참고 문헌

- Kim, Convolutional neural networks for sentence classification, EMNLP, 2014
- Devlin et al., BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, ACL, 2019
- [Lee, KcBERT: Korean comments BERT, GitHub, 2020](https://github.com/Beomi/KcBERT)

---

## 2. Chat Summarization

### 2.1. 패키지 설치

```bash
pip install torch torchvision torchaudio
pip install torch_optimizer
pip install pytorch-ignite
pip install transformers
pip install scikit-learn
pip install wandb
pip install absl-py
pip install datesets
pip install nltk
pip install rouge-score
pip install evaluate
```

### 2.2. 설정 파일(config.yaml)

`config.yaml` 파일은 모델 학습 및 추론을 위한 설정 파일로 보입니다. 설정 파일의 각 섹션이 의미하는 바를 간단히 설명하겠습니다.

#### 2.2.1. `general` 섹션
- **data_path**: 데이터가 저장된 경로를 지정합니다.
- **model_name**: 사용할 사전 학습된 모델의 이름입니다. 여기서는 `digit82/kobart-summarization` 모델을 사용합니다.
- **output_dir**: 결과 파일이 저장될 디렉토리를 지정합니다.

#### 2.2.2. `inference` 섹션
- **batch_size**: 추론 시 사용할 배치 크기입니다.
- **ckt_path**: 저장된 체크포인트 파일 경로입니다.
- **early_stopping**: 조기 종료를 사용할지 여부를 설정합니다.
- **generate_max_length**: 생성되는 텍스트의 최대 길이를 지정합니다.
- **no_repeat_ngram_size**: 생성 텍스트에서 반복되지 않을 n-gram의 크기입니다.
- **num_beams**: 빔 서치에서 사용할 빔의 개수를 설정합니다.
- **remove_tokens**: 추론 시 제거할 토큰 리스트입니다.
- **result_path**: 예측 결과가 저장될 경로입니다.

#### 2.2.3. `tokenizer` 섹션
- **bos_token**: 문장의 시작 토큰입니다.
- **decoder_max_len**: 디코더에서 사용되는 최대 입력 길이입니다.
- **encoder_max_len**: 인코더에서 사용되는 최대 입력 길이입니다.
- **eos_token**: 문장의 끝을 나타내는 토큰입니다.
- **special_tokens**: 추가적인 특별 토큰 리스트입니다.

#### 2.2.4. `training` 섹션
- **do_eval**: 평가를 수행할지 여부입니다.
- **do_train**: 학습을 수행할지 여부입니다.
- **early_stopping_patience**: 조기 종료를 위한 인내 횟수입니다.
- **early_stopping_threshold**: 조기 종료를 위한 손실 변화 임계값입니다.
- **evaluation_strategy**: 평가 전략을 설정합니다 (여기서는 `epoch` 단위).
- **fp16**: FP16 혼합 정밀도 학습을 사용할지 여부입니다.
- **generation_max_length**: 생성 텍스트의 최대 길이입니다.
- **gradient_accumulation_steps**: 기울기 누적 단계 수입니다.
- **learning_rate**: 학습률입니다.
- **load_best_model_at_end**: 학습이 끝날 때 가장 좋은 모델을 로드할지 여부입니다.
- **logging_dir**: 로그가 저장될 디렉토리입니다.
- **logging_strategy**: 로그 기록 전략을 설정합니다 (여기서는 `epoch` 단위).
- **lr_scheduler_type**: 학습률 스케줄러 타입입니다.
- **num_train_epochs**: 학습할 에폭 수입니다.
- **optim**: 최적화 알고리즘을 지정합니다.
- **overwrite_output_dir**: 출력 디렉토리를 덮어쓸지 여부입니다.
- **per_device_eval_batch_size**: 평가 시 장치당 배치 크기입니다.
- **per_device_train_batch_size**: 학습 시 장치당 배치 크기입니다.
- **predict_with_generate**: 예측 시 텍스트 생성을 할지 여부입니다.
- **report_to**: 로그를 기록할 대상입니다 (여기서는 `wandb`).
- **save_strategy**: 체크포인트 저장 전략을 설정합니다 (여기서는 `epoch` 단위).
- **save_total_limit**: 저장할 체크포인트의 최대 개수입니다.
- **seed**: 무작위성 제어를 위한 시드 값입니다.
- **warmup_ratio**: 학습률 워밍업 비율입니다.
- **weight_decay**: 가중치 감소 값을 설정합니다.

#### 2.2.5. .

2.5. `wandb` 섹션
- **entity**: WandB 프로젝트 엔티티 이름입니다.
- **name**: 이번 실험의 이름입니다.
- **project**: 실험이 속한 프로젝트 이름입니다.

#### 2.2.6. 설정 파일을 통해 할 수 있는 일:
이제 이 설정 파일을 사용하여 모델 학습 또는 추론을 실행하거나, 설정을 수정하여 실험을 진행할 수 있습니다. 예를 들어, 배치 크기, 학습률, 에폭 수 등과 같은 하이퍼파라미터를 조정하거나, 로깅 및 체크포인트 저장 옵션을 변경할 수 있습니다.

### 2.3. dataset.py

`chat_summarization’ 디렉토리 하위에 ‘dataset.py` 파일은 모델 학습 및 평가를 위한 데이터 준비와 관련된 여러 가지 작업을 수행하는 스크립트입니다. 주요 클래스와 함수들의 역할을 요약해 드리겠습니다.

#### 2.3.1. 주요 클래스 및 함수 설명

1. **Preprocess 클래스**
   - `__init__`: 시작 및 끝 토큰을 초기화합니다.
   - `make_set_as_df`: 주어진 CSV 파일에서 데이터셋을 로드하고, 훈련용 또는 테스트용 데이터프레임을 반환합니다. 훈련용 데이터에는 `fname`, `dialogue`, `summary` 열이 포함되며, 테스트용 데이터에는 `fname`, `dialogue` 열만 포함됩니다.
   - `make_input`: 훈련 또는 테스트용 데이터셋을 입력으로 받아, 인코더 및 디코더에 입력할 데이터를 준비합니다. 훈련용 데이터는 인코더 입력, 디코더 입력 및 디코더 출력으로 나누어지며, 테스트용 데이터는 인코더 입력과 디코더 시작 토큰을 반환합니다.

2. **DatasetForTrain 클래스**
   - 이 클래스는 PyTorch의 `Dataset`을 상속하여, 학습용 데이터를 저장하고, 학습 중 모델이 사용할 수 있도록 데이터를 제공합니다.
   - `__getitem__`: 주어진 인덱스에 해당하는 데이터를 반환하며, 인코더와 디코더의 입력, 디코더의 출력 라벨을 포함합니다.
   - `__len__`: 데이터셋의 길이를 반환합니다.

3. **DatasetForVal 클래스**
   - `DatasetForTrain` 클래스와 거의 동일하지만, 검증 데이터셋을 위해 사용됩니다.

4. **DatasetForInference 클래스**
   - 테스트 데이터를 저장하고, 추론 시 사용할 데이터를 제공합니다.
   - `__getitem__`: 주어진 인덱스에 해당하는 테스트 데이터를 반환하며, 테스트 ID와 인코더 입력을 포함합니다.
   - `__len__`: 데이터셋의 길이를 반환합니다.

5. **prepare_train_dataset 함수**
   - 훈련 데이터와 검증 데이터를 로드하고 전처리합니다.
   - 데이터를 토크나이저를 사용해 토큰화한 후, `DatasetForTrain` 및 `DatasetForVal` 클래스를 사용해 학습 및 검증 데이터셋을 생성합니다.

6. **prepare_test_dataset 함수**
   - 테스트 데이터를 로드하고 전처리합니다.
   - 테스트 데이터를 토큰화한 후, `DatasetForInference` 클래스를 사용해 테스트 데이터셋을 생성합니다.

7. **create_dataloaders 함수**
   - 학습 및 검증 데이터셋으로부터 DataLoader를 생성하여, 배치 처리를 가능하게 합니다.

8. **compute_metrics 함수**
   - 모델 예측값과 실제 라벨을 비교하여 성능을 평가합니다.
   - Rouge 점수를 계산하여 요약 성능을 측정하며, 필요에 따라 특정 토큰을 제거한 후 점수를 계산합니다.

### 2.4. 소스 및 디렉토리 구조

#### 2.4.1. Lightening 기반 학습/예측 관련 소스 

```plaintext
project_root/
│
├── config.yaml
├── training-plm-summarization-lightening.py
├── inference-plm-summarization-lightening.py
└── chat_summarization/
    └── dataset.py
```

#### 2.4.2. Ingite 기반 학습/예측 관련 소스 

```plaintext
project_root/
│
├── config-plm-ignite.yaml
├── training-plm-summarization-ignite.py
├── inference-plm-summarization-ignite.py
└── chat_summarization/
    └── dataset.py
```

#### 2.4.3. Seq2SeqTrainer를 사용하지 않는 Ingite 기반 학습/예측 관련 소스 

```plaintext
project_root/
│
├── config-plm-ignite.yaml
├── training-plm-summarization-ignite-withoutSeq2SeqTrainer.py
├── inference-plm-summarization-ignite-withoutSeq2SeqTrainer.py
└── chat_summarization/
    └── dataset.py
```

### 2.5. 학습 테스트 관련 커맨드 라인 명령어

#### 2.5.1. Lightening 기반 학습

```bash
python training-plm-summarization-lightening.py --config config.yaml
```

#### 2.5.2. Lightening 기반 예측

```bash
python inference-plm-summarization-lightening.py --config config.yaml
```

#### 2.5.3. Ignite 기반 학습

```bash
python training-plm-summarization-ignite.py --config config-plm-ignite.yaml
```

#### 2.5.4. Ignite 기반 예측

```bash
python inference-plm-summarization-ignite.py --config config-plm-ignite.yaml
```

#### 2.5.5. Seq2SeqTrainer를 사용하지 않는 Ignite 기반 학습

```bash
python training-plm-summarization-ignite-withoutSeq2SeqTrainer.py --config config-plm-ignite.yaml
```

#### 2.5.6. Seq2SeqTrainer를 사용하지 않는 Ignite 기반 예측

```bash
python inference-plm-summarization-ignite-withoutSeq2SeqTrainer.py --config config-plm-ignite.yaml
```

### 2.6. Lightening 대신 Ignite 적용 관련

#### 2.6.1. 학습 소스 변경

PyTorch-Ignite는 PyTorch 프로젝트의 학습과 평가를 더 쉽게 관리할 수 있도록 도와주는 고수준 라이브러리입니다. 아래는 `training-plm-summarization-lightening.py`를 Ignite 기반으로의 적용과 관련된 내용입니다.

Ignite를 사용하여 트레이닝 루프를 작성하고, 특히 `ignite.engine.Engine`과 `ignite.engine.Events`, `ignite.metrics`를 사용하여 학습 및 평가 절차를 간소화합니다.

##### 2.6.1.1. 주요 변경 사항 설명

1. **Engine 및 이벤트 사용**: 
   - Ignite의 `Engine`을 사용하여 학습 및 평가 루프를 정의했습니다.
   - `Events`를 통해 학습 및 평가 단계에서 발생하는 이벤트(예: 에포크 완료, 평가 완료 등)에 콜백을 연결했습니다.

2. **WandBLogger**: 
   - Ignite의 `WandBLogger`를 사용하여 학습 중간 결과를 로그로 남기도록 설정했습니다.

3. **EarlyStopping 및 ModelCheckpoint**: 
   - Ignite의 `EarlyStopping`과 `ModelCheckpoint` 핸들러를 사용하여 학습을 관리했습니다.

4. **ProgressBar**: 
   - 학습 진행 상황을 표시하기 위해 `ProgressBar`를 사용했습니다.

5. **로깅**:
   - `ignite.utils.setup_logger`를 사용하여 로그를 출력하도록 설정했습니다.

이제 이 코드로 PyTorch-Ignite를 사용해 모델을 학습할 수 있습니다. Ignite는 PyTorch와 자연스럽게 통합되며, 학습 루프를 유연하게 구성할 수 있도록 돕습니다.

#### 2.6.2. 예측 소스 변경

아래는 PyTorch-Ignite를 사용하여 `inference-plm-summarization-lightening.py`를 변환과 관련된 내용입니다. Ignite는 학습뿐만 아니라 추론 과정도 관리할 수 있습니다.

##### 2.6.2.1. 주요 변경 사항 설명

1. **Engine 및 이벤트 사용**: 
   - Ignite의 `Engine`을 사용하여 추론 단계를 정의했습니다. `Engine`은 입력 데이터를 받아서 모델을 통해 예측을 수행하는 단계를 관리합니다.

2. **ProgressBar**:
   - `ProgressBar`를 사용하여 추론 진행 상황을 표시합니다.

3. **결과 처리**:
   - 각 `ITERATION_COMPLETED` 이벤트에서 결과를 처리하고 `all_outputs` 속성에 저장하여 모든 결과를 추적합니다.
   - 마지막에 모든 결과를 CSV 파일로 저장합니다.

4. **결과 저장**:
   - 추론 결과는 지정된 경로에 CSV 파일로 저장됩니다.

이 코드를 사용하여 Ignite를 기반으로 모델 추론을 수행할 수 있습니다. Ignite는 간단한 API로 모델 학습 및 추론 절차를 쉽게 관리할 수 있게 해줍니다.

### 2.7. Ignite로 변환된 소스에 WandB 추가 설정 

`PyTorch-Ignite`를 사용하여 `WandB`에서 다양한 학습 메트릭을 모니터링할 수 있도록 코드를 수정했습니다. 아래는 당신이 제안한 항목들을 모니터링할 수 있도록 코드에 추가한 내용입니다.

#### 2.7.1. 추가된 기능 설명:

1. **훈련 및 검증 손실**:
   - `RunningAverage` 메트릭을 통해 각 에폭마다 `training_loss`와 `validation_loss`를 계산하고, `wandb.log`를 사용해 기록합니다.

2. **학습 속도 및 에폭 시간**:
   - `ignite.engine.Engine`의 이벤트 시스템을 활용해, 각 에폭 완료 시 손실을 로그합니다.

3. **러닝 레이트**:
   - `Events.ITERATION_COMPLETED` 이벤트에 러닝 레이트를 추적하여 `wandb`에 기록합니다.

4. **그래디언트**:
   - 각 레이어의 그래디언트를 추적하여, 해당 값을 `wandb.log`를 통해 기록합니다. 이 작업은 `Events.ITERATION_COMPLE

TED`에서 수행됩니다.

5. **메트릭**:
   - 사용자 정의 메트릭(예: `Rouge` 스코어 등)이 있다면, 추가적으로 계산하여 `wandb`에 로그할 수 있습니다.

#### 2.7.2. 실행 시 `wandb` 대시보드에서 다음과 같은 정보를 확인할 수 있습니다:
- 각 에폭마다의 훈련 및 검증 손실
- 학습 과정에서 러닝 레이트의 변화
- 그래디언트의 변화
- 기타 원하는 메트릭

이렇게 설정된 코드는 `PyTorch Lightning` 수준의 모니터링 기능을 `WandB`에 통합하여, 학습 과정의 세부적인 정보를 추적하고 분석할 수 있습니다. `Rouge` 메트릭은 텍스트 요약과 같은 작업에서 자주 사용됩니다. 이를 모니터링하기 위해서는 `ignite`의 `Engine`을 활용하여 평가 과정에서 `Rouge` 점수를 계산하고, `wandb`에 로그하는 부분을 추가해야 합니다.

##### 2.7.2.1. 추가된 `ROUGE` 메트릭 모니터링 기능 설명:

1. **ROUGE 메트릭 로드**:
   - `datasets` 라이브러리에서 `load_metric("rouge")`를 사용해 `ROUGE` 메트릭을 로드합니다.

2. **평가 단계에서 ROUGE 계산**:
   - `ignite_evaluator`의 `COMPLETED` 이벤트에서 `ROUGE` 점수를 계산합니다.
   - 모델의 예측값과 레이블(참조 텍스트)을 `trainer.tokenizer.decode()`를 사용해 디코딩한 후, `ROUGE` 점수를 계산합니다.

3. **WandB에 ROUGE 점수 로그**:
   - 계산된 `ROUGE-1`, `ROUGE-2`, `ROUGE-L`의 F-1 점수를 `wandb.log`를 사용하여 기록합니다.

##### 2.7.2.2. 이 코드의 결과:
- **WandB 대시보드**에서 `ROUGE` 점수와 더불어 훈련 및 검증 손실, 러닝 레이트, 그래디언트 등 다양한 메트릭을 실시간으로 모니터링할 수 있습니다. 이로써, `PyTorch Lightning` 수준의 모니터링을 `Ignite`와 `WandB`를 사용해 구현할 수 있습니다.

### 2.8. Ignite로 변환된 소스에 체크포인트 관련 추가 코딩 

현재 제공된 코드는 `Seq2SeqTrainer`를 사용하여 모델을 학습하고 있으며, `Seq2SeqTrainer`는 자동으로 체크포인트를 저장하는 기능을 내장하고 있습니다. 하지만 `Ignite`를 사용하여 커스터마이징된 체크포인트 저장 기능은 포함되어 있지 않습니다. `Seq2SeqTrainer`가 관리하는 기본 체크포인트 저장 기능에 대해 설명한 후, `Ignite`에서 추가적으로 체크포인트를 관리하는 방법을 소개하겠습니다.

#### 2.8.1. Seq2SeqTrainer의 체크포인트 관리

`Seq2SeqTrainer`는 `training_args`를 통해 체크포인트를 자동으로 저장합니다. 아래 설정들이 중요한 역할을 합니다:

- **save_strategy**: 
  - `"epoch"` 또는 `"steps"`로 설정할 수 있으며, 체크포인트가 저장될 주기를 결정합니다.
  - `"epoch"`로 설정하면 각 에폭이 끝날 때마다 체크포인트가 저장됩니다.
  - `"steps"`로 설정하면 지정된 스텝마다 체크포인트가 저장됩니다.

- **save_steps**: 
  - `save_strategy`가 `"steps"`일 때, 몇 스텝마다 체크포인트를 저장할지 결정합니다.

- **save_total_limit**:
  - 저장할 체크포인트의 최대 개수를 설정합니다. 이 수를 초과하면 가장 오래된 체크포인트가 삭제됩니다.

- **load_best_model_at_end**:
  - `True`로 설정하면 학습이 끝날 때 최고의 성능을 보인 모델을 자동으로 로드합니다.

예를 들어, 현재 코드에서는 `Seq2SeqTrainingArguments`가 아래와 같이 설정되어 있습니다:

```python
training_args = Seq2SeqTrainingArguments(
    output_dir=config['general']['output_dir'],
    save_strategy=config['training']['save_strategy'],  # 'epoch' 또는 'steps'
    save_total_limit=config['training']['save_total_limit'],  # 저장할 체크포인트의 최대 개수
    load_best_model_at_end=config['training']['load_best_model_at_end'],  # 학습 종료 시 최고의 모델 로드
    # 추가 설정들...
)
```

이 설정에 따라 `Seq2SeqTrainer`가 자동으로 체크포인트를 저장합니다.

`config-plm-ignite.yaml` 파일을 확인합니다. 이제 수행해야 할 작업에 대해 알려드리겠습니다. 만약 `inference-plm-summarization-ignite.py`나 `training-plm-summarization-ignite.py`에서 문제가 발생했거나 추가 작업이 필요하다면, 다음 단계를 수행할 수 있습니다.

### 2.9. Stage server에서 실험 진행 순서

#### 2.9.1. 체크포인트 경로 확인

먼저 `config-plm-ignite.yaml` 파일에서 지정된 체크포인트 경로가 올바른지 확인하세요. 경로가 잘못되었거나 파일이 누락되었을 경우, 모델 로딩 단계에서 오류가 발생할 수 있습니다.

- `inference.ckt_path` 항목이 `./checkpoints/summarization.hft.kobart.run.02`로 설정되어 있습니다. 이 경로에 실제 체크포인트 파일이 존재하는지 확인하세요. 일반적으로 체크포인트 파일은 `pytorch_model.bin`이나 `.h5` 등의 파일 형식을 가집니다.

#### 2.9.2. 모델 로딩 오류 해결

오류(`HFValidationError`)는 올바른 경로와 파일을 가리키지 않을 때 발생합니다. 

- 체크포인트 경로를 실제 체크포인트 디렉토리 경로로 업데이트하십시오. 예를 들어, 체크포인트 디렉토리 이름이 `checkpoint-epoch-1`이라면:
  ```yaml
  ckt_path: ./checkpoints/summarization.hft.kobart.run.02/checkpoint-epoch-1
  ```

#### 2.9.3. Lightening 기반 학습 및 추론 진행

이제 모델 학습과 추론을 시도할 수 있습니다. 만약 Ignite 기반의 스크립트를 사용하는 경우 다음 단계를 따라 수행합니다.

1. **학습 스크립트 실행**:
   ```bash
   python training-plm-summarization-lightening.py --config config-.yaml
   ```
   이 명령어는 `config.yaml` 설정에 따라 모델을 학습시킵니다.

2. **추론 스크립트 실행**:
   학습이 완료된 후, 모델이 잘 작동하는지 확인하기 위해 추론 스크립트를 실행합니다.
   ```bash
   python inference-plm-summarization-lightening.py --config config.yaml
   ```

#### 2.9.4. Ignite 기반 학습 및 추론 진행

이제 모델 학습과 추론을 시도할 수 있습니다. 만약 Ignite 기반의 스크립트를 사용하는 경우 다음 단계를 따라 수행합니다.

1. **학습 스크립트 실행**:
   ```bash
   python training-plm-summarization-ignite.py --config config-plm-ignite.yaml
   ```
   이 명령어는 `config-plm-ignite.yaml` 설정에 따라 모델을 학습시킵니다.

2. **추론 스크립트 실행**:
   학습이 완료된 후, 모델이 잘 작동하는지 확인하기 위해 추론 스크립트를 실행합니다.
   ```bash
   python inference-plm-summarization-ignite.py --config config-plm-ignite.yaml
   ```

#### 2.9.5. 결과 확인 및 로깅

학습과 추론 결과는 각각 `./logs`와 `./prediction/` 디렉토리에 저장됩니다. 또한, `wandb`를 사용하고 있으므로 학습 과정과 결과가 `wandb` 대시보드에도 기록됩니다.

1. **WandB 설정 확인**:
   `wandb` 계정과 프로젝트 이름이 `config.yaml` 나 `config-plm-ignite.yaml`의 `wandb` 섹션에 올바르게 설정되었는지 확인하세요. 잘못된 설정으로 인해 로그가 기록되지 않을 수 있습니다.
   
2. **결과 파일 확인**:
   `./prediction/` 디렉토리에서 생성된 요약 결과 파일을 확인하여 모델이 제대로 작동하는지 검토하세요.

#### 2.9.6. 성능 평가 및 튜닝

모델 성능이 기대에 미치지 못한다면 `config.yaml` 나 `config-plm-ignite.yaml`의 하이퍼파라미터를 조정하거나, 추가 데이터로 모델을 재학습시키는 등의 튜닝 작업을 수행할 수 있습니다.

- **학습률 조정**: `learning_rate` 값을 조정해보세요.
- **에폭 수 조정**: `num_train_epochs`를 늘리거나 줄여서 모델의 과적합 또는 과소적합을

 방지하세요.
- **배치 크기 조정**: `per_device_train_batch_size`와 `per_device_eval_batch_size`를 조정하여 학습 속도와 메모리 사용량의 균형을 맞추세요.

#### 2.9.7. 요약

1. `config.yaml` 나 `config-plm-ignite.yaml` 파일의 경로가 올바르게 설정되었는지 확인하세요.
2. 학습 및 추론 파이썬스크립트(.py)를 실행하여 모델을 학습시키고 테스트하세요.
3. 결과를 `wandb`와 파일에서 확인하고, 필요한 경우 하이퍼파라미터를 조정하세요.

이제 문제를 해결하고 원하는 작업을 수행할 준비가 되셨습니다.

### 2.10. Early Stopping 관련

**Early Stopping**이 설정된 경우 전체 `epoch` 수가 20으로 지정되어 있어도, 모델의 학습이 중간에 끝날 수 있습니다. 이는 Early Stopping의 목적과 동작 원리에 따른 것입니다.

#### 2.10.1. Early Stopping의 동작 원리

**Early Stopping**은 모델의 학습 중에 검증 데이터셋에서 성능이 더 이상 개선되지 않을 때 학습을 조기 종료하는 기법입니다. 이를 통해 과적합(overfitting)을 방지하고, 불필요한 학습 시간을 줄일 수 있습니다.

##### 2.10.1.1. 주요 매개변수:
- **patience**: 성능이 개선되지 않는 연속된 `epoch` 수. 예를 들어 `patience=3`이면, 모델 성능이 3번의 연속된 `epoch` 동안 개선되지 않으면 학습을 중단합니다.
- **monitor**: 개선 여부를 판단하는 기준 지표(예: `validation loss`).
- **min_delta**: 성능 개선을 판단할 때 사용하는 최소 변화 값. 이 값보다 작으면 개선되지 않은 것으로 간주합니다.

#### 2.10.2. Early Stopping이 동작하는 예시

1. **설정**: 
   - `epoch=20`
   - `patience=3`
   - 모니터링 지표: `validation loss`

2. **학습 과정**:
   - 1~10번째 `epoch`에서 모델의 성능이 점점 개선됨 (`validation loss`가 감소).
   - 11번째 `epoch`부터 `validation loss`가 개선되지 않음.
   - 12, 13, 14번째 `epoch`에서도 `validation loss`가 개선되지 않음.

3. **Early Stopping 작동**:
   - 14번째 `epoch`까지 개선이 없었으므로, `patience=3`에 따라 학습이 조기 종료됨.

#### 2.10.3. Early Stopping 결론

따라서, Early Stopping을 사용하면 전체 `epoch` 수가 지정되어 있더라도, 지정된 `patience`에 따라 학습이 중간에 종료될 수 있습니다. 이는 모델이 더 이상 학습할 필요가 없다고 판단될 때 학습을 중단하여, 자원의 낭비를 줄이고 과적합을 방지하는 데 유용합니다.

### 2.11. Ignite를 활용한 커스마이징 범위 정리

이 코드에서 `ignite`를 사용하여 학습 과정과 평가 과정을 커스터마이징한 부분을 아래와 같이 설명할 수 있습니다. 특히 `ignite`의 `Engine`과 이벤트 시스템을 활용하여 학습과 평가의 각 단계를 제어하고 있습니다.

#### 2.11.1. `ignite_trainer`와 `ignite_evaluator` 커스터마이징:

```python
# Ignite engine for training and evaluation
def update_engine(engine, batch):
    return trainer.prediction_step(trainer.model, batch, prediction_loss_only=False)

def evaluation_step(engine, batch):
    return trainer.prediction_step(trainer.model, batch, prediction_loss_only=True)

ignite_trainer = Engine(update_engine)
ignite_evaluator = Engine(evaluation_step)
```

- **`update_engine`**: `ignite_trainer`의 `Engine`에서 사용되며, 매 `iteration`마다 `trainer.prediction_step`을 호출하여 학습 배치를 처리합니다.
- **`evaluation_step`**: `ignite_evaluator`의 `Engine`에서 사용되며, 매 `iteration`마다 `trainer.prediction_step`을 호출하여 평가 배치를 처리합니다.

#### 2.11.2. 이벤트 핸들러를 통한 커스터마이징

`ignite`의 이벤트 시스템을 활용하여 학습 과정 중 특정 이벤트에서 커스터마이징된 로직을 실행합니다.

##### 2.11.2.1. 에포크 완료 시 로직 (`EPOCH_COMPLETED` 이벤트)

```python
@ignite_trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    global best_val_loss  # Ensure that best_val_loss is recognized as a global variable
    epoch = engine.state.epoch
    wandb.log({'epoch': epoch, 'training_loss': engine.state.metrics['training_loss']})

    # Evaluate the model and save checkpoint if the performance improves
    ignite_evaluator.run(val_loader)
    val_loss = ignite_evaluator.state.metrics['validation_loss']
    wandb.log({'epoch': engine.state.epoch, 'validation_loss': val_loss})

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trainer.save_model(output_dir=os.path.join(config['general']['output_dir'], f"checkpoint-epoch-{epoch}"))
```

- **`EPOCH_COMPLETED` 이벤트 핸들러**:
  - 에포크가 끝날 때마다 학습 손실과 검증 손실을 로깅합니다.
  - 검증 손실이 개선된 경우에만 체크포인트를 저장합니다.

##### 2.11.2.2. `iteration` 완료 시 로직 (`ITERATION_COMPLETED` 이벤트)

```python
# Monitor learning rate
@ignite_trainer.on(Events.ITERATION_COMPLETED)
def log_learning_rate(engine):
    if trainer.optimizer is not None:  # Check if the optimizer is initialized
        lr = trainer.optimizer.param_groups[0]['lr']
        wandb.log({'learning_rate': lr, 'iteration': engine.state.iteration})

# Monitor gradients
@ignite_trainer.on(Events.ITERATION_COMPLETED)
def log_gradients(engine):
    for name, param in generate_model.named_parameters():
        if param.requires_grad and param.grad is not None:
            wandb.log({f'gradients/{name}': torch.norm(param.grad).item(), 'iteration': engine.state.iteration})
```

- **`ITERATION_COMPLETED` 이벤트 핸들러**:
  - 각 `iteration`이 완료될 때마다 현재 러닝 레이트와 각 레이어의 그래디언트 크기를 로깅합니다.

##### 2.11.2.3. 평가 완료 시 로직 (`COMPLETED` 이벤트)

```python
@ignite_evaluator.on(Events.COMPLETED)
def compute_rouge(engine):
    # Ensure engine.state.output is not None
    if engine.state.output is not None:
        predictions = engine.state.output[1]
        references = engine.state.output[2]
        
        if predictions is not None and references is not None:
            decoded_preds = [trainer.tokenizer.decode(g, skip_special_tokens=True) for g in predictions]
            decoded_refs = [trainer.tokenizer.decode(l, skip_special_tokens=True) for l in references]
            
            rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_refs)
            wandb.log({'rouge1': rouge_scores['rouge1'].mid.fmeasure,
                       'rouge2': rouge_scores['rouge2'].mid.fmeasure,
                       'rougeL': rouge_scores['rougeL'].mid.fmeasure,
                       'epoch': engine.state.epoch})
```

- **`COMPLETED` 이벤트 핸들러**:
  - 평가가 완료되면, `ROUGE` 점수를 계산하여 로깅합니다.

#### 2.11.3. Ignite를 활용한 커스마이징 요약:

- **학습 및 평가 루프**는 `ignite`의 `Engine`을 사용하여 정의되었고, `ignite_trainer`와 `ignite_evaluator`로 나뉘어 각각 학습과 평가를 수행합니다.
- **이벤트 핸들러**를 사용하여 각 에포크 및 `iteration`의 끝에서 로깅과 체크포인트 저장 등의 작업을 커스터마이징하고 있습니다.

이러한 방식으로 `ignite`를 사용하면 학습 과정의 여러 측면을 세밀하게 제어할 수 있으며, `transformers`의 `Seq2SeqTrainer`와 결합하여 더욱 유연한 학습 관리가 가능합니다.

### 2.12. Hugging Faces Transformers 가 제공하는 Seq2SeqTrainer 커스마징 범위

`Seq2SeqTrainer` 대신 `ignite`를 사용하여 학습 과정을 직접 커스터마이징하는 것은 가능합니다. 이렇게 하면 학습 과정의 모든 세부 사항을 제어할 수 있습니다. 아래는 `Seq2SeqTrainer`를 사용하지 않고 `ignite`를 통해 학습 과정을 커스터마이징하는 방법을 보여주는 코드 예제입니다.

#### 2.12.1. 필요한 모듈 불러오기

먼저 필요한 모듈들을 불러옵니다.

```python
import os


import yaml
import torch
import wandb
import argparse
from transformers import AutoTokenizer, BartForConditionalGeneration, BartConfig, AdamW, get_scheduler
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from torch.utils.data import DataLoader
from ignite.metrics import RunningAverage
from chat_summarization.dataset import Preprocess, prepare_train_dataset, compute_metrics
from datasets import load_metric
```

#### 2.12.2. 모델 및 데이터 로드

모델과 데이터를 불러오는 함수들을 정의합니다.

```python
def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config)

    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    return model, tokenizer

def prepare_optimizer_and_scheduler(model, config, train_loader):
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    lr_scheduler = get_scheduler(
        name=config['training']['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=config['training']['warmup_ratio'] * len(train_loader),
        num_training_steps=config['training']['num_train_epochs'] * len(train_loader),
    )
    return optimizer, lr_scheduler
```

#### 2.12.3. Ignite 기반 학습 및 평가 루프 구현

`ignite`를 사용하여 학습 및 평가 루프를 정의합니다.

```python
def train_step(engine, batch):
    model.train()
    inputs = batch['input_ids'].to(engine.state.device)
    labels = batch['labels'].to(engine.state.device)
    
    outputs = model(input_ids=inputs, labels=labels)
    loss = outputs.loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr_scheduler.step()

    return loss.item()

def eval_step(engine, batch):
    model.eval()
    with torch.no_grad():
        inputs = batch['input_ids'].to(engine.state.device)
        labels = batch['labels'].to(engine.state.device)
        
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss

    return loss.item()

def inference_step(engine, batch):
    model.eval()
    with torch.no_grad():
        inputs = batch['input_ids'].to(engine.state.device)
        generated_ids = model.generate(input_ids=inputs)
    
    return generated_ids, batch['labels']
```

#### 2.12.4. 학습 과정 설정

`ignite`의 `Engine`을 사용하여 학습 엔진과 평가 엔진을 설정하고, 필요한 핸들러를 추가합니다.

```python
def main(config):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load tokenizer and model
    model, tokenizer = load_tokenizer_and_model_for_train(config, device)

    # Prepare dataset
    preprocessor = Preprocess(config['tokenizer']['bos_token'], config['tokenizer']['eos_token'])
    data_path = config['general']['data_path']
    train_dataset, val_dataset = prepare_train_dataset(config, preprocessor, data_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=config['training']['per_device_train_batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['per_device_eval_batch_size'])

    # Prepare optimizer and scheduler
    global optimizer, lr_scheduler
    optimizer, lr_scheduler = prepare_optimizer_and_scheduler(model, config, train_loader)

    # Define training and evaluation engines
    trainer = Engine(train_step)
    evaluator = Engine(eval_step)
    inferencer = Engine(inference_step)

    # Attach running average metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

    # Attach progress bar
    pbar = ProgressBar()
    pbar.attach(trainer)
    pbar.attach(evaluator)

    # Log training and validation losses at the end of each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(val_loader)
        val_loss = evaluator.state.metrics['loss']
        wandb.log({'epoch': engine.state.epoch, 'training_loss': engine.state.metrics['loss'], 'validation_loss': val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config['general']['output_dir'], f"best_model.pth"))

    # Save model at regular intervals
    @trainer.on(Events.ITERATION_COMPLETED(every=250))
    def save_model_checkpoint(engine):
        torch.save(model.state_dict(), os.path.join(config['general']['output_dir'], f"checkpoint-{engine.state.iteration}.pth"))

    # Run the trainer
    trainer.run(train_loader, max_epochs=config['training']['num_train_epochs'])

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training script for PLM summarization.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    main(config)
```

#### 2.12.5. 핵심 커스터마이징 사항

- **학습 루프**: `ignite`의 `Engine`을 통해 직접 정의되었습니다. `train_step`, `eval_step`, `inference_step`을 각각 학습, 평가, 예측 단계로 사용합니다.
- **최적화**: `AdamW` 옵티마이저와 학습률 스케줄러를 수동으로 설정했습니다.
- **모델 저장**: 학습 중 일정 간격(250 `iteration`)마다 모델 체크포인트를 저장하고, 검증 손실이 개선될 때마다 최상의 모델을 저장합니다.
- **로깅**: `wandb`를 사용하여 에포크별 손실 및 성능 지표를 로깅합니다.

이러한 접근 방식은 `Seq2SeqTrainer`의 추상화를 사용하지 않고, 전체 학습 과정을 직접 제어할 수 있게 해줍니다. 이 방식은 학습 과정의 모든 측면을 커스터마이징할 수 있는 유연성을 제공합니다.

### 2.13. Hugging Faces Transformers 가 제공하는 Seq2SeqTrainer 없이 커스터마이지이 하기

`Seq2SeqTrainer`를 사용하지 않는 학습 소스에서는, 모델의 훈련과 평가 과정을 직접 정의하고 제어해야 합니다. `Seq2SeqTrainer`는 Hugging Face의 트랜스포머 라이브러리에서 제공하는 고수준 API로, 텍스트 요약과 같은 시퀀스-투-시퀀스(seq2seq) 작업을 쉽게 수행할 수 있도록 해줍니다. 그러나 이 클래스를 사용하지 않고 학습을 구현할 때는, 모델의 훈련과 평가를 수동으로 관리해야 합니다. 

아래는 `Seq2SeqTrainer`를 사용하지 않는 학습 소스의 주요 구성 요소와 그 의미를 설명한 내용입니다.

#### 2.13.1. 모델과 토크나이저 로드

```python
def load_tokenizer_and_model_for_train(config, device):
    model_name = config['general']['model_name']
    bart_config = BartConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generate_model = BartForConditionalGeneration.from_pretrained(config['general']['model_name'], config=bart_config)

    special_tokens_dict = {'additional_special_tokens': config['tokenizer']['special_tokens']}
    tokenizer.add_special_tokens(special_tokens_dict)
    generate_model.resize_token_embeddings(len(tokenizer))
    generate_model.to(device)

    return generate_model, tokenizer
```
- **설명**: 이 함수는 주어진 구성 파일(`config`)에서 모델 이름과 디바이스 정보를 읽어와, 토크나이저와 BART 모델을 로드하고, 특수 토큰을 추가한 뒤, 모델을 GPU로 이동시킵니다.

#### 2.13.2. 훈련 및 평가 엔진 생성

```python
def create_trainer_and_evaluator(config, generate_model, tokenizer, optimizer, device):
    def update_engine(engine, batch):
        generate_model.train()
        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = generate_model(**inputs)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss.item()

    def evaluation_step(engine, batch):
        generate_model.eval()
        with torch.no_grad():
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = generate_model(**inputs)
            loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            references = inputs['labels']
        return loss.item(), predictions, references

    trainer = Engine(update_engine)
    evaluator = Engine(evaluation_step)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'training_loss')
    RunningAverage(output_transform=lambda x: x[0]).attach(evaluator, 'validation_loss')

    return trainer, evaluator
```
- **설명**: 이 함수는 `Ignite` 라이브러리를 사용하여 훈련과 평가를 위한 엔진을 생성합니다.
  - **훈련 엔진 (trainer)**: 각 배치에 대해 모델의 순전파와 역전파를 수행하고, 옵티마이저를 통해 파라미터를 업데이트합니다.
  - **평가 엔진 (evaluator)**: 모델을 평가 모드로 전환한 후, 평가 데이터셋에서 손실과 예측값을 계산합니다.
  - **`RunningAverage`**: 손실의 이동 평균을 계산해 훈련 및 평가 과정에서 로깅합니다.

#### 2.13.3. 훈련 루프 및 체크포인트 저장

```python
@trainer.on(Events.EPOCH_COMPLETED)
def log_training_loss(engine):
    global best_val_loss
    epoch = engine.state.epoch
    wandb.log({'epoch': epoch, 'training_loss': engine.state.metrics['training_loss']})

    # Run evaluation
    evaluator.run(val_loader)
    val_loss = evaluator.state.metrics['validation_loss']
    wandb.log({'epoch': epoch, 'validation_loss': val_loss})

    # Save checkpoint if the validation loss has improved
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_dir = os.path.join(config['general']['output_dir'], f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        generate_model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, 'optimizer.pt'))
```
- **설명**: 이 코드에서는 각 에포크가 끝날 때마다 훈련 손실을 로그로 남기고, 검증을 실행한 후 검증 손실이 향상되었는지 확인합니다. 검증 손실이 향상된 경우에만 모델 체크포인트를 저장합니다.

#### 2.13.4. 평가 및 ROUGE 점수 계산

```python
@evaluator.on(Events.COMPLETED)
def compute_rouge(engine):
    predictions = engine.state.output[1]
    references = engine.state.output[2]

    if predictions is not None and references is not None:
        decoded_preds = [tokenizer.decode(g, skip_special_tokens=True) for g in predictions]
        decoded_refs = [tokenizer.decode(l, skip_special_tokens=True) for l in references]

        rouge_scores = rouge.compute(predictions=decoded_preds, references=decoded_refs)
        wandb.log({'rouge1': rouge_scores['rouge1'].mid.fmeasure,
                   'rouge2': rouge_scores['rouge2'].mid.fmeasure,
                   'rougeL': rouge_scores['rougeL'].mid.fmeasure,
                   'epoch': engine.state.epoch})
```
- **설명**: 평가가 완료될 때마다 ROUGE 점수를 계산하여 성능을 측정하고, 그 결과를 로그로 남깁니다.

#### 2.13.5. 학습 시작

```python
trainer.run(train_loader, max_epochs=config['training']['num_train_epochs'])
```
- **설명**: 학습을 시작하며, 주어진 에포크 수만큼 모델을 학습시킵니다.

#### 2.13.6. 요약

`Seq2SeqTrainer`를 사용하지 않는 학습 소스에서는 모델 학습 과정의 세부 사항을 수동으로 제어할 수 있는 유연성이 있습니다. 이를 통해 더 복잡한 커스터마이징이나 특정 요구사항에 맞는 설정을 직접 구현할 수 있습니다. 하지만, 이는 코드의 복잡도를 높이며, 훈련, 평가, 체크포인트 저장 등과 같은 작업을 수동으로 구현해야 한다는 부담이 있습니다.