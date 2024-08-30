# 2024 국립국어원 인공지능의 한국어 능력 평가

대화 맥락 추론 (가 유형) - **모두의 말뿡치** 팀
> 리더보드 2위 모델- **'정말뿡'**


<br>
본 리포지토리는 '대화 맥락 추론'에 대한 모두의 말뿡치 팀의 제출 모델의 학습과 평가를 재현하기 위한 코드를 포함하고 있습니다.

<br>
<p align="center">
<img src="./asset/guklip.png" width="300">
</p>

<br>

_학습 및 추론의 실행 방법은 아래에서 확인하실 수 있습니다._
</br>

## Table of Contents  
1. [소개](#1._소개)  
2. [데이터셋 소개](#2._데이터셋_소개) 
3. [EDA](#3._EDA)
4. [모델 개요](#4._모델_개요)
<br> a. [사용 모델 선택](##a._사용_모델_선택)
<br> b. [CoT](##b._CoT)
<br> c. [Persona](##c.Persona)
5. [사용 프롬프트](#5._사용_프롬프트)
... 추가 예정

---
## 1. 소개
‘대화 맥락 추론' 과제에서 **‘가'** 유형으로 외부 데이터를 사용하거나 데이터 증강 사용이 불가합니다.
<br>본 과제는 주어진 대화 내용을 바탕으로, 특정된 대상 발화로부터 다섯 가지의 추론문 유형인 ‘원인’, ‘후행 사건’, ‘전제 조건’, ‘내적 동기’, ‘감정 반응’ 중 하나에 상응하는 문장을 정확하게 추론하는 것을 목표로 합니다.
모델은 세 개의 추론 옵션 중 가장 적합한 답을 선택해야 하며, 이를 통해 모델의 대화 맥락 이해 능력과 적합한 추론 선택 능력을 평가합니다. </br>

<br>저희 모델은 대회의 기준 모델인 **불로섬(Bllossom)**-[`teddysum/Korean_CCI_2024`](https://github.com/MLP-Lab/Bllossom) 을 기반하여 구현되었으며, Bllossom은 한국어 데이터로 파인튜닝된 LLaMA3 기반의 한국어 대화 추론을 위해 설계된 모델입니다. 

### 모두의 말뿡치 팀 소개
> 연세대학교 빅데이터 학회 'YBIGTA'의 Data Science 팀 소속 학생들

<p align="center">
<img src="./asset/ybigta.png" width="280">
</p>

---
## 2. 데이터셋 소개
국립국어원에서 제공되는 대화 맥락 추론 말뭉치 데이터셋은 대화문, 대상 발화, 추론문의 유형, 추론문이 포함되어 있습니다. 
> 데이터 형식의 예시
<p align="center">
<img src="./asset/data.png" width="450">
</p>


---
## 3. EDA
대화 맥락 추론 말뭉치 데이터셋의 train과 test에 대한 EDA (Exploratory Data Analysis) 결과는 [`eda/train`](./eda/train) 와 [`eda/test`](./eda/test) 디렉토리에 저장되어 있습니다.

> 대화 맥락 추론 말뭉치 train 데이터의 EDA 결과 예시
> <img src="./eda/train/train_utterance_wordcloud.png" width="650">
> 탐색적 데이터 분석을 통한 인사이트를 통한 전처리 진행

---
## 4. 모델 개요

### a. 사용 모델 선택

대화 맥락 추론 과제를 수행하기 위해 사용된 모델의 종류는 아래와 같습니다.

- 
-
- 



### b. 하이퍼 파라미터
솰라솰라

### c. Parameter Efficient Fine Tuning
모델 훈련의 효율성을 높이기 위해, 파라미터 효율적으로 fine-tuning 하는 Parameter-Efficient-Fine-Tuning(PEFT)방법을 사용하여 학습에 필요한 메모리 용량과 계산량의 크기를 줄였습니다. PEFT에서도 [LoRA](https://arxiv.org/pdf/2106.09685)와 [QLoRA](https://arxiv.org/pdf/2305.14314)라는 두 가지 기법을 활용하여 자원 소모를 최소화하였습니다.

- **LoRA (Low-Rank Adaptation)**: Low-rank factorization 방법을 활용하여 LLM의 linear layer에 대한 업데이트를 근사화하는 방법입니다.
<p align="center">
<img src="./asset/lora.png" width="230">
</p>

- **QLoRA (Quantized LoRA Adapters)**: LoRA의 개념을 양자화하여 메모리와 연산 자원을 절약하면서도 효율적으로 모델을 적응시킵니다.




---
## 5. 사용 프롬프트
### a. Textgrad
blah
### b. CoT
blah

### c. Persona
blah 

### d. Categorized
blah

### e. Korean & English

---
## 6. Ensemble
blah

---
## 7. ORPO
모델의 성능을 높이기 위해, '[ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/pdf/2403.07691)' 논문에서 소개된 ORPO 방식을 사용하였습니다. 
<br>ORPO 방식은 "Odds Ratio Preference Optimization"의 약자로, reference model 없이도 선호도를 최적화하는 접근법으로, 추론문 유형별로 모델의 선택을 더욱 정교하게 조정하는 데 기여할 수 있는 방법입니다. </br>

기존 RL 방식과의 비교:

![ORPO 비교 사진](./asset/orpo.png)

Forward pass 가 기존의 DPO와 RLHF 방식에 비해 절반으로 줄임으로써 메모리와 연산 효율성을 크게 향상시킵니다.

---
## 8.레포지토리 구조 (Repository Structure)

블라


---
## 8. 실행 방법 (How to Run)

#### Miniconda 설치
먼저 새로운 conda 환경을 생성합니다.
```bash
$ mkdir -p ~/miniconda3    # 사용자 홈 디렉토리에 'miniconda3' 디렉토리를 생성 (디렉토리가 이미 존재해도 오류 없이 진행)
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh    # Miniconda 설치 스크립트를 다운로드
$ bash Miniconda3-latest-Linux-x86_64.sh    # 설치 스크립트를 실행하여 Miniconda를 설치
$ rm ~/miniconda3/miniconda.sh   # 설치 후 불필요한 설치 스크립트 파일을 삭제
```
설치가 완료된 후, 새로 설치한 Miniconda를 초기화합니다.

```bash
$ ~/miniconda3/bin/conda init bash    # bash 셸에 conda 초기화 설정을 추가
$ ~/miniconda3/bin/conda init zsh    # zsh 셸에 conda 초기화 설정을 추가
$ conda --version    # 설치된 conda의 버전을 확인하여 설치가 성공적으로 완료되었는지 확인

``` 
새로운 터미널에서 conda 환경 생성 밎 실행

```bash
$ conda create -n Bbung python=3.9   #새로운 conda 환경을 생성
$ conda activate Bbung    #'Bbung' 환경 활성화
``` 

#### 환경 설정
```bash
$ git clone https://github.com/seodaegal/Ko_Conversational_Context_Inference.git    # 프로젝트 레포지토리를 클론
$ cd Ko_Conversational_Context_Inference    # 프로젝트 디렉토리로 이동
$ pip install -r requirements.txt    # 프로젝트에 필요한 패키지를 설치

```
### 데이터셋 업로드
[대회맥락추론 가 유형](https://kli.korean.go.kr/benchmark/taskOrdtm/taskDownload.do?taskOrdtmId=144&clCd=END_TASK&subMenuId=sub02) 내 `대화맥락추론_데이터.zip`을 다운받아주세요.

다운받은 파일을 [`resource/data`](/resource/data) 폴더에 넣습니다.
```bash
$ unzip resource/data/대화맥락추론_데이터.zip
```
이후 '대화맥락추론'을 'data'로 바꿔주세요.
```bash
$ mv 대화맥락추론_dev.json data_dev.json
$ mv 대화맥락추론_test.json data_test.json
$ mv 대화맥락추론_train.json data_train.json
```

최종 `resource/data` 파일 형식:

```bash
KR-Conversation-Inference
├── resource
│   │ 
│   ├── data # 데이터 파일 형식
│   │   └── data_dev.json
│   │   └── data_test.json
│   │   └── data_train.json
│   │   └── sample.json
│   │ 
│   └── ...  
└── ... 
```


---
## 9. 평가 결과
어쩌구


---
## 10. 추후 연구 계획


---
## 11. License
저쩌고

---
## 12. Reference