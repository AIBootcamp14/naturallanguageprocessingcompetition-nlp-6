[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/HS6nBbT4)
# Dialogue Summarization 경진대회

<br>

## 프로젝트 소개
### <프로젝트 소개>
- Dialogue Summarization 경진대회는 주어진 데이터를 활용하여 일상 대화에 대한 요약을 효과적으로 생성하는 모델을 개발하는 대회임.

### <작품 소개>
- baseline code 에서 digit82/kobart-summarization, gogamza/kobart-base-v2(이하 gogamza) 모델을 학습해보고 gogamza 모델의 rouge 점수가 약간 높음을 확인함.


<br>

## 팀 구성원

| ![박패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![이패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![최패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![김패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) | ![오패캠](https://avatars.githubusercontent.com/u/156163982?v=4) |
| :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: | :--------------------------------------------------------------: |
|            [김시진](https://github.com/UpstageAILab)             |            [신준엽](https://github.com/UpstageAILab)             |            [이가은](https://github.com/UpstageAILab)             |            [이건희](https://github.com/UpstageAILab)             |            [이찬](https://github.com/UpstageAILab)             |            [송인섭](https://github.com/UpstageAILab)             |
|                            팀장, 담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |                            담당 역할                             |

<br>

## 1. 개발 환경 및 기술 스택
- 주 언어 : python_
- 버전 및 이슈관리 : github, 슬랙
- 협업 툴 : _ex) github, notion, 슬랙

<br>

## 2. 프로젝트 구조
```
├── code
│   ├── jupyter_notebooks
│   │   └── baseline.ipynb
│   │   └── solar_api.ipynb
│   ├── config.yaml
│   └── requirements.txt
└── data
    ├── dev.csv
    ├── sample_submission.csv
    ├── test.csv
    └── train.csv
```

<br>

## 3. 시도해본 것들
### Try 1
- baseline code 에서 digit82/kobart-summarization, gogamza/kobart-base-v2(이하 gogamza) 모델을 학습해보고 gogamza 모델의 rouge 점수가 약간 높음을 확인함.
- 추론 결과는 아래 명령어를 통해 확인함.
- grep -A 5 -B 5 "best_model" checkpoint-*/trainer_state.json
- gogamza 모델의 checkpoint bestfit 추론 결과:
- final result 45.4018

### Try 2
- k-fold 앙상블 학습 및 추론
train.csv 학습 파일(이하 학습데이터)과 dev.csv 검증 파일(이하 검증데이터)을 모두 학습에 사용해 보았으나 성능 개선 없었음. 

### Try 3
- 데이터 EDA
- 학습 데이터 12,457 쌍
- 검증 데이터 499 쌍
- test.csv 파일 (이하 평가데이터) 499 쌍
- 평가데이터는 공개(250 쌍), 비공개(249 쌍) 데이터 나뉨
- 학습, 평가데이터에서 topic 컬럼이 합쳐서 약 9000 건 이상을 확인함. 학습과 평가 데이터에서는 많이 겹치지는 않았음.
- 학습, 평가데이터에서 #Person1#, #Person2#, #Person3#" 등의 발화자는 #Person7#까지 7명이 나오는 대화를 확인하였고 평가데이터에서는 #Person3#까지 나오는 대화를 확인함 \n
- 학습데이터 확인결과:
 - #Person3# -> 116대화
 - #Person4# -> 15대화
 - #Person5# -> 5대화
 - #Person6# -> 2대화
 - #Person7# -> 1대화
 <br>

- 학습, 검증데이터에서는 #Person1#, #PhoneNumber#, #Address# 등의 Special tokens 24개가 나오는 것을 확인하였으나 평가데이터에서는 기본 config 설정과 마찬가지로 6가지의 Special tokens을 확인함 
<br>

### Try 4
- solar API 프롬프트 엔지니어링을 통한 요약
- final result 43.2775

### Try 5
- EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks 논문 참고하여 데이터 증강 
1. 유의어로 교체(Synonym Replacement, SR): 문장에서 랜덤으로 stop words가 아닌 n 개의 단어들을 선택해 임의로 선택한 동의어들 중 하나로 바꾸는 기법.
2. 랜덤 삽입(Random Insertion, RI): 문장 내에서 stop word를 제외한 나머지 단어들 중에서, 랜덤으로 선택한 단어의 동의어를 임의로 정한다. 그리고 동의어를 문장 내 임의의 자리에 넣는걸 n번 반복한다.
3. 랜덤 교체(Random Swap, RS): 무작위로 문장 내에서 두 단어를 선택하고 위치를 바꾼다. 이것도 n번 반복
4. 랜덤 삭제(Random Deletion, RD): 확률 p를 통해 문장 내에 있는 각 단어들을 랜덤하게 삭제한다.

- Word2vector, TF-IDF 기술을 통해 유의어 사전 생성 
- nltk, konlpy 설치

### Try 6
- 학습 및 검증 데이터 노이즈 제거
- 노이즈 제거 이후 다시 증강 

<br>

## 4. 작품 아키텍처(필수X)
- #### _아래 이미지는 예시입니다_
![이미지 설명](https://www.cadgraphics.co.kr/UPLOAD/editor/2024/07/04//2024726410gH04SyxMo3_editor_image.png)

<br>

## 5. 트러블 슈팅
### 1. OOO 에러 발견

#### 설명
- _프로젝트 진행 중 발생한 트러블에 대해 작성해주세요_

#### 해결
- _프로젝트 진행 중 발생한 트러블 해결방법 대해 작성해주세요_

<br>

## 6. 프로젝트 회고
### 박패캠
- _프로젝트 회고를 작성해주세요_

<br>

## 7. 참고자료
- _참고자료를 첨부해주세요_
