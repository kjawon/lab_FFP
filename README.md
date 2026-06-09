# lab_FFP

## 프로젝트 개요

`lab_FFP`는 파일 포렌식 기반 연구 프로젝트로, GovDocs1 데이터셋의 원시 파일 조각(fragment)을 이용해 파일 형식과 확장자를 분류하고, 클러스터링/복원 가능성을 탐색합니다.

## 연구 목표

- 헤더가 제거된 4KB 파일 조각에서 `PDF`, `JPG`, `TXT` 등의 파일 유형을 자동 분류
- 바이트 시퀀스, 바이트 히스토그램, 엔트로피 맵을 결합한 하이브리드 모델 개발
- 다중 작업 학습(MTL), Hard Negative Mining, ArcFace 임베딩을 통해 분류 및 파일 복원 성능 향상
- 분류 결과를 기반으로 HDBSCAN/KMeans 클러스터링을 이용한 파일 재조합 검증

## 데이터셋 및 처리

- 데이터원: `https://digitalcorpora.s3.amazonaws.com/corpora/files/govdocs1/zipfiles/`
- 주요 처리 방식
  - 각 파일을 4KB 단위로 조각화
  - 첫 4KB 헤더 조각은 제거
  - 부족한 길이는 `0x00`으로 패딩
  - `.pdf`, `.jpg`, `.txt` 등의 확장자 라벨링
  - 추가 실험에서는 확장자별 히스토그램 및 엔트로피 맵 특징 생성

## 주요 실험 스크립트

- `lab01.py`
  - GovDocs1 10개 ZIP 파일 다운로드
  - 4KB 조각 데이터셋 구성
  - 1D CNN + Transformer 기반 `HybridByteModel` 학습

- `lab02.py`
  - 학습된 `HybridByteModel` 로드 및 검증
  - `000.zip` 테스트 데이터로 성능 측정 및 혼동 행렬 시각화

- `lab03.py`
  - 파일 ID 기반 MTL 데이터셋 구성
  - 3개 클래스 균형 데이터셋 생성
  - DBSCAN/TSNE 클러스터링 분석

- `lab04.py`
  - 5개 ZIP 데이터에서 다양한 확장자 클래스 수집
  - 256차원 바이트 히스토그램 특징 추가
  - `UltimateMTLModel` 학습

- `lab05.py`
  - 10개 ZIP 기반 대규모 학습
  - 하이브리드 모델 저장 및 평가

- `lab06.py`
  - 3-way 특징(바이트, 히스토그램, 엔트로피) 기반 `Final_Top3_Deepzzle` 모델 학습
  - 클래스별 샘플 제한 및 가중치 균형 적용

- `lab07.py`
  - 실세계 데이터 분포 그대로 수용한 학습
  - 언더샘플링 없이 실제 확률 분포 유지

- `lab08.py`
  - `RealWorld_MTL_Deepzzle` 모델 평가
  - Top-1/Top-3 분석, 시각화, 임베딩 기반 클러스터링

- `lab09.py`
  - Hard Negative Mining 도입 모델 학습
  - 동일 확장자 내에서 더 어려운 네거티브 샘플 사용

- `lab10.py`
  - Hard Negative 모델 평가
  - HDBSCAN 기반 파일 복원 품질 측정

- `lab11.py`
  - ArcFace 기반 두 단계 학습
  - 파일 단위 고유 ID를 이용한 강력한 임베딩 학습

- `lab12.py` ~ `lab15.py`
  - ArcFace 모델 평가 및 클러스터링 검증
  - `KMeans`, `HDBSCAN`, `TSNE` 기반 분석 및 시각화

## 연구 키워드

- File Fragment Classification
- Raw Byte Modeling
- 1D CNN + Transformer
- Multi-Task Learning
- Byte Histogram / Entropy Map
- Hard Negative Mining
- ArcFace Embedding
- HDBSCAN / KMeans 클러스터링

## 실행 환경

- Google Colab 권장
- Google Drive 마운트 필요
- Python 주요 라이브러리
  - `torch`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `tqdm`
  - `scikit-learn`

## 주의 사항

- 데이터 다운로드 용량이 크고, 학습 시 GPU/디스크 공간이 많이 필요합니다.
- 각 스크립트는 Google Drive에 모델을 저장하고 불러오는 구조입니다.

---

본 `README.md`는 현재 코드 베이스의 실험 흐름과 목적을 정리한 문서입니다. 필요하다면 각 `labXX.py`별 상세 설명이나 실행 예시도 추가하겠습니다.
