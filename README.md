# Visual Dialogue 파인튜닝 프로젝트

Visual Dialogue 데이터셋을 활용한 Gemma-3 모델 파인튜닝 및 평가 시스템에 대한 오픈소스 프로젝트입니다. 이 프로젝트는 교육 및 연구 목적으로 자유롭게 사용 가능합니다.

## 📌 프로젝트 개요

본 프로젝트는 VisDial 데이터셋을 활용하여 멀티모달 언어 모델인 Gemma-3를 시각적 대화(Visual Dialogue)에 특화시키기 위한 파인튜닝 시스템을 제공합니다. 모델은 이미지를 보고 사용자와의 대화를 통해 적절한 응답을 생성하도록 학습됩니다.

## 📂 프로젝트 구조

```
reminder_tag/
├── README.md                    # 교수자용 메인 가이드
├── requirements.txt             # 의존성 패키지 목록
├── setup.py                     # 패키지 설치 스크립트
├── data/                        # 데이터 관련 파일
│   ├── README.md                # 데이터 설명서
│   └── download_data.py         # 자동 다운로드 스크립트
├── src/                         # 소스 코드 모듈
│   ├── data_generation/         # 데이터 생성 모듈
│   │   ├── __init__.py
│   │   ├── gen_base.py          # 기본 데이터 생성
│   │   ├── gen_api.py           # API 기반 데이터 생성
│   │   └── gen_eval.py          # 평가 데이터 생성
│   ├── training/                # 파인튜닝 모듈
│   │   ├── __init__.py
│   │   └── finetune.py          # 파인튜닝 스크립트
│   ├── inference/               # 추론 모듈
│   │   ├── __init__.py
│   │   └── inference.py         # 모델 추론 스크립트
│   ├── evaluation/              # 평가 모듈
│   │   ├── __init__.py
│   │   └── evaluation.py        # 성능 평가 스크립트
│   └── utils/                   # 공통 유틸리티
│       ├── __init__.py
│       └── helpers.py           # 공통 함수 모음
├── configs/                     # 설정 파일
│   ├── training_config.yaml     # 훈련 설정
│   └── evaluation_config.yaml   # 평가 설정
├── scripts/                     # 실행 스크립트
│   ├── run_pipeline.sh          # 전체 파이프라인 실행
│   └── quick_start.sh           # 빠른 시작 스크립트
├── notebooks/                   # 교육용 노트북
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_finetuning.ipynb
│   └── 03_evaluation.ipynb
└── tests/                       # 단위 테스트
    └── test_evaluation.py
```

## 🚀 시작하기

### 필수 요구사항

- Python 3.9+
- CUDA 지원 GPU (최소 16GB 이상 VRAM 권장)
- 최소 32GB RAM
- 200GB 이상의 저장 공간 (데이터셋 및 모델 포함)

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/2025-1-nlp-intro-project/multimodal_memory.git
cd multimodal_memory

# 2. 가상 환경 생성 및 활성화
conda create -n visdial python=3.9 -y
conda activate visdial

# 3. 의존성 설치
pip install -r requirements.txt

# 4. Unsloth 설치 (Gemma-3 모델 최적화 라이브러리)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 데이터셋 다운로드

프로젝트에서 제공하는 스크립트를 통해 Visual Dialogue 데이터셋과 COCO 이미지를 자동으로 다운로드할 수 있습니다:

```bash
# 데이터셋 자동 다운로드
python data/download_data.py

# 특정 분할만 다운로드 (선택적)
python data/download_data.py --split train
python data/download_data.py --split val
```

## 💾 데이터 생성

모델 훈련에 필요한 데이터 생성 과정입니다.

### 기본 데이터 생성

```bash
# 기본 데이터셋 생성
python src/data_generation/gen_base.py \
  --output_path data/generated/base_dataset.json \
  --sample_count 50000
```

### API 기반 데이터 생성

고품질 응답 생성을 위한 API 기반 데이터 생성:

```bash
python src/data_generation/gen_api.py \
  --output_path data/generated/api_dataset.json \
  --model "gemma3:27b"
```

### 평가 데이터 생성

모델 평가를 위한 데이터셋 생성:

```bash
python src/data_generation/gen_eval.py \
  --output_path data/generated/eval_dataset.json
```

## 🔧 모델 파인튜닝

모델 파인튜닝은 다음 명령어로 실행할 수 있습니다:

```bash
# 기본 설정으로 파인튜닝
python src/training/finetune.py --config configs/training_config.yaml

# 고급 설정으로 파인튜닝
python src/training/finetune.py \
  --model_name "unsloth/gemma-3-4b-it" \
  --dataset_path data/generated/api_dataset.json \
  --output_dir outputs/gemma3_finetune \
  --batch_size 2 \
  --grad_accum 4 \
  --lr 1e-5 \
  --epochs 3
```

### 파인튜닝 설정 옵션

`configs/training_config.yaml`에서 다음과 같은 설정을 조정할 수 있습니다:

```yaml
# 모델 설정
model:
  name: "unsloth/gemma-3-4b-it"
  load_in_4bit: true
  use_gradient_checkpointing: "unsloth"

# PEFT 설정
peft:
  finetune_vision_layers: true
  finetune_language_layers: true
  finetune_attention_modules: true
  finetune_mlp_modules: true
  r: 16
  lora_alpha: 16
  lora_dropout: 0
  bias: "none"
  use_rslora: false

# 훈련 설정
training:
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  warmup_steps: 5
  learning_rate: 1e-5
  fp16: false
  bf16: true
  logging_steps: 1
  optim: "adamw_8bit"
  weight_decay: 0.01
  lr_scheduler_type: "linear"
  max_seq_length: 2048
  output_dir: "outputs"
```

## 🔍 추론 실행

파인튜닝된 모델을 사용하여 추론을 실행하는 방법:

```bash
# 기본 추론
python src/inference/inference.py \
  --model_path outputs/gemma3_finetune \
  --image_path path/to/image.jpg \
  --conversation "Q: What do you see in this image?\nA: I see children playing on the beach.\nQ: How many children are there?"

# 배치 추론
python src/inference/inference.py \
  --model_path outputs/gemma3_finetune \
  --batch_file data/test_batch.json \
  --output_file results/batch_results.json
```

## 📊 평가 실행

모델 성능 평가를 위한 스크립트 실행:

```bash
# 기본 평가
python src/evaluation/evaluation.py \
  --prediction_file results/batch_results.json \
  --groundtruth_file data/visdial/data/visdial_1.0_val.json

# 상세 평가 지표 출력
python src/evaluation/evaluation.py \
  --prediction_file results/batch_results.json \
  --groundtruth_file data/visdial/data/visdial_1.0_val.json \
  --verbose
```

### 평가 지표

Visual Dialogue 모델 평가에 사용되는 주요 지표:

- **MRR (Mean Reciprocal Rank)**: 정답의 평균 역순위 (높을수록 좋음)
- **Recall@k**: 상위 k개 응답 중에 정답이 있는 비율 (k=1, 5, 10)
- **NDCG (Normalized Discounted Cumulative Gain)**: 순위 기반 정규화된 평가 지표
- **Mean Rank**: 정답의 평균 순위 (낮을수록 좋음)

## 🧪 전체 파이프라인 실행

단일 명령으로 전체 파이프라인(데이터 생성, 훈련, 평가)을 실행할 수 있습니다:

```bash
# 전체 파이프라인 실행
./scripts/run_pipeline.sh

# 고급 설정으로 실행
./scripts/run_pipeline.sh --config configs/custom_config.yaml --output experiment_1
```
