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

### 빠른 시작
```bash
bash quick_start.sh
```

### 설치 방법

```bash
# 1. 저장소 클론
git clone https://github.com/2025-1-nlp-intro-project/multimodal_memory.git
cd multimodal_memory/reminder_tag

# 2. 가상 환경 생성 및 활성화
conda create -n visdial python=3.10 -y
conda activate visdial

# 3. 의존성 설치
pip install -r requirements.txt
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
# 기본 데이터 생성 (Ollama 사용)
python -m src.data_generation.gen_base \
    --annotation_file data/visdial/annotations/instances_train2014.json \
    --visdial_path data/visdial/data/visdial_1.0_train.json \
    --output_path outputs/data_base.json \
    --max_samples 1000

# 추론 과정 포함 데이터 생성
python -m src.data_generation.gen_api \
    --annotation_file data/visdial/annotations/instances_train2014.json \
    --visdial_path data/visdial/data/visdial_1.0_train.json \
    --output_path outputs/data_api.json \
    --export_training outputs/training_data.json \
    --max_samples 1000
```

### 평가 데이터 생성

모델 평가를 위한 데이터셋 생성:

```bash
# 평가 데이터 생성
python -m src.data_generation.gen_eval \
    --visdial_path data/visdial/data/visdial_1.0_val.json \
    --image_dir data/visdial/images/val2018/ \
    --output_path outputs/eval_data.json \
    --export_predictions outputs/predictions.json \
    --max_samples 500

# 평가 지표 계산
python src/evaluation/evaluate.py \
    --generated outputs/predictions.json \
    --ground_truth data/visdial/data/visdial_1.0_val.json \
    --output outputs/evaluation_results.json
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

### 평가 지표 (Evaluation Metrics)

Visual Dialogue 파인튜닝 프로젝트는 **BERTScore를 주요 메트릭**으로 사용하며, 보조 메트릭들과 함께 포괄적인 평가를 제공합니다.

**🎯 주요 메트릭: BERTScore**

#### BERTScore란?

BERTScore는 BERT의 사전 훈련된 컨텍스추얼 임베딩을 활용하여 텍스트 간 의미적 유사성을 측정하는 혁신적인 평가 지표입니다. 기존의 n-gram 기반 메트릭들(BLEU, ROUGE)과 달리, **단순한 단어 매칭을 넘어서 실제 의미를 이해**합니다.

#### 왜 BERTScore인가?

🔍 기존 메트릭의 한계
```
참조 답변: "The food was delicious."
생성 답변: "I loved the meal."
```

- **BLEU/ROUGE**: 단어가 달라서 낮은 점수 (0.0)
- **BERTScore**: 의미가 유사하므로 높은 점수 (0.85+)

✨ BERTScore의 장점

1. **의미적 이해**: 동의어, 패러프레이즈를 올바르게 인식
2. **컨텍스트 고려**: 문장 내 단어의 문맥적 의미 파악
3. **유연한 평가**: 다양한 표현 방식을 공정하게 평가
4. **인간 평가와 높은 상관관계**: 실제 인간의 판단과 매우 유사

## 🧪 전체 파이프라인 실행

단일 명령으로 전체 파이프라인(데이터 생성, 훈련, 평가)을 실행할 수 있습니다:

```bash
# 전체 파이프라인 실행
./scripts/run_pipeline.sh

# 고급 설정으로 실행
./scripts/run_pipeline.sh --config configs/custom_config.yaml --output experiment_1
```
