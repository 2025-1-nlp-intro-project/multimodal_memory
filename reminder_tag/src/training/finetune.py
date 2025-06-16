#!/usr/bin/env python3

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# Unsloth imports
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

# Transformers and training
from transformers import AutoProcessor
from trl import SFTTrainer, SFTConfig

# Utilities
import yaml
import argparse


@dataclass
class TrainingConfig:
    """
    학습 파라미터를 관리하는 설정 클래스
    """
    model_name: str = "unsloth/gemma-3-4b-it"
    load_in_4bit: bool = True
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    max_seq_length: int = 2048
    
    # Output and logging
    output_dir: str = "outputs"
    logging_steps: int = 1
    save_steps: int = 200
    save_total_limit: int = 2
    
    # Optimizer
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"


class VisualDialogueDataProcessor:
    """
    Visual Dialogue 학습 데이터를 전처리하는 클래스
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.system_message = """
You are given a conversation between a human and an AI, regarding a single image.
"""
    
    def load_training_data(self, data_path: str) -> List[Dict]:
        """
        학습 데이터를 로드하고 유효성을 검사합니다.
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Training data not found: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logging.info(f"Loaded {len(dataset)} training samples from {data_path}")
        return dataset
    
    def load_image_from_url(self, url: str) -> Image.Image:
        """
        이미지 URL에서 이미지를 불러옵니다.
        """
        import requests
        from io import BytesIO
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to load image from {url}: {e}")
            raise
    
    def convert_to_conversation(self, sample: Dict) -> Dict:
        """
        샘플 데이터를 학습용 대화 포맷으로 변환합니다.
        """
        try:
            img = self.load_image_from_url(sample["image_url"])
            
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": self.system_message}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": sample["conversation"]},
                            {"type": "image", "image": img},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": sample["response"]}],
                    },
                ],
            }
        except Exception as e:
            logging.error(f"Failed to convert sample: {e}")
            return None


class VisualDialogueTrainer:
    """Main trainer class for Visual Dialogue fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{config.output_dir}/training.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_model(self):
        """Load and configure the model"""
        logging.info(f"Loading model: {self.config.model_name}")
        
        # Load model and tokenizer
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            self.config.model_name,
            load_in_4bit=self.config.load_in_4bit,
            use_gradient_checkpointing="unsloth",
        )
        
        # Initialize processor
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name, 
            use_fast=True
        )
        
        # Apply PEFT configuration
        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        logging.info("Model loaded and configured successfully")
    
    def prepare_dataset(self, data_path: str) -> List[Dict]:
        """Prepare dataset for training"""
        processor = VisualDialogueDataProcessor(self.config)
        raw_dataset = processor.load_training_data(data_path)
        
        # Convert to conversation format
        converted_dataset = []
        for sample in tqdm(raw_dataset, desc="Converting dataset"):
            converted_sample = processor.convert_to_conversation(sample)
            if converted_sample is not None:
                converted_dataset.append(converted_sample)
        
        logging.info(f"Converted {len(converted_dataset)} samples for training")
        return converted_dataset
    
    def setup_trainer(self, train_dataset: List[Dict]):
        """Setup the SFT trainer"""
        FastVisionModel.for_training(self.model)
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            data_collator=UnslothVisionDataCollator(self.model, self.tokenizer),
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                warmup_steps=self.config.warmup_steps,
                learning_rate=self.config.learning_rate,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=self.config.logging_steps,
                optim=self.config.optim,
                weight_decay=self.config.weight_decay,
                lr_scheduler_type=self.config.lr_scheduler_type,
                seed=3407,
                output_dir=self.config.output_dir,
                report_to="none",
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=4,
                max_seq_length=self.config.max_seq_length,
                num_train_epochs=self.config.num_train_epochs,
                save_steps=self.config.save_steps,
                save_total_limit=self.config.save_total_limit,
            ),
        )
        
        logging.info("Trainer configured successfully")
    
    def train(self, data_path: str):
        """Execute the complete training pipeline"""
        logging.info("Starting Visual Dialogue fine-tuning...")
        
        # Load model
        self.load_model()
        
        # Prepare dataset
        train_dataset = self.prepare_dataset(data_path)
        
        # Setup trainer
        self.setup_trainer(train_dataset)
        
        # Start training
        logging.info("Beginning training...")
        trainer_stats = self.trainer.train()
        
        # Save model
        model_save_path = f"{self.config.output_dir}/final_model"
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        
        logging.info(f"Training completed. Model saved to {model_save_path}")
        logging.info(f"Training stats: {trainer_stats}")
        
        return trainer_stats


def load_config(config_path: str) -> TrainingConfig:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        logging.warning(f"Config file not found: {config_path}. Using default config.")
        return TrainingConfig()
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert dict to TrainingConfig
    config = TrainingConfig()
    for key, value in config_dict.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Fine-tune Gemma-3 for Visual Dialogue")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to training data JSON file")
    parser.add_argument("--output", type=str, default="outputs",
                       help="Output directory for model and logs")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    config.output_dir = args.output
    
    # Initialize trainer
    trainer = VisualDialogueTrainer(config)
    
    # Start training
    try:
        trainer.train(args.data)
        print("✅ Training completed successfully!")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()