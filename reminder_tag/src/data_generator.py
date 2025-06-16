#!/usr/bin/env python3
"""
Visual Dialogue 학습을 위한 데이터 생성 모듈

이 모듈은 Visual Dialogue 데이터셋을 다양한 방식으로 가공하여 학습 데이터를 생성합니다:
1. Ollama 모델을 활용한 기본 API 생성
2. 추론(reasoning) 기반 프롬프트 생성
3. 평가용 데이터 생성
"""

import json
import os
import base64
import requests
from io import BytesIO
from typing import Dict, List, Optional, Any
import logging
from tqdm import tqdm
from pycocotools.coco import COCO
import ollama
import argparse


class VisualDialogueDataGenerator:
    """
    Visual Dialogue 모델 학습 데이터를 생성하는 클래스
    """
    
    def __init__(self, 
                 visdial_path: str,
                 annotation_file: str,
                 model_name: str = "gemma3:27b",
                 output_path: str = "output_data.json"):
        """
        데이터 생성기 초기화 함수
        
        Args:
            visdial_path: VisDial 데이터셋 JSON 경로
            annotation_file: COCO 어노테이션 파일 경로
            model_name: 데이터 생성에 사용할 Ollama 모델명
            output_path: 생성 데이터 저장 경로
        """
        self.visdial_path = visdial_path
        self.annotation_file = annotation_file
        self.model_name = model_name
        self.output_path = output_path
        
        # 로깅 설정
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 데이터 로드
        self.coco = COCO(annotation_file) if annotation_file else None
        self.visdial_data = self._load_visdial_data()
        self.questions = self.visdial_data["data"]["questions"]
        self.answers = self.visdial_data["data"]["answers"]
        self.dialogs = self.visdial_data["data"]["dialogs"]
        
        # 기존 결과 로드
        self.results = self._load_existing_results()
        self.processed_ids = {entry["image_id"] for entry in self.results}
    
    def _load_visdial_data(self) -> Dict:
        """
        Visual Dialogue 데이터셋을 JSON 파일에서 불러옵니다.
        """
        with open(self.visdial_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_existing_results(self) -> List[Dict]:
        """
        기존 결과 파일이 있으면 불러와서 이어서 생성할 수 있도록 합니다.
        """
        if os.path.exists(self.output_path):
            try:
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []
    
    def encode_image_from_url(self, url: str) -> str:
        """
        이미지 URL을 base64로 인코딩합니다.
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            img_bytes = BytesIO(response.content)
            return base64.b64encode(img_bytes.read()).decode('utf-8')
        except Exception as e:
            self.logger.error(f"Failed to encode image from {url}: {e}")
            raise
    
    def build_base_prompt(self, conversation_text: str) -> str:
        """
        기본 데이터 생성을 위한 프롬프트를 만듭니다.
        """
        return f"""
You are given a conversation between a human and an AI, regarding a single image.
Based on the conversation above, state the final short conclusion that answers the last question.
"""
    
    def build_reasoning_prompt(self, conversation_text: str) -> str:
        """
        reasoning 태그가 포함된 고품질 학습 데이터 생성을 위한 프롬프트를 만듭니다.
        """
        return f"""
You are given a conversation between a human and an AI, regarding a single image.
You are an AI assistant performing image-based Q&A.

Before answering the user's final question, you must generate a <reasoning> tag that includes:
1. A brief summary of the prior conversation flow and context
2. The key visual elements in the image you're focusing on
3. The user's intention or what they're asking for
4. Your internal reasoning steps in arriving at the answer

After the </reasoning> tag, provide your direct answer to the user's question.

Respond in this format:
<reasoning>
...reasoning here...
</reasoning>
Final answer sentence.

Example format:
<reasoning>
The conversation has been about identifying animals in a zoo setting. Looking at the image, I can see black and white striped animals in an enclosure. The user is asking about the specific type of animal. Based on the distinctive stripe pattern, these are clearly zebras.
</reasoning>
These are zebras.
"""
    
    def build_conversation_string(self, dialog: Dict, include_last_answer: bool = False) -> str:
        """Build conversation string from dialog turns"""
        conversation = ""
        turns = dialog["dialog"]
        
        for i, turn in enumerate(turns):
            if "question" in turn:
                q = self.questions[turn["question"]]
                conversation += f"Q: {q}\n"
            
            if "answer" in turn:
                # Skip last answer if not including it
                if not include_last_answer and i == len(turns) - 1:
                    continue
                a = self.answers[turn["answer"]]
                conversation += f"A: {a}\n"
        
        return conversation.strip()
    
    def generate_with_ollama(self, prompt: str, conversation: str, image_data: str) -> str:
        """Generate response using Ollama model"""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"conversation:\n{conversation}", "images": [image_data]}
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            self.logger.error(f"Ollama generation failed: {e}")
            raise
    
    def generate_base_data(self, max_samples: int = 50000) -> None:
        """Generate basic training data"""
        self.logger.info("Starting base data generation...")
        
        processed_count = 0
        
        for dialog in tqdm(self.dialogs[:max_samples], desc="Generating base data"):
            image_id = dialog["image_id"]
            
            if image_id in self.processed_ids:
                continue
            
            try:
                # Get image URL from COCO if available
                if self.coco:
                    img_info = self.coco.loadImgs(image_id)
                    if not img_info:
                        continue
                    image_url = img_info[0].get("coco_url")
                    if not image_url:
                        continue
                    
                    # Encode image
                    image_b64 = self.encode_image_from_url(image_url)
                else:
                    self.logger.warning("No COCO annotations provided, skipping image loading")
                    continue
                
                # Build conversation
                conversation = self.build_conversation_string(dialog)
                
                # Generate response
                prompt = self.build_base_prompt(conversation)
                response = self.generate_with_ollama(prompt, conversation, image_b64)
                
                # Save result
                result_entry = {
                    "image_id": image_id,
                    "image_url": image_url,
                    "conversation": conversation,
                    "response": response
                }
                
                self.results.append(result_entry)
                self.processed_ids.add(image_id)
                processed_count += 1
                
                # Save periodically
                if processed_count % 100 == 0:
                    self._save_results()
                    self.logger.info(f"Processed {processed_count} samples, saved checkpoint")
                
            except Exception as e:
                self.logger.error(f"Failed to process image {image_id}: {e}")
                continue
        
        # Final save
        self._save_results()
        self.logger.info(f"Base data generation completed. Total: {len(self.results)} samples")
    
    def generate_reasoning_data(self, max_samples: int = 50000) -> None:
        """Generate training data with reasoning"""
        self.logger.info("Starting reasoning data generation...")
        
        processed_count = 0
        
        for dialog in tqdm(self.dialogs[:max_samples], desc="Generating reasoning data"):
            image_id = dialog["image_id"]
            
            if image_id in self.processed_ids:
                continue
            
            try:
                # Get image URL from COCO
                if self.coco:
                    img_info = self.coco.loadImgs(image_id)
                    if not img_info:
                        continue
                    image_url = img_info[0].get("coco_url")
                    if not image_url:
                        continue
                    
                    # Encode image
                    image_b64 = self.encode_image_from_url(image_url)
                else:
                    continue
                
                # Build conversation (without last answer)
                conversation = self.build_conversation_string(dialog, include_last_answer=False)
                
                # Generate response with reasoning
                prompt = self.build_reasoning_prompt(conversation)
                response = self.generate_with_ollama(prompt, conversation, image_b64)
                
                # Save result
                result_entry = {
                    "image_id": image_id,
                    "image_url": image_url,
                    "conversation": conversation,
                    "response": response
                }
                
                self.results.append(result_entry)
                self.processed_ids.add(image_id)
                processed_count += 1
                
                # Save periodically
                if processed_count % 100 == 0:
                    self._save_results()
                    self.logger.info(f"Processed {processed_count} samples, saved checkpoint")
                
            except Exception as e:
                self.logger.error(f"Failed to process image {image_id}: {e}")
                continue
        
        # Final save
        self._save_results()
        self.logger.info(f"Reasoning data generation completed. Total: {len(self.results)} samples")
    
    def _save_results(self) -> None:
        """Save results to output file"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Generate Visual Dialogue training data")
    parser.add_argument("--visdial_path", type=str, required=True,
                       help="Path to VisDial dataset JSON file")
    parser.add_argument("--annotation_file", type=str,
                       help="Path to COCO annotations file")
    parser.add_argument("--model_name", type=str, default="gemma3:27b",
                       help="Ollama model name for generation")
    parser.add_argument("--output_path", type=str, default="generated_data.json",
                       help="Output file path")
    parser.add_argument("--max_samples", type=int, default=50000,
                       help="Maximum number of samples to process")
    parser.add_argument("--mode", type=str, choices=["base", "reasoning"], default="base",
                       help="Generation mode: base or reasoning")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = VisualDialogueDataGenerator(
        visdial_path=args.visdial_path,
        annotation_file=args.annotation_file,
        model_name=args.model_name,
        output_path=args.output_path
    )
    
    # Generate data based on mode
    if args.mode == "base":
        generator.generate_base_data(args.max_samples)
    elif args.mode == "reasoning":
        generator.generate_reasoning_data(args.max_samples)
    
    print(f"✅ Data generation completed! Output saved to {args.output_path}")


if __name__ == "__main__":
    main()