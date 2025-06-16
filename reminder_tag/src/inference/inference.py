#!/usr/bin/env python3

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse

import torch
from PIL import Image
from tqdm import tqdm

# Unsloth imports
from unsloth import FastVisionModel
from transformers import TextStreamer

class VisualDialogueInference:
    """
    Visual Dialogue 모델 추론을 위한 엔진 클래스
    """
    
    def __init__(self, 
                 model_path: str,
                 load_in_4bit: bool = True,
                 device: str = "auto"):
        """
        추론 엔진 초기화 함수
        
        Args:
            model_path: 파인튜닝된 모델 경로
            load_in_4bit: 4비트 정밀도로 모델 로드 여부
            device: 추론에 사용할 디바이스
        """
        self.model_path = model_path
        self.load_in_4bit = load_in_4bit
        self.device = device
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.model = None
        self.tokenizer = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """
        파인튜닝된 모델과 토크나이저를 로드합니다.
        """
        self.logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")
        
        try:
            # Load model and tokenizer
            self.model, self.tokenizer = FastVisionModel.from_pretrained(
                self.model_path,
                load_in_4bit=self.load_in_4bit,
            )
            
            # Set to inference mode
            FastVisionModel.for_inference(self.model)
            
            # Move to device if specified
            if self.device != "auto" and torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.logger.info("Model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_conversation_input(self, conversation: str, image: Image.Image) -> Dict:
        """
        대화와 이미지를 모델 입력에 맞게 준비합니다.
        
        Args:
            conversation: 대화 이력 문자열
            image: PIL 이미지 객체
        Returns:
            모델 입력용 딕셔너리
        """
        system_instruction = """
You are given a conversation between a human and an AI, regarding a single image.
"""
        
        # Prepare messages in chat format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_instruction}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": conversation.strip()},
                    {"type": "image", "image": image},
                ],
            },
        ]
        
        return messages
    
    def generate_response(self, 
                         conversation: str, 
                         image: Image.Image,
                         max_new_tokens: int = 512,
                         temperature: float = 1.2,
                         top_p: float = 0.9,
                         use_streamer: bool = False) -> str:
        """
        Generate response for a visual dialogue
        
        Args:
            conversation: Dialog history string
            image: PIL Image object
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            use_streamer: Whether to use text streamer (for real-time output)
            
        Returns:
            Generated response string
        """
        try:
            # Prepare input
            messages = self.prepare_conversation_input(conversation, image)
            
            # Apply chat template
            input_text = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            )
            
            # Move to device
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Setup streamer if requested
            streamer = None
            if use_streamer:
                streamer = TextStreamer(self.tokenizer, skip_prompt=True)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode response
            if not use_streamer:
                # Get only the newly generated tokens
                new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                return response.strip()
            else:
                return ""  # Output is streamed directly
                
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"
    
    def batch_inference(self, 
                       conversations: List[str], 
                       images: List[Image.Image],
                       **generation_kwargs) -> List[str]:
        """
        Run batch inference on multiple conversations
        
        Args:
            conversations: List of dialog history strings
            images: List of PIL Image objects
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated responses
        """
        if len(conversations) != len(images):
            raise ValueError("Number of conversations and images must match")
        
        responses = []
        
        for conv, img in tqdm(zip(conversations, images), 
                             total=len(conversations), 
                             desc="Generating responses"):
            response = self.generate_response(conv, img, **generation_kwargs)
            responses.append(response)
        
        return responses
    
    def inference_from_visdial_data(self,
                                   visdial_path: str,
                                   image_dir: str,
                                   output_path: str,
                                   max_samples: Optional[int] = None,
                                   **generation_kwargs) -> None:
        """
        Run inference on Visual Dialogue dataset
        
        Args:
            visdial_path: Path to VisDial JSON file
            image_dir: Directory containing images
            output_path: Output file for predictions
            max_samples: Maximum number of samples to process
            **generation_kwargs: Additional generation parameters
        """
        self.logger.info(f"Loading VisDial data from: {visdial_path}")
        
        # Load VisDial data
        with open(visdial_path, 'r', encoding='utf-8') as f:
            visdial_data = json.load(f)
        
        questions = visdial_data["data"]["questions"]
        answers = visdial_data["data"]["answers"]
        dialogs = visdial_data["data"]["dialogs"]
        
        # Limit samples if specified
        if max_samples:
            dialogs = dialogs[:max_samples]
        
        results = []
        failed_count = 0
        
        for dialog in tqdm(dialogs, desc="Processing dialogs"):
            image_id = dialog["image_id"]
            
            try:
                # Load image
                image_filename = f"VisualDialog_val2018_{str(image_id).zfill(12)}.jpg"
                image_path = os.path.join(image_dir, image_filename)
                
                if not os.path.exists(image_path):
                    self.logger.warning(f"Image not found: {image_path}")
                    failed_count += 1
                    continue
                
                image = Image.open(image_path).convert("RGB")
                
                # Build conversation string
                conversation = ""
                for turn in dialog["dialog"]:
                    if "question" in turn and "answer" in turn:
                        q = questions[turn["question"]]
                        a = answers[turn["answer"]]
                        conversation += f"Q: {q}\nA: {a}\n\n"
                
                # Generate response
                response = self.generate_response(
                    conversation.strip(), 
                    image, 
                    **generation_kwargs
                )
                
                # Save result
                result = {
                    "image_id": image_id,
                    "conversation": conversation.strip(),
                    "response": response
                }
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process image {image_id}: {e}")
                failed_count += 1
                continue
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Inference completed!")
        self.logger.info(f"Processed: {len(results)} samples")
        self.logger.info(f"Failed: {failed_count} samples")
        self.logger.info(f"Results saved to: {output_path}")


def main():
    """Main function for inference"""
    parser = argparse.ArgumentParser(description="Visual Dialogue Model Inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to fine-tuned model")
    parser.add_argument("--data_path", type=str,
                       help="Path to VisDial JSON file")
    parser.add_argument("--image_dir", type=str,
                       help="Directory containing images")
    parser.add_argument("--image_path", type=str,
                       help="Single image path for inference")
    parser.add_argument("--conversation", type=str,
                       help="Conversation string for single inference")
    parser.add_argument("--output_path", type=str, default="predictions.json",
                       help="Output file for predictions")
    parser.add_argument("--max_samples", type=int,
                       help="Maximum number of samples to process")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.2,
                       help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--load_in_4bit", action="store_true", default=True,
                       help="Load model in 4-bit precision")
    parser.add_argument("--stream", action="store_true",
                       help="Enable text streaming for single inference")
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inference_engine = VisualDialogueInference(
        model_path=args.model_path,
        load_in_4bit=args.load_in_4bit
    )
    
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    
    # Single image inference
    if args.image_path and args.conversation:
        print(f"🖼️  Processing single image: {args.image_path}")
        print(f"💬 Conversation: {args.conversation}")
        
        # Load image
        image = Image.open(args.image_path).convert("RGB")
        
        # Generate response
        print("🤖 Generating response...")
        response = inference_engine.generate_response(
            args.conversation,
            image,
            use_streamer=args.stream,
            **generation_kwargs
        )
        
        if not args.stream:
            print(f"📝 Response: {response}")
    
    # Batch inference on VisDial data
    elif args.data_path and args.image_dir:
        print(f"📊 Processing VisDial data: {args.data_path}")
        inference_engine.inference_from_visdial_data(
            visdial_path=args.data_path,
            image_dir=args.image_dir,
            output_path=args.output_path,
            max_samples=args.max_samples,
            **generation_kwargs
        )
    
    else:
        print("❌ Please provide either:")
        print("   1. --image_path and --conversation for single inference")
        print("   2. --data_path and --image_dir for batch inference")
        print("Use --help for more information")


if __name__ == "__main__":
    main()