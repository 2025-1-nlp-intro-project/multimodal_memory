# helpers.py
"""
Visual Dialogue 프로젝트를 위한 공통 유틸리티 함수들
"""

import json
import base64
import os
import requests
from io import BytesIO
from PIL import Image
from typing import Dict, List, Optional, Union, Any
import logging
from tqdm import tqdm
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageProcessor:
    """이미지 처리 관련 유틸리티 클래스"""
    
    @staticmethod
    def load_image_from_path(image_path: str) -> Image.Image:
        """로컬 경로에서 이미지 로드"""
        try:
            return Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            raise
        except Exception as e:
            logger.error(f"이미지 로드 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def load_image_from_url(image_url: str) -> Image.Image:
        """URL에서 이미지 다운로드 및 로드"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except requests.RequestException as e:
            logger.error(f"이미지 URL에서 다운로드 실패: {image_url}, 오류: {e}")
            raise
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """이미지를 base64로 인코딩"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except FileNotFoundError:
            logger.error(f"이미지 파일을 찾을 수 없습니다: {image_path}")
            raise
        except Exception as e:
            logger.error(f"Base64 인코딩 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def encode_image_url_to_base64(image_url: str) -> str:
        """URL 이미지를 base64로 인코딩"""
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            return base64.b64encode(response.content).decode('utf-8')
        except requests.RequestException as e:
            logger.error(f"이미지 URL 다운로드 실패: {image_url}, 오류: {e}")
            raise

class DataProcessor:
    """데이터 처리 관련 유틸리티 클래스"""
    
    @staticmethod
    def load_json(file_path: str) -> Dict:
        """JSON 파일 로드"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON 파일을 찾을 수 없습니다: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 오류: {e}")
            raise
    
    @staticmethod
    def save_json(data: Any, file_path: str, indent: int = 2) -> None:
        """JSON 파일 저장"""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            logger.info(f"JSON 파일 저장 완료: {file_path}")
        except Exception as e:
            logger.error(f"JSON 파일 저장 중 오류 발생: {e}")
            raise
    
    @staticmethod
    def load_existing_results(file_path: str) -> tuple[List[Dict], set]:
        """기존 결과 파일 로드 및 처리된 ID 세트 반환"""
        if os.path.exists(file_path):
            try:
                results = DataProcessor.load_json(file_path)
                processed_ids = {entry["image_id"] for entry in results}
                logger.info(f"기존 결과 {len(results)}개 로드됨: {file_path}")
                return results, processed_ids
            except Exception as e:
                logger.warning(f"기존 결과 파일 로드 실패: {e}")
                return [], set()
        else:
            logger.info(f"기존 결과 파일이 없음, 새로 시작: {file_path}")
            return [], set()

class DialogueFormatter:
    """대화 포맷팅 관련 유틸리티 클래스"""
    
    @staticmethod
    def format_conversation(dialog_entry: Dict, questions: List[str], answers: List[str], 
                          include_last_answer: bool = True) -> str:
        """대화 엔트리를 텍스트로 포맷팅"""
        conversation = ""
        dialog_turns = dialog_entry.get("dialog", [])
        
        for i, turn in enumerate(dialog_turns):
            if "question" in turn:
                q_idx = turn["question"]
                question_text = questions[q_idx]
                conversation += f"Q: {question_text}\n"
            
            if "answer" in turn:
                # 마지막 답변 포함 여부 결정
                if not include_last_answer and i == len(dialog_turns) - 1:
                    continue
                a_idx = turn["answer"]
                answer_text = answers[a_idx]
                conversation += f"A: {answer_text}\n\n"
        
        return conversation.strip()
    
    @staticmethod
    def extract_answer_from_response(response: str) -> str:
        """응답에서 최종 답변 추출"""
        # <reasoning> 태그가 있는 경우 그 이후 텍스트 추출
        if '</reasoning>' in response:
            return response.split('</reasoning>')[-1].strip()
        elif '<reasoning>' in response and '</reasoning>' not in response:
            # 열린 태그만 있는 경우 그 이후 텍스트
            return response.split('<reasoning>')[-1].strip()
        else:
            # 태그가 없는 경우 전체 응답 반환
            return response.strip()
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """텍스트 정규화 (평가용)"""
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text

class PromptBuilder:
    """프롬프트 생성 관련 유틸리티 클래스"""
    
    @staticmethod
    def build_system_prompt(mode: str = "base") -> str:
        """시스템 프롬프트 생성"""
        if mode == "base":
            return """
You are given a conversation between a human and an AI, regarding a single image.
Based on the conversation above, state the final short conclusion that answers the last question.
"""
        elif mode == "reasoning":
            return """
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
"""
        elif mode == "training":
            return """
You are given a conversation between a human and an AI, regarding a single image.
"""
        else:
            raise ValueError(f"Unknown prompt mode: {mode}")
    
    @staticmethod
    def build_training_messages(conversation: str, image: Image.Image, 
                              response: str) -> List[Dict]:
        """훈련용 메시지 포맷 생성"""
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": PromptBuilder.build_system_prompt("training")}]
            },
            {
                "role": "user", 
                "content": [
                    {"type": "text", "text": conversation},
                    {"type": "image", "image": image}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}]
            }
        ]

class PathManager:
    """파일 경로 관리 유틸리티 클래스"""
    
    @staticmethod
    def get_image_path(image_id: int, image_dir: str, dataset: str = "val2018") -> str:
        """이미지 ID로부터 이미지 경로 생성"""
        if dataset == "val2018":
            filename = f"VisualDialog_val2018_{str(image_id).zfill(12)}.jpg"
        elif dataset == "train2014":
            filename = f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
        else:
            filename = f"{str(image_id).zfill(12)}.jpg"
        
        return os.path.join(image_dir, filename)
    
    @staticmethod
    def ensure_dir_exists(file_path: str) -> None:
        """디렉토리가 존재하지 않으면 생성"""
        dir_path = os.path.dirname(file_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

class ProgressTracker:
    """진행 상황 추적 유틸리티 클래스"""
    
    @staticmethod
    def create_progress_bar(iterable, desc: str = "Processing") -> tqdm:
        """진행률 표시바 생성"""
        return tqdm(iterable, desc=desc)
    
    @staticmethod
    def log_progress(current: int, total: int, item_name: str = "항목") -> None:
        """진행 상황 로그"""
        percentage = (current / total) * 100
        logger.info(f"{item_name} 진행 상황: {current}/{total} ({percentage:.1f}%)")

class ErrorHandler:
    """에러 처리 유틸리티 클래스"""
    
    @staticmethod
    def safe_execute(func, *args, error_msg: str = "실행 중 오류 발생", 
                    default_return=None, **kwargs):
        """안전한 함수 실행 (에러 시 기본값 반환)"""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"{error_msg}: {e}")
            return default_return
    
    @staticmethod
    def validate_file_exists(file_path: str, file_type: str = "파일") -> bool:
        """파일 존재 여부 확인"""
        if not os.path.exists(file_path):
            logger.error(f"{file_type}이 존재하지 않습니다: {file_path}")
            return False
        return True

# 전역 설정
DEFAULT_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 1.2,
    "top_p": 0.9,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-5,
    "max_seq_length": 2048
}

def get_config(key: str, default=None):
    """설정값 가져오기"""
    return DEFAULT_CONFIG.get(key, default)