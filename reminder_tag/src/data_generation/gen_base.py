# gen_base.py
"""
Visual Dialogue 기본 데이터 생성기
COCO 데이터셋과 Ollama를 사용하여 기본 대화 데이터를 생성합니다.
"""

import os
import json
from typing import Dict, List, Optional
from pycocotools.coco import COCO
import ollama
from ..utils.helpers import (
    ImageProcessor, DataProcessor, DialogueFormatter, 
    PromptBuilder, ProgressTracker, ErrorHandler, logger
)

class BaseDataGenerator:
    """
    Visual Dialogue 기본 데이터 생성기
    
    COCO 데이터셋의 이미지와 Visual Dialogue 대화를 활용하여
    Ollama 모델을 통해 기본 응답 데이터를 생성합니다.
    """
    
    def __init__(self, 
                 annotation_file: str,
                 visdial_path: str, 
                 output_path: str,
                 model_name: str = "gemma3:27b"):
        """
        초기화
        
        Args:
            annotation_file: COCO 주석 파일 경로
            visdial_path: Visual Dialogue 데이터 파일 경로  
            output_path: 출력 파일 경로
            model_name: 사용할 Ollama 모델명
        """
        self.annotation_file = annotation_file
        self.visdial_path = visdial_path
        self.output_path = output_path
        self.model_name = model_name
        
        # 유효성 검사
        if not ErrorHandler.validate_file_exists(annotation_file, "COCO 주석 파일"):
            raise FileNotFoundError(f"COCO 주석 파일을 찾을 수 없습니다: {annotation_file}")
        if not ErrorHandler.validate_file_exists(visdial_path, "Visual Dialogue 파일"):
            raise FileNotFoundError(f"Visual Dialogue 파일을 찾을 수 없습니다: {visdial_path}")
        
        # COCO 및 데이터 초기화
        self.coco = COCO(annotation_file)
        self.visdial_data = DataProcessor.load_json(visdial_path)
        self.questions = self.visdial_data["data"]["questions"]
        self.answers = self.visdial_data["data"]["answers"] 
        self.dialogs = self.visdial_data["data"]["dialogs"]
        
        # 기존 결과 로드
        self.results, self.processed_ids = DataProcessor.load_existing_results(output_path)
        
        logger.info(f"BaseDataGenerator 초기화 완료")
        logger.info(f"- 총 대화 수: {len(self.dialogs)}")
        logger.info(f"- 기존 처리된 항목: {len(self.processed_ids)}")
        logger.info(f"- 사용 모델: {model_name}")
    
    def _get_image_info(self, image_id: int) -> Optional[Dict]:
        """COCO에서 이미지 정보 가져오기"""
        img_info = self.coco.loadImgs(image_id)
        if not img_info:
            logger.warning(f"이미지 ID {image_id}의 정보를 찾을 수 없습니다")
            return None
        return img_info[0]
    
    def _build_conversation(self, dialog_entry: Dict) -> str:
        """대화 엔트리를 텍스트로 변환"""
        return DialogueFormatter.format_conversation(
            dialog_entry, self.questions, self.answers, include_last_answer=True
        )
    
    def _generate_response(self, conversation: str, image_url: str) -> Optional[str]:
        """Ollama를 사용하여 응답 생성"""
        try:
            # 이미지를 base64로 인코딩
            image_b64 = ImageProcessor.encode_image_url_to_base64(image_url)
            
            # 프롬프트 생성
            system_prompt = PromptBuilder.build_system_prompt("base")
            
            # Ollama 호출
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user", 
                        "content": f"conversation:\n{conversation}",
                        "images": [image_b64]
                    }
                ]
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {e}")
            return None
    
    def generate_data(self, max_samples: Optional[int] = None) -> None:
        """
        데이터 생성 실행
        
        Args:
            max_samples: 처리할 최대 샘플 수 (None이면 전체)
        """
        dialogs_to_process = self.dialogs[:max_samples] if max_samples else self.dialogs
        
        logger.info(f"데이터 생성 시작: {len(dialogs_to_process)}개 대화 처리")
        
        progress_bar = ProgressTracker.create_progress_bar(
            dialogs_to_process, 
            desc="기본 데이터 생성"
        )
        
        successful_count = 0
        failed_count = 0
        
        for dialog in progress_bar:
            image_id = dialog["image_id"]
            
            # 이미 처리된 경우 건너뛰기
            if image_id in self.processed_ids:
                logger.debug(f"이미지 ID {image_id} 이미 처리됨, 건너뛰기")
                continue
            
            # 이미지 정보 가져오기
            img_info = self._get_image_info(image_id)
            if not img_info:
                failed_count += 1
                continue
            
            image_url = img_info.get("coco_url")
            if not image_url:
                logger.warning(f"이미지 ID {image_id}의 URL이 없습니다")
                failed_count += 1
                continue
            
            # 대화 생성
            conversation = self._build_conversation(dialog)
            
            # 응답 생성
            response = self._generate_response(conversation, image_url)
            if not response:
                failed_count += 1
                continue
            
            # 결과 저장
            result_entry = {
                "image_id": image_id,
                "image_url": image_url,
                "conversation": conversation,
                "response": response
            }
            
            self.results.append(result_entry)
            self.processed_ids.add(image_id)
            successful_count += 1
            
            # 주기적으로 파일에 저장
            if successful_count % 10 == 0:
                self._save_results()
                logger.info(f"중간 저장 완료: {successful_count}개 항목")
        
        # 최종 저장
        self._save_results()
        
        logger.info(f"데이터 생성 완료:")
        logger.info(f"- 성공: {successful_count}개")
        logger.info(f"- 실패: {failed_count}개") 
        logger.info(f"- 총 결과: {len(self.results)}개")
    
    def _save_results(self) -> None:
        """결과를 파일에 저장"""
        try:
            DataProcessor.save_json(self.results, self.output_path)
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
    
    def get_statistics(self) -> Dict:
        """생성된 데이터의 통계 정보 반환"""
        if not self.results:
            return {"total": 0, "average_conversation_length": 0}
        
        total = len(self.results)
        avg_conv_length = sum(len(r["conversation"]) for r in self.results) / total
        
        return {
            "total": total,
            "average_conversation_length": avg_conv_length,
            "processed_ids_count": len(self.processed_ids)
        }

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Dialogue 기본 데이터 생성")
    parser.add_argument("--annotation_file", required=True, 
                       help="COCO 주석 파일 경로")
    parser.add_argument("--visdial_path", required=True,
                       help="Visual Dialogue 데이터 파일 경로")
    parser.add_argument("--output_path", required=True,
                       help="출력 파일 경로")
    parser.add_argument("--model_name", default="gemma3:27b",
                       help="사용할 Ollama 모델명")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="처리할 최대 샘플 수")
    
    args = parser.parse_args()
    
    try:
        generator = BaseDataGenerator(
            annotation_file=args.annotation_file,
            visdial_path=args.visdial_path,
            output_path=args.output_path,
            model_name=args.model_name
        )
        
        generator.generate_data(max_samples=args.max_samples)
        
        # 통계 출력
        stats = generator.get_statistics()
        print(f"\n=== 생성 통계 ===")
        print(f"총 생성된 항목: {stats['total']}")
        print(f"평균 대화 길이: {stats['average_conversation_length']:.1f}자")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()