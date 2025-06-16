# gen_api.py
"""
Visual Dialogue API 기반 데이터 생성기
고급 프롬프트 엔지니어링을 활용한 추론 과정 포함 데이터 생성
"""

import os
import json
from typing import Dict, List, Optional
from pycocotools.coco import COCO
import ollama
from .helpers import (
    ImageProcessor, DataProcessor, DialogueFormatter,
    PromptBuilder, ProgressTracker, ErrorHandler, logger
)

class APIDataGenerator:
    """
    Visual Dialogue API 기반 데이터 생성기
    
    추론 과정을 포함한 고품질 응답 데이터를 생성합니다.
    <reasoning> 태그를 활용하여 모델의 사고 과정을 기록합니다.
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
        
        logger.info("APIDataGenerator 초기화 완료")
        logger.info(f"- 총 대화 수: {len(self.dialogs)}")
        logger.info(f"- 기존 처리된 항목: {len(self.processed_ids)}")
        logger.info(f"- 사용 모델: {model_name}")
    
    def _build_advanced_prompt(self) -> str:
        """고급 프롬프트 생성 (추론 과정 포함)"""
        return """
You are given a conversation between a human and an AI, regarding a single image.

Based on the conversation above, write a <reasoning> tag that summarizes the main visual and reasoning steps that took place.
Inside the <reasoning> tag, remind and summarize the previous conversation.

Then, state the final short conclusion that answers the last question.

Respond in this format:
<reasoning>
...reasoning here...
</reasoning>
Final answer sentence.

A few examples:

[conversation history]
Q: is this picture in color
A: yes
Q: is this the only person inside the photo
A: yes  
Q: how old does she look
A: 20s
Q: what color is the light

[Answer]
<reasoning>
single person standing under a traffic signal. After confirming it's in color, only one person, and estimating her age, the conversation returns to the color of the illuminated light. I focus again on the top lamp of the signal, which is shining red.
</reasoning>
The light is red.

[conversation history]
Q: is the plant a tall tree
A: yes
Q: is the zebra eating grass  
A: yes
Q: is it a large zebra
A: yes
Q: are there a lot of trees around
A: no
Q: do you see any mountains
A: no
Q: do you seen any watering holes
A: no
Q: are there any other animals

[Answer]
<reasoning>
We've been walking through a sequence of simple yes/no questions about this scene—tall trees, a zebra eating grass, size, surrounding environment. The user now asks if any other animals appear. Reflecting on the image and our prior focus on the zebra amid sparse foliage, I scan the frame for additional wildlife and find none.
</reasoning>
No.
"""
    
    def _get_image_info(self, image_id: int) -> Optional[Dict]:
        """COCO에서 이미지 정보 가져오기"""
        img_info = self.coco.loadImgs(image_id)
        if not img_info:
            logger.warning(f"이미지 ID {image_id}의 정보를 찾을 수 없습니다")
            return None
        return img_info[0]
    
    def _build_conversation(self, dialog_entry: Dict) -> str:
        """대화 엔트리를 텍스트로 변환 (마지막 답변 제외)"""
        return DialogueFormatter.format_conversation(
            dialog_entry, self.questions, self.answers, include_last_answer=False
        )
    
    def _generate_response(self, conversation: str, image_url: str) -> Optional[str]:
        """Ollama를 사용하여 추론 과정 포함 응답 생성"""
        try:
            # 이미지를 base64로 인코딩
            image_b64 = ImageProcessor.encode_image_url_to_base64(image_url)
            
            # 고급 프롬프트 생성
            system_prompt = self._build_advanced_prompt()
            
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
    
    def _extract_reasoning_and_answer(self, response: str) -> tuple[str, str]:
        """응답에서 추론 과정과 최종 답변 분리"""
        reasoning = ""
        answer = response
        
        if "<reasoning>" in response and "</reasoning>" in response:
            parts = response.split("<reasoning>")
            if len(parts) > 1:
                reasoning_part = parts[1].split("</reasoning>")
                if len(reasoning_part) > 1:
                    reasoning = reasoning_part[0].strip()
                    answer = reasoning_part[1].strip()
        
        return reasoning, answer
    
    def generate_data(self, max_samples: Optional[int] = None) -> None:
        """
        데이터 생성 실행
        
        Args:
            max_samples: 처리할 최대 샘플 수 (None이면 전체)
        """
        dialogs_to_process = self.dialogs[:max_samples] if max_samples else self.dialogs
        
        logger.info(f"API 데이터 생성 시작: {len(dialogs_to_process)}개 대화 처리")
        
        progress_bar = ProgressTracker.create_progress_bar(
            dialogs_to_process,
            desc="API 데이터 생성 (추론 포함)"
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
            
            # 대화 생성 (마지막 답변 제외)
            conversation = self._build_conversation(dialog)
            
            # 응답 생성
            response = self._generate_response(conversation, image_url)
            if not response:
                failed_count += 1
                continue
            
            # 추론과 답변 분리
            reasoning, answer = self._extract_reasoning_and_answer(response)
            
            # 결과 저장
            result_entry = {
                "image_id": image_id,
                "image_url": image_url,
                "conversation": conversation,
                "response": response,
                "reasoning": reasoning,
                "answer": answer
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
        
        logger.info(f"API 데이터 생성 완료:")
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
            return {
                "total": 0,
                "average_conversation_length": 0,
                "reasoning_coverage": 0
            }
        
        total = len(self.results)
        avg_conv_length = sum(len(r["conversation"]) for r in self.results) / total
        reasoning_count = sum(1 for r in self.results if r.get("reasoning", "").strip())
        reasoning_coverage = reasoning_count / total if total > 0 else 0
        
        return {
            "total": total,
            "average_conversation_length": avg_conv_length,
            "reasoning_coverage": reasoning_coverage,
            "processed_ids_count": len(self.processed_ids)
        }
    
    def export_for_training(self, export_path: str) -> None:
        """훈련용 형식으로 데이터 내보내기"""
        training_data = []
        
        for result in self.results:
            if result.get("answer"):  # 답변이 있는 경우만
                training_entry = {
                    "image_id": result["image_id"],
                    "image_url": result["image_url"],
                    "conversation": result["conversation"],
                    "response": result["answer"]  # 추론 과정 제외하고 답변만
                }
                training_data.append(training_entry)
        
        DataProcessor.save_json(training_data, export_path)
        logger.info(f"훈련용 데이터 {len(training_data)}개 내보내기 완료: {export_path}")

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Dialogue API 데이터 생성")
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
    parser.add_argument("--export_training", type=str, default=None,
                       help="훈련용 데이터 내보내기 경로")
    
    args = parser.parse_args()
    
    try:
        generator = APIDataGenerator(
            annotation_file=args.annotation_file,
            visdial_path=args.visdial_path,
            output_path=args.output_path,
            model_name=args.model_name
        )
        
        generator.generate_data(max_samples=args.max_samples)
        
        # 훈련용 데이터 내보내기
        if args.export_training:
            generator.export_for_training(args.export_training)
        
        # 통계 출력
        stats = generator.get_statistics()
        print(f"\n=== 생성 통계 ===")
        print(f"총 생성된 항목: {stats['total']}")
        print(f"평균 대화 길이: {stats['average_conversation_length']:.1f}자")
        print(f"추론 과정 포함률: {stats['reasoning_coverage']:.1%}")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()