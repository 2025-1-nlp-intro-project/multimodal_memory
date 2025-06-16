# gen_eval.py
"""
Visual Dialogue 평가 데이터 생성기
검증 세트를 활용한 평가용 데이터 생성 및 처리
"""

import os
import json
from typing import Dict, List, Optional
import ollama
from ..utils.helpers import (
    ImageProcessor, DataProcessor, DialogueFormatter,
    PromptBuilder, ProgressTracker, ErrorHandler, PathManager, logger
)

class EvalDataGenerator:
    """
    Visual Dialogue 평가 데이터 생성기
    
    검증 데이터셋을 활용하여 모델 평가를 위한 응답 데이터를 생성합니다.
    로컬 이미지 파일을 직접 사용하여 보다 안정적인 처리를 제공합니다.
    """
    
    def __init__(self,
                 visdial_path: str,
                 image_dir: str,
                 output_path: str,
                 model_name: str = "gemma3:27b"):
        """
        초기화
        
        Args:
            visdial_path: Visual Dialogue 검증 데이터 파일 경로
            image_dir: 이미지 디렉토리 경로
            output_path: 출력 파일 경로
            model_name: 사용할 Ollama 모델명
        """
        self.visdial_path = visdial_path
        self.image_dir = image_dir
        self.output_path = output_path
        self.model_name = model_name
        
        # 유효성 검사
        if not ErrorHandler.validate_file_exists(visdial_path, "Visual Dialogue 파일"):
            raise FileNotFoundError(f"Visual Dialogue 파일을 찾을 수 없습니다: {visdial_path}")
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"이미지 디렉토리를 찾을 수 없습니다: {image_dir}")
        
        # 데이터 초기화
        self.visdial_data = DataProcessor.load_json(visdial_path)
        self.questions = self.visdial_data["data"]["questions"]
        self.answers = self.visdial_data["data"]["answers"]
        self.dialogs = self.visdial_data["data"]["dialogs"]
        
        # 기존 결과 로드
        self.results, self.processed_ids = DataProcessor.load_existing_results(output_path)
        
        logger.info("EvalDataGenerator 초기화 완료")
        logger.info(f"- 총 대화 수: {len(self.dialogs)}")
        logger.info(f"- 기존 처리된 항목: {len(self.processed_ids)}")
        logger.info(f"- 사용 모델: {model_name}")
        logger.info(f"- 이미지 디렉토리: {image_dir}")
    
    def _build_evaluation_prompt(self) -> str:
        """평가용 프롬프트 생성"""
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
    
    def _get_image_path(self, image_id: int) -> str:
        """이미지 ID로부터 이미지 파일 경로 생성"""
        return PathManager.get_image_path(image_id, self.image_dir, "val2018")
    
    def _build_conversation(self, dialog_entry: Dict) -> str:
        """대화 엔트리를 텍스트로 변환 (마지막 답변 제외)"""
        return DialogueFormatter.format_conversation(
            dialog_entry, self.questions, self.answers, include_last_answer=False
        )
    
    def _generate_response(self, conversation: str, image_path: str) -> Optional[str]:
        """Ollama를 사용하여 평가용 응답 생성"""
        try:
            # 프롬프트 생성
            system_prompt = self._build_evaluation_prompt()
            
            # Ollama 호출 (로컬 이미지 파일 사용)
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"conversation:\n{conversation}",
                        "images": [image_path]
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
        평가 데이터 생성 실행
        
        Args:
            max_samples: 처리할 최대 샘플 수 (None이면 전체)
        """
        dialogs_to_process = self.dialogs[:max_samples] if max_samples else self.dialogs
        
        logger.info(f"평가 데이터 생성 시작: {len(dialogs_to_process)}개 대화 처리")
        
        progress_bar = ProgressTracker.create_progress_bar(
            dialogs_to_process,
            desc="평가 데이터 생성"
        )
        
        successful_count = 0
        failed_count = 0
        missing_images = 0
        
        for dialog in progress_bar:
            image_id = dialog["image_id"]
            
            # 이미 처리된 경우 건너뛰기
            if image_id in self.processed_ids:
                logger.debug(f"이미지 ID {image_id} 이미 처리됨, 건너뛰기")
                continue
            
            # 이미지 파일 경로 생성
            image_path = self._get_image_path(image_id)
            
            # 이미지 파일 존재 확인
            if not os.path.exists(image_path):
                logger.warning(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                missing_images += 1
                continue
            
            # 대화 생성 (마지막 답변 제외)
            conversation = self._build_conversation(dialog)
            
            # 응답 생성
            response = self._generate_response(conversation, image_path)
            if not response:
                failed_count += 1
                continue
            
            # 추론과 답변 분리
            reasoning, answer = self._extract_reasoning_and_answer(response)
            
            # 결과 저장
            result_entry = {
                "image_id": image_id,
                "image_path": image_path,
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
        
        logger.info(f"평가 데이터 생성 완료:")
        logger.info(f"- 성공: {successful_count}개")
        logger.info(f"- 실패: {failed_count}개")
        logger.info(f"- 누락 이미지: {missing_images}개")
        logger.info(f"- 총 결과: {len(self.results)}개")
    
    def _save_results(self) -> None:
        """결과를 파일에 저장"""
        try:
            DataProcessor.save_json(self.results, self.output_path)
        except Exception as e:
            logger.error(f"결과 저장 중 오류 발생: {e}")
    
    def prepare_for_evaluation(self) -> Dict[str, List]:
        """평가를 위한 데이터 준비"""
        evaluation_data = {
            "predictions": [],
            "ground_truth": []
        }
        
        for result in self.results:
            if result.get("answer"):
                prediction_entry = {
                    "image_id": result["image_id"],
                    "predicted_answer": result["answer"],
                    "reasoning": result.get("reasoning", "")
                }
                evaluation_data["predictions"].append(prediction_entry)
        
        return evaluation_data
    
    def export_predictions(self, predictions_path: str) -> None:
        """예측 결과를 별도 파일로 내보내기"""
        predictions = []
        
        for result in self.results:
            if result.get("answer"):
                prediction = {
                    "image_id": result["image_id"],
                    "answer": result["answer"]
                }
                predictions.append(prediction)
        
        DataProcessor.save_json(predictions, predictions_path)
        logger.info(f"예측 결과 {len(predictions)}개 내보내기 완료: {predictions_path}")
    
    def get_statistics(self) -> Dict:
        """생성된 데이터의 통계 정보 반환"""
        if not self.results:
            return {
                "total": 0,
                "average_conversation_length": 0,
                "reasoning_coverage": 0,
                "average_reasoning_length": 0
            }
        
        total = len(self.results)
        avg_conv_length = sum(len(r["conversation"]) for r in self.results) / total
        
        reasoning_items = [r for r in self.results if r.get("reasoning", "").strip()]
        reasoning_coverage = len(reasoning_items) / total if total > 0 else 0
        avg_reasoning_length = sum(len(r["reasoning"]) for r in reasoning_items) / len(reasoning_items) if reasoning_items else 0
        
        return {
            "total": total,
            "average_conversation_length": avg_conv_length,
            "reasoning_coverage": reasoning_coverage,
            "average_reasoning_length": avg_reasoning_length,
            "processed_ids_count": len(self.processed_ids)
        }

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Dialogue 평가 데이터 생성")
    parser.add_argument("--visdial_path", required=True,
                       help="Visual Dialogue 검증 데이터 파일 경로")
    parser.add_argument("--image_dir", required=True,
                       help="이미지 디렉토리 경로")
    parser.add_argument("--output_path", required=True,
                       help="출력 파일 경로")
    parser.add_argument("--model_name", default="gemma3:27b",
                       help="사용할 Ollama 모델명")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="처리할 최대 샘플 수")
    parser.add_argument("--export_predictions", type=str, default=None,
                       help="예측 결과 내보내기 경로")
    
    args = parser.parse_args()
    
    try:
        generator = EvalDataGenerator(
            visdial_path=args.visdial_path,
            image_dir=args.image_dir,
            output_path=args.output_path,
            model_name=args.model_name
        )
        
        generator.generate_data(max_samples=args.max_samples)
        
        # 예측 결과 내보내기
        if args.export_predictions:
            generator.export_predictions(args.export_predictions)
        
        # 통계 출력
        stats = generator.get_statistics()
        print(f"\n=== 생성 통계 ===")
        print(f"총 생성된 항목: {stats['total']}")
        print(f"평균 대화 길이: {stats['average_conversation_length']:.1f}자")
        print(f"추론 과정 포함률: {stats['reasoning_coverage']:.1%}")
        print(f"평균 추론 길이: {stats['average_reasoning_length']:.1f}자")
        
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()