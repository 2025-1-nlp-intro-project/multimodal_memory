#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visual Dialogue 생성 결과 평가를 위한 BERTScore 기반 평가 시스템

이 모듈은 Visual Dialogue 생성 결과를 BERTScore를 중심으로 평가하는 포괄적인 시스템을 제공합니다.
BERTScore는 BERT의 사전 훈련된 컨텍스추얼 임베딩을 활용하여 텍스트 간의 의미적 유사성을 측정하는
혁신적인 평가 지표로, 기존의 n-gram 기반 메트릭(BLEU, ROUGE)보다 의미적 유사성을 더 잘 포착합니다.

주요 기능:
- BERTScore를 주요 메트릭으로 한 평가 (Precision, Recall, F1)
- ROUGE, BLEU를 보조 메트릭으로 제공
- 상세한 성능 분석 및 품질 분포 제공
- 효율적인 배치 처리 및 모델 캐싱
- 직관적인 명령줄 인터페이스

사용법:
    python bert_evaluate.py --generated predictions.json --ground_truth visdial_1.0_val.json

작성자: Visual Dialogue 프로젝트 팀
라이센스: MIT
"""

import json
import re
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from bert_score import score as bert_score_function, BERTScorer
from rouge_score import rouge_scorer
from sacrebleu import corpus_bleu


@dataclass
class EvaluationConfig:
    """평가 설정을 위한 데이터 클래스"""
    bert_model: str = "bert-base-uncased"  # BERT 모델 타입
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lang: str = "en"  # 언어 설정
    rescale_with_baseline: bool = True  # 베이스라인 재스케일링 적용
    idf_weighting: bool = True  # IDF 가중치 적용
    batch_size: int = 64  # 배치 크기
    max_length: int = 512  # 최대 토큰 길이
    verbose: bool = True  # 상세 출력
    cache_dir: Optional[str] = None  # BERT 모델 캐시 디렉토리
    use_fast_tokenizer: bool = True  # 빠른 토크나이저 사용


class VisualDialogEvaluator:
    """
    Visual Dialogue 평가를 위한 BERTScore 중심 평가 시스템

    이 클래스는 다음과 같은 메트릭을 제공합니다:
    1. BERTScore (Precision, Recall, F1) - 주요 메트릭
    2. ROUGE (보조 메트릭)
    3. BLEU (보조 메트릭)  
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        평가자 초기화

        Args:
            config: 평가 설정 (None인 경우 기본값 사용)
        """
        self.config = config or EvaluationConfig()
        self.logger = self._setup_logger()

        # BERTScorer 초기화 (캐싱을 위해)
        self.logger.info(f"BERTScorer 초기화 ({self.config.bert_model}, {self.config.device})")
        try:
            self.bert_scorer = BERTScorer(
                model_type=self.config.bert_model,
                lang=self.config.lang,
                rescale_with_baseline=self.config.rescale_with_baseline,
                idf=self.config.idf_weighting,
                device=self.config.device,
                batch_size=self.config.batch_size,
                nthreads=4,  # 병렬 처리 스레드 수
                use_fast_tokenizer=self.config.use_fast_tokenizer
            )
            self.logger.info("BERTScorer 초기화 완료")
        except Exception as e:
            self.logger.error(f"BERTScorer 초기화 실패: {e}")
            raise

        # ROUGE 스코어러 초기화
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'], 
            use_stemmer=True
        )

        self.logger.info(f"Visual Dialog Evaluator 초기화 완료")
        self.logger.info(f"BERT Model: {self.config.bert_model}")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"Language: {self.config.lang}")

    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger("VisualDialogEvaluator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _clean_response(self, response: str) -> str:
        """
        응답 텍스트를 정리하는 헬퍼 함수

        Args:
            response: 원본 응답 텍스트

        Returns:
            정리된 응답 텍스트
        """
        # 1. reasoning 태그 제거
        reasoning_pattern = re.compile(r'<reasoning>.*?</reasoning>', re.DOTALL | re.IGNORECASE)
        cleaned = reasoning_pattern.sub('', response).strip()

        # 2. 기타 XML 태그 제거
        tag_pattern = re.compile(r'<.*?>', re.DOTALL)
        cleaned = tag_pattern.sub('', cleaned).strip()

        # 3. 불필요한 공백 정리
        cleaned = ' '.join(cleaned.split())

        return cleaned

    def load_generated_answers(self, generated_path: str, max_samples: Optional[int] = None) -> Dict[int, str]:
        """
        생성된 답변 로드 및 전처리

        Args:
            generated_path: 생성된 답변 JSON 파일 경로
            max_samples: 최대 샘플 수 (None인 경우 전체)

        Returns:
            image_id -> cleaned_response 매핑
        """
        self.logger.info(f"생성된 답변 로드 중: {generated_path}")

        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"생성된 답변 파일을 찾을 수 없습니다: {generated_path}")

        try:
            with open(generated_path, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"JSON 파싱 오류: {generated_path}")
            raise

        if max_samples:
            gen_data = gen_data[:max_samples]

        gen_answers = {}

        for entry in tqdm(gen_data, desc="생성된 답변 처리"):
            img_id = entry.get("image_id")
            if img_id is None:
                continue

            response = entry.get("response", "").strip()

            # 응답 정리
            cleaned_response = self._clean_response(response)

            if cleaned_response:
                gen_answers[img_id] = cleaned_response

        self.logger.info(f"총 {len(gen_answers)}개의 생성된 답변 로드 완료")
        return gen_answers

    def load_ground_truth_answers(self, raw_path: str) -> Dict[int, str]:
        """
        VisDial 데이터셋에서 정답 로드

        Args:
            raw_path: VisDial 원본 JSON 파일 경로

        Returns:
            image_id -> ground_truth_answer 매핑
        """
        self.logger.info(f"정답 데이터 로드 중: {raw_path}")

        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"정답 파일을 찾을 수 없습니다: {raw_path}")

        try:
            with open(raw_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            self.logger.error(f"JSON 파싱 오류: {raw_path}")
            raise

        questions = raw_data["data"]["questions"]
        answers = raw_data["data"]["answers"]

        gt_answers = {}

        for dialog in tqdm(raw_data["data"]["dialogs"], desc="정답 처리"):
            img_id = dialog["image_id"]

            if not dialog["dialog"]:
                continue

            last_turn = dialog["dialog"][-1]
            gt_text = ""

            # 직접 답변이 있는 경우
            if "answer" in last_turn:
                answer_id = last_turn["answer"]
                if 0 <= answer_id < len(answers):
                    gt_text = answers[answer_id]

            # 답변 옵션에서 선택하는 경우
            elif "answer_options" in last_turn and "gt_index" in last_turn:
                opt_list = last_turn["answer_options"]
                gt_idx = last_turn["gt_index"]
                if 0 <= gt_idx < len(opt_list) and 0 <= opt_list[gt_idx] < len(answers):
                    gt_text = answers[opt_list[gt_idx]]

            # 정리된 정답 저장
            if gt_text.strip():
                gt_answers[img_id] = self._clean_response(gt_text)

        self.logger.info(f"총 {len(gt_answers)}개의 정답 로드 완료")
        return gt_answers

    def compute_bertscore(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """
        BERTScore 계산 (주요 메트릭)

        Args:
            candidates: 생성된 텍스트 리스트
            references: 참조 텍스트 리스트

        Returns:
            BERTScore 결과 (precision, recall, f1)
        """
        self.logger.info("BERTScore 계산 중...")

        try:
            # BERTScorer 사용 (캐싱된 모델)
            P, R, F1 = bert_score_function(candidates, [r[0] for r in references],
                                      lang="en", device='cuda', rescale_with_baseline=False)

            # 텐서를 numpy로 변환 후 평균 계산
            precision = P.mean().item()
            recall = R.mean().item()
            f1 = F1.mean().item()

            self.logger.info(f"BERTScore 계산 완료: P={precision:.4f}, R={recall:.4f}, F1={f1:.4f}")

            return {
                "bert_precision": precision,
                "bert_recall": recall,
                "bert_f1": f1,
                "bert_f1_std": F1.std().item(),  # 표준편차 추가
            }
        except Exception as e:
            self.logger.error(f"BERTScore 계산 오류: {e}")
            raise

    def compute_auxiliary_metrics(self, candidates: List[str], references: List[str]) -> Dict[str, float]:
        """
        보조 메트릭 계산 (ROUGE, BLEU)

        Args:
            candidates: 생성된 텍스트 리스트
            references: 참조 텍스트 리스트

        Returns:
            보조 메트릭 결과
        """
        self.logger.info("보조 메트릭 계산 중...")

        # ROUGE 점수 누적
        rouge_scores = {
            "rouge1": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rouge2": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0},
            "rougeL": {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
        }

        total_pairs = len(candidates)

        # 각 쌍에 대해 ROUGE 계산
        for cand, ref in tqdm(zip(candidates, references), 
                             total=total_pairs, desc="ROUGE 계산"):
            try:
                # ROUGE 계산
                rouge_result = self.rouge_scorer.score(ref, cand)
                for metric in rouge_scores:
                    for sub_metric in rouge_scores[metric]:
                        rouge_scores[metric][sub_metric] += getattr(rouge_result[metric], sub_metric)
            except Exception as e:
                self.logger.warning(f"일부 메트릭 계산 중 오류 발생, 건너뜀: {e}")
                continue

        # ROUGE 평균 계산
        for metric in rouge_scores:
            for sub_metric in rouge_scores[metric]:
                if total_pairs > 0:
                    rouge_scores[metric][sub_metric] /= total_pairs
                else:
                    rouge_scores[metric][sub_metric] = 0.0

        # BLEU 점수 계산 (corpus-level)
        try:
            bleu_result = corpus_bleu(candidates, [[ref] for ref in references])
            bleu_score = bleu_result.score / 100.0  # 0-1 범위로 정규화
        except Exception as e:
            self.logger.warning(f"BLEU 계산 중 오류 발생: {e}")
            bleu_score = 0.0

        self.logger.info(f"보조 메트릭 계산 완료")
        self.logger.info(f"ROUGE-1 F1: {rouge_scores['rouge1']['fmeasure']:.4f}")
        self.logger.info(f"ROUGE-L F1: {rouge_scores['rougeL']['fmeasure']:.4f}")
        self.logger.info(f"BLEU: {bleu_score:.4f}")

        return {
            "rouge1_f1": rouge_scores["rouge1"]["fmeasure"],
            "rouge2_f1": rouge_scores["rouge2"]["fmeasure"],
            "rougeL_f1": rouge_scores["rougeL"]["fmeasure"],
            "bleu": bleu_score,
            "total_samples": total_pairs
        }

    def compute_detailed_bertscore_analysis(self, candidates: List[str], references: List[str]) -> Dict[str, Any]:
        """
        BERTScore 상세 분석 수행

        Args:
            candidates: 생성된 텍스트 리스트
            references: 참조 텍스트 리스트

        Returns:
            상세 분석 결과
        """
        self.logger.info("BERTScore 상세 분석 수행 중...")

        try:
            P, R, F1 = self.bert_scorer.score(candidates, references)

            # 개별 점수를 numpy 배열로 변환
            p_scores = P.cpu().numpy()
            r_scores = R.cpu().numpy()
            f1_scores = F1.cpu().numpy()

            # 분포 분석
            analysis = {
                "score_distribution": {
                    "precision": {
                        "mean": float(np.mean(p_scores)),
                        "std": float(np.std(p_scores)),
                        "min": float(np.min(p_scores)),
                        "max": float(np.max(p_scores)),
                        "median": float(np.median(p_scores)),
                        "q25": float(np.percentile(p_scores, 25)),
                        "q75": float(np.percentile(p_scores, 75))
                    },
                    "recall": {
                        "mean": float(np.mean(r_scores)),
                        "std": float(np.std(r_scores)),
                        "min": float(np.min(r_scores)),
                        "max": float(np.max(r_scores)),
                        "median": float(np.median(r_scores)),
                        "q25": float(np.percentile(r_scores, 25)),
                        "q75": float(np.percentile(r_scores, 75))
                    },
                    "f1": {
                        "mean": float(np.mean(f1_scores)),
                        "std": float(np.std(f1_scores)),
                        "min": float(np.min(f1_scores)),
                        "max": float(np.max(f1_scores)),
                        "median": float(np.median(f1_scores)),
                        "q25": float(np.percentile(f1_scores, 25)),
                        "q75": float(np.percentile(f1_scores, 75))
                    }
                },
                "performance_brackets": {
                    "high_quality": int(np.sum(f1_scores >= 0.8)),
                    "medium_quality": int(np.sum((f1_scores >= 0.6) & (f1_scores < 0.8))),
                    "low_quality": int(np.sum(f1_scores < 0.6))
                }
            }

            # 최고/최저 점수 샘플 찾기 (인덱스 및 점수)
            top_idx = np.argmax(f1_scores)
            bottom_idx = np.argmin(f1_scores)

            analysis["examples"] = {
                "highest_score": {
                    "f1": float(f1_scores[top_idx]),
                    "candidate": candidates[top_idx],
                    "reference": references[top_idx]
                },
                "lowest_score": {
                    "f1": float(f1_scores[bottom_idx]),
                    "candidate": candidates[bottom_idx],
                    "reference": references[bottom_idx]
                }
            }

            return analysis
        except Exception as e:
            self.logger.error(f"상세 분석 중 오류 발생: {e}")
            # 최소한의 분석 결과 반환
            return {
                "score_distribution": {},
                "performance_brackets": {
                    "high_quality": 0,
                    "medium_quality": 0,
                    "low_quality": 0
                },
                "examples": {}
            }

    def evaluate(self, generated_path: str, ground_truth_path: str, 
                max_samples: Optional[int] = None, 
                output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        전체 평가 수행

        Args:
            generated_path: 생성된 답변 파일 경로
            ground_truth_path: 정답 파일 경로
            max_samples: 최대 평가 샘플 수
            output_path: 결과 저장 경로 (None인 경우 저장하지 않음)

        Returns:
            평가 결과 딕셔너리
        """
        self.logger.info("=== Visual Dialogue BERTScore 평가 시작 ===")

        # 데이터 로드
        gen_answers = self.load_generated_answers(generated_path, max_samples)
        gt_answers = self.load_ground_truth_answers(ground_truth_path)

        # 공통 이미지 ID 찾기
        common_ids = set(gen_answers.keys()) & set(gt_answers.keys())
        self.logger.info(f"평가 가능한 공통 샘플 수: {len(common_ids)}")

        if not common_ids:
            raise ValueError("평가 가능한 공통 샘플이 없습니다.")

        # 정렬된 리스트로 변환
        sorted_ids = sorted(common_ids)
        candidates = [gen_answers[img_id] for img_id in sorted_ids]
        references = [gt_answers[img_id] for img_id in sorted_ids]

        # BERTScore 계산 (주요 메트릭)
        bert_results = self.compute_bertscore(candidates, references)

        # 보조 메트릭 계산
        auxiliary_results = self.compute_auxiliary_metrics(candidates, references)

        # BERTScore 상세 분석
        detailed_analysis = self.compute_detailed_bertscore_analysis(candidates, references)

        # 결과 통합
        evaluation_results = {
            "evaluation_info": {
                "generated_file": generated_path,
                "ground_truth_file": ground_truth_path,
                "total_samples": len(common_ids),
                "bert_model": self.config.bert_model,
                "evaluation_config": {
                    "rescale_with_baseline": self.config.rescale_with_baseline,
                    "idf_weighting": self.config.idf_weighting,
                    "language": self.config.lang
                }
            },
            "primary_metrics": {
                "bertscore": bert_results
            },
            "auxiliary_metrics": auxiliary_results,
            "detailed_analysis": detailed_analysis,
            "sample_comparisons": self._generate_sample_comparisons(
                candidates[:5], references[:5], sorted_ids[:5]
            )
        }

        # 결과 저장
        if output_path:
            self._save_results(evaluation_results, output_path)

        # 결과 요약 출력
        self._print_summary(evaluation_results)

        return evaluation_results

    def _generate_sample_comparisons(self, candidates: List[str], references: List[str], 
                                   image_ids: List[int]) -> List[Dict]:
        """샘플 비교 결과 생성"""
        comparisons = []

        # 개별 BERTScore 계산
        P, R, F1 = bert_score_function(candidates, [r[0] for r in references],
                                      lang="en", device='cuda', rescale_with_baseline=False)

        for i, (cand, ref, img_id) in enumerate(zip(candidates, references, image_ids)):
            comparisons.append({
                "image_id": img_id,
                "candidate": cand,
                "reference": ref,
                "bertscore": {
                    "precision": P[i].item(),
                    "recall": R[i].item(),
                    "f1": F1[i].item()
                }
            })

        return comparisons

    def _save_results(self, results: Dict[str, Any], output_path: str):
        """결과를 JSON 파일로 저장"""
        self.logger.info(f"평가 결과 저장 중: {output_path}")

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            self.logger.info("평가 결과 저장 완료")
        except Exception as e:
            self.logger.error(f"결과 저장 중 오류 발생: {e}")
            raise

    def _print_summary(self, results: Dict[str, Any]):
        """평가 결과 요약 출력"""
        print("" + "="*60)
        print("           VISUAL DIALOGUE EVALUATION RESULTS")
        print("="*60)

        # 주요 메트릭 (BERTScore)
        bert_scores = results["primary_metrics"]["bertscore"]
        print(f"🎯 PRIMARY METRICS (BERTScore)")
        print(f"   Precision: {bert_scores['bert_precision']:.4f}")
        print(f"   Recall:    {bert_scores['bert_recall']:.4f}")
        print(f"   F1-Score:  {bert_scores['bert_f1']:.4f} (±{bert_scores['bert_f1_std']:.4f})")

        # 보조 메트릭
        aux_metrics = results["auxiliary_metrics"]
        print(f"📊 AUXILIARY METRICS")
        print(f"   ROUGE-1:   {aux_metrics['rouge1_f1']:.4f}")
        print(f"   ROUGE-2:   {aux_metrics['rouge2_f1']:.4f}")
        print(f"   ROUGE-L:   {aux_metrics['rougeL_f1']:.4f}")
        print(f"   BLEU:      {aux_metrics['bleu']:.4f}")

        # 품질 분포
        brackets = results["detailed_analysis"]["performance_brackets"]
        total = sum(brackets.values())
        if total > 0:
            print(f"📈 QUALITY DISTRIBUTION")
            print(f"   High (F1≥0.8):   {brackets['high_quality']:4d} ({brackets['high_quality']/total*100:.1f}%)")
            print(f"   Medium (0.6≤F1<0.8): {brackets['medium_quality']:4d} ({brackets['medium_quality']/total*100:.1f}%)")
            print(f"   Low (F1<0.6):    {brackets['low_quality']:4d} ({brackets['low_quality']/total*100:.1f}%)")

        print(f"   Total Samples: {results['evaluation_info']['total_samples']}")
        print("="*60 + "")


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Visual Dialogue BERTScore 평가")
    parser.add_argument("--generated", "-g", required=True, 
                       help="생성된 답변 JSON 파일 경로")
    parser.add_argument("--ground_truth", "-t", required=True,
                       help="VisDial 정답 JSON 파일 경로")
    parser.add_argument("--output", "-o", default="evaluation_results.json",
                       help="결과 저장 파일 경로")
    parser.add_argument("--max_samples", "-m", type=int,
                       help="최대 평가 샘플 수")
    parser.add_argument("--bert_model", "-b", default="bert-base-uncased",
                       help="사용할 BERT 모델")
    parser.add_argument("--device", "-d", default="auto",
                       help="연산 디바이스 (cuda/cpu/auto)")
    parser.add_argument("--no_baseline", action="store_true",
                       help="베이스라인 재스케일링 비활성화")
    parser.add_argument("--no_idf", action="store_true",
                       help="IDF 가중치 비활성화")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="배치 크기")

    args = parser.parse_args()

    # 디바이스 설정
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # 설정 생성
    config = EvaluationConfig(
        bert_model=args.bert_model,
        device=device,
        rescale_with_baseline=not args.no_baseline,
        idf_weighting=not args.no_idf,
        batch_size=args.batch_size
    )

    # 평가 실행
    evaluator = VisualDialogEvaluator(config)

    try:
        results = evaluator.evaluate(
            generated_path=args.generated,
            ground_truth_path=args.ground_truth,
            max_samples=args.max_samples,
            output_path=args.output
        )

        print(f"✅ 평가 완료! 결과가 {args.output}에 저장되었습니다.")

    except Exception as e:
        print(f"❌ 평가 중 오류 발생: {e}")
        raise


if __name__ == "__main__":
    main()
