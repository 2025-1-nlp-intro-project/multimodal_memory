#!/usr/bin/env python3

import json
import re
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from pathlib import Path

# VisualDialogueEvaluator 클래스는 Visual Dialogue 데이터셋 기반의 모델 평가를 위한 클래스입니다.
class VisualDialogueEvaluator:
    
    def __init__(self, visdial_data_path: str):
        """
        Visual Dialogue 데이터셋을 불러와서 평가에 필요한 정보를 초기화합니다.
        
        Args:
            visdial_data_path: VisDial 데이터셋 JSON 파일 경로
        """
        self.visdial_data_path = visdial_data_path
        self.visdial_data = self._load_visdial_data()
        self.answers_list = self.visdial_data["data"]["answers"]
        self.img_candidates = self._build_candidate_mapping()
    
    def _load_visdial_data(self) -> Dict:
        """
        Visual Dialogue 데이터셋을 JSON 파일에서 불러옵니다.
        """
        with open(self.visdial_data_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _build_candidate_mapping(self) -> Dict[int, List[str]]:
        """
        각 이미지 ID별로 정답 후보 리스트를 매핑합니다.
        """
        img_candidates = {}
        
        for dialog in self.visdial_data["data"]["dialogs"]:
            img_id = dialog["image_id"]
            # 각 turn에서 answer_options가 있는 부분을 찾음
            for turn in dialog["dialog"]:
                if "answer_options" in turn:
                    # 후보 인덱스를 실제 정답 텍스트로 변환
                    opts = [self.answers_list[idx] for idx in turn["answer_options"]]
                    img_candidates[img_id] = opts
                    break
        
        return img_candidates
    
    def normalize_text(self, text: str) -> str:
        """
        텍스트를 소문자 및 특수문자 제거 등으로 정규화합니다.
        """
        text = text.lower().strip()
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def extract_answer_from_response(self, response: str) -> str:
        """
        모델 응답에서 실제 정답 부분만 추출합니다.
        <reasoning> 태그가 있으면 그 뒤를, 없으면 마지막 문장 또는 전체를 반환합니다.
        """
        # Try to extract answer after reasoning tag
        if '<reasoning>' in response and '</reasoning>' in response:
            parts = response.split('</reasoning>')
            if len(parts) > 1:
                return parts[-1].strip()
        
        # If no reasoning tag, return the last sentence or the whole response
        sentences = response.split('.')
        if len(sentences) > 1:
            return sentences[-2].strip() if sentences[-1].strip() == '' else sentences[-1].strip()
        
        return response.strip()
    
    def mean_reciprocal_rank(self, relevance_lists: List[List[int]]) -> float:
        """
        Mean Reciprocal Rank(MRR) 지표를 계산합니다.
        """
        rr_scores = []
        for r in relevance_lists:
            try:
                idx = r.index(1)
                rr_scores.append(1.0 / (idx + 1))
            except ValueError:
                rr_scores.append(0.0)
        return np.mean(rr_scores)
    
    def recall_at_k(self, relevance_lists: List[List[int]], k: int) -> float:
        """
        Recall@k 지표를 계산합니다. 상위 k개 중 정답 포함 여부를 평가합니다.
        """
        hits = [1 if any(r[:k]) else 0 for r in relevance_lists]
        return np.mean(hits)
    
    def mean_rank(self, relevance_lists: List[List[int]]) -> float:
        """
        Mean Rank(평균 순위) 지표를 계산합니다.
        """
        ranks = []
        for r in relevance_lists:
            try:
                ranks.append(r.index(1) + 1)
            except ValueError:
                ranks.append(len(r) + 1)
        return np.mean(ranks)
    
    def dcg_at_k(self, relevance: List[int], k: int) -> float:
        """
        DCG(Discounted Cumulative Gain)@k 지표를 계산합니다.
        """
        relevance = np.asarray(relevance)[:k]
        if relevance.size:
            gains = (2**relevance - 1)
            discounts = np.log2(np.arange(2, relevance.size + 2))
            return np.sum(gains / discounts)
        return 0.0
    
    def ndcg_at_k(self, relevance_lists: List[List[int]], k: int) -> float:
        """
        NDCG(Normalized Discounted Cumulative Gain)@k 지표를 계산합니다.
        """
        ndcgs = []
        for r in relevance_lists:
            dcg = self.dcg_at_k(r, k)
            ideal = sorted(r, reverse=True)
            idcg = self.dcg_at_k(ideal, k)
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        return np.mean(ndcgs)
    
    def build_relevance_lists(self, predictions: List[Dict], 
                            use_partial_match: bool = True) -> List[List[int]]:
        """
        예측 결과와 후보 정답 리스트를 비교하여 relevance list(정답 여부 리스트)를 만듭니다.
        use_partial_match가 True면 부분 일치도 허용합니다.
        """
        relevance_lists = []
        failed_matches = 0
        
        for pred in predictions:
            img_id = pred["image_id"]
            raw_response = pred["response"]
            
            # Extract answer from response
            predicted_answer = self.extract_answer_from_response(raw_response)
            normalized_pred = self.normalize_text(predicted_answer)
            
            # Get candidates for this image
            candidates = self.img_candidates.get(img_id, [])
            if not candidates:
                logging.warning(f"No candidates found for image {img_id}")
                continue
            
            # Build relevance list
            relevance_list = []
            match_found = False
            
            for candidate in candidates:
                normalized_cand = self.normalize_text(candidate)
                
                if use_partial_match:
                    # Allow partial matching
                    if (normalized_cand == normalized_pred or 
                        normalized_cand in normalized_pred or 
                        normalized_pred in normalized_cand):
                        relevance_list.append(1)
                        match_found = True
                    else:
                        relevance_list.append(0)
                else:
                    # Exact match only
                    if normalized_cand == normalized_pred:
                        relevance_list.append(1)
                        match_found = True
                    else:
                        relevance_list.append(0)
            
            if not match_found:
                failed_matches += 1
                if failed_matches <= 5:  # Log first 5 failures for debugging
                    logging.info(f"No match found for image {img_id}")
                    logging.info(f"  Predicted: '{predicted_answer}' -> '{normalized_pred}'")
                    logging.info(f"  Candidates: {candidates[:3]}...")  # Show first 3 candidates
            
            relevance_lists.append(relevance_list)
        
        if failed_matches > 0:
            logging.warning(f"Total failed matches: {failed_matches}/{len(predictions)} "
                          f"({failed_matches/len(predictions)*100:.1f}%)")
        
        return relevance_lists
    
    def evaluate_predictions(self, predictions: List[Dict], 
                           use_partial_match: bool = True) -> Dict[str, float]:
        """
        예측 결과에 대해 주요 평가 지표(MRR, Recall@k, NDCG 등)를 계산합니다.
        """
        logging.info(f"Evaluating {len(predictions)} predictions...")
        
        # Build relevance lists
        relevance_lists = self.build_relevance_lists(predictions, use_partial_match)
        
        if not relevance_lists:
            raise ValueError("No valid relevance lists generated. Check your predictions format.")
        
        # Calculate metrics
        results = {
            'mrr': self.mean_reciprocal_rank(relevance_lists),
            'r1': self.recall_at_k(relevance_lists, 1),
            'r5': self.recall_at_k(relevance_lists, 5),
            'r10': self.recall_at_k(relevance_lists, 10),
            'mean_rank': self.mean_rank(relevance_lists),
            'ndcg': self.ndcg_at_k(relevance_lists, len(relevance_lists[0]) if relevance_lists else 100)
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, float]):
        """
        평가 결과를 보기 좋게 출력합니다.
        """
        print("\n" + "="*50)
        print("Visual Dialogue Evaluation Results")
        print("="*50)
        print(f"MRR (Mean Reciprocal Rank):     {results['mrr']:.4f}")
        print(f"NDCG (Norm. Disc. Cum. Gain):  {results['ndcg']:.4f}")
        print(f"Recall@1:                      {results['r1']:.4f}")
        print(f"Recall@5:                      {results['r5']:.4f}")
        print(f"Recall@10:                     {results['r10']:.4f}")
        print(f"Mean Rank:                     {results['mean_rank']:.2f}")
        print("="*50)
    
    def save_results(self, results: Dict[str, float], output_path: str):
        """
        평가 결과를 JSON 파일로 저장합니다.
        """
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {output_path}")


def load_predictions(predictions_path: str) -> List[Dict]:
    """
    예측 결과를 JSON 파일에서 불러옵니다.
    """
    with open(predictions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    """
    평가 스크립트의 메인 함수입니다. 명령행 인자를 받아 평가를 수행합니다.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Visual Dialogue predictions")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions JSON file")
    parser.add_argument("--visdial_data", type=str, required=True,
                       help="Path to VisDial dataset JSON file")
    parser.add_argument("--output", type=str, default="evaluation_results.json",
                       help="Output path for results")
    parser.add_argument("--partial_match", action="store_true", default=True,
                       help="Use partial string matching")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')
    
    try:
        # Load predictions
        predictions = load_predictions(args.predictions)
        logging.info(f"Loaded {len(predictions)} predictions from {args.predictions}")
        
        # Initialize evaluator
        evaluator = VisualDialogueEvaluator(args.visdial_data)
        
        # Evaluate
        results = evaluator.evaluate_predictions(predictions, args.partial_match)
        
        # Print results
        evaluator.print_evaluation_results(results)
        
        # Save results
        evaluator.save_results(results, args.output)
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()