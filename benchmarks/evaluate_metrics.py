"""
Evaluation Metrics Script
Calculates precision, recall, accuracy, top-1, top-5 identification rates
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix

from src.detect_faces import FaceDetector
from src.extract_embeddings import FeatureExtractor
from src.match_faces import FaceMatcher
from src.utils import load_image


class FRSEvaluator:
    """Evaluate Face Recognition System performance"""
    
    def __init__(self):
        """Initialize evaluator"""
        self.detector = FaceDetector()
        self.extractor = FeatureExtractor()
        self.matcher = FaceMatcher()
        
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "detection_metrics": {},
            "recognition_metrics": {},
            "identification_metrics": {}
        }
    
    def evaluate_detection(self, test_images: List[np.ndarray], 
                          ground_truth_boxes: List[List]) -> Dict:
        """
        Evaluate face detection performance
        
        Args:
            test_images: List of test images
            ground_truth_boxes: List of ground truth bounding boxes for each image
        
        Returns:
            Dictionary with precision, recall, F1-score
        """
        print("\n" + "="*60)
        print("EVALUATING FACE DETECTION")
        print("="*60)
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        iou_threshold = 0.5  # IoU threshold for matching
        
        for img, gt_boxes in zip(test_images, ground_truth_boxes):
            # Detect faces
            pred_boxes, _, confidences = self.detector.detect(img, return_confidence=True)
            
            # Match predicted boxes with ground truth
            matched_gt = set()
            
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = self._calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gt.add(best_gt_idx)
                else:
                    false_positives += 1
            
            # Count unmatched ground truth boxes as false negatives
            false_negatives += len(gt_boxes) - len(matched_gt)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "total_ground_truth": sum(len(boxes) for boxes in ground_truth_boxes),
            "total_predictions": sum(len(self.detector.detect(img)[0]) for img in test_images)
        }
        
        self.results["detection_metrics"] = metrics
        self._print_detection_metrics(metrics)
        
        return metrics
    
    def evaluate_recognition_accuracy(self, gallery_embeddings: Dict[int, np.ndarray],
                                     query_embeddings: List[Tuple[np.ndarray, int]],
                                     threshold: float = 0.6) -> Dict:
        """
        Evaluate face recognition accuracy
        
        Args:
            gallery_embeddings: Dict mapping identity_id to embedding
            query_embeddings: List of (embedding, true_identity_id) tuples
            threshold: Similarity threshold
        
        Returns:
            Dictionary with accuracy metrics
        """
        print("\n" + "="*60)
        print("EVALUATING FACE RECOGNITION")
        print("="*60)
        
        # Build gallery
        self.matcher.clear()
        for identity_id, embedding in gallery_embeddings.items():
            self.matcher.add_embedding(embedding, identity_id, f"Person_{identity_id}")
        
        true_positives = 0  # Correct recognition
        false_positives = 0  # Wrong person recognized
        true_negatives = 0   # Correctly rejected (unknown)
        false_negatives = 0  # Should recognize but didn't
        
        predictions = []
        ground_truths = []
        similarities = []
        
        for query_emb, true_id in query_embeddings:
            matches = self.matcher.match(query_emb, threshold=threshold, top_k=1)
            
            if matches:
                pred_id = matches[0][0]  # Best match ID
                similarity = matches[0][2]  # Similarity score
                
                similarities.append(similarity)
                predictions.append(pred_id)
                ground_truths.append(true_id)
                
                if pred_id == true_id:
                    true_positives += 1
                else:
                    false_positives += 1
            else:
                # No match found
                if true_id in gallery_embeddings:
                    false_negatives += 1  # Should have matched
                else:
                    true_negatives += 1  # Correctly rejected unknown
                
                predictions.append(-1)
                ground_truths.append(true_id)
                similarities.append(0.0)
        
        # Calculate metrics
        accuracy = (true_positives + true_negatives) / len(query_embeddings) if query_embeddings else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # False positive and negative rates
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "true_negatives": true_negatives,
            "false_negatives": false_negatives,
            "threshold": threshold,
            "total_queries": len(query_embeddings),
            "avg_similarity": np.mean(similarities) if similarities else 0
        }
        
        self.results["recognition_metrics"] = metrics
        self._print_recognition_metrics(metrics)
        
        return metrics
    
    def evaluate_identification_rate(self, gallery_embeddings: Dict[int, np.ndarray],
                                    query_embeddings: List[Tuple[np.ndarray, int]],
                                    k_values: List[int] = [1, 5, 10]) -> Dict:
        """
        Evaluate top-K identification rates
        
        Args:
            gallery_embeddings: Dict mapping identity_id to embedding
            query_embeddings: List of (embedding, true_identity_id) tuples
            k_values: List of K values to evaluate
        
        Returns:
            Dictionary with top-K identification rates
        """
        print("\n" + "="*60)
        print("EVALUATING IDENTIFICATION RATES")
        print("="*60)
        
        # Build gallery
        self.matcher.clear()
        for identity_id, embedding in gallery_embeddings.items():
            self.matcher.add_embedding(embedding, identity_id, f"Person_{identity_id}")
        
        identification_rates = {}
        
        for k in k_values:
            correct_identifications = 0
            
            for query_emb, true_id in query_embeddings:
                # Get top-K matches
                matches = self.matcher.match(query_emb, threshold=0.0, top_k=k)  # No threshold for top-K
                
                # Check if true identity is in top-K
                predicted_ids = [match[0] for match in matches]
                if true_id in predicted_ids:
                    correct_identifications += 1
            
            identification_rate = correct_identifications / len(query_embeddings) if query_embeddings else 0
            identification_rates[f"top_{k}"] = identification_rate
            
            print(f"Top-{k} Identification Rate: {identification_rate*100:.2f}%")
        
        metrics = {
            "identification_rates": identification_rates,
            "total_queries": len(query_embeddings),
            "gallery_size": len(gallery_embeddings)
        }
        
        self.results["identification_metrics"] = metrics
        
        return metrics
    
    def _calculate_iou(self, box1, box2) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        x_min = max(x1_min, x2_min)
        y_min = max(y1_min, y2_min)
        x_max = min(x1_max, x2_max)
        y_max = min(y1_max, y2_max)
        
        if x_max < x_min or y_max < y_min:
            return 0.0
        
        intersection = (x_max - x_min) * (y_max - y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _print_detection_metrics(self, metrics: Dict):
        """Print detection metrics"""
        print("\nDetection Metrics:")
        print("-" * 40)
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
    
    def _print_recognition_metrics(self, metrics: Dict):
        """Print recognition metrics"""
        print("\nRecognition Metrics:")
        print("-" * 40)
        print(f"  Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  False Positive Rate: {metrics['false_positive_rate']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f}")
        print(f"  True Positives: {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
    
    def generate_synthetic_test_data(self, num_identities: int = 20, 
                                    samples_per_identity: int = 5) -> Tuple:
        """
        Generate synthetic test data for evaluation
        
        Returns:
            Tuple of (gallery_embeddings, query_embeddings)
        """
        print(f"\nGenerating synthetic test data...")
        print(f"  Identities: {num_identities}")
        print(f"  Samples per identity: {samples_per_identity}")
        
        gallery_embeddings = {}
        query_embeddings = []
        
        for identity_id in range(num_identities):
            # Generate base embedding for this identity
            base_embedding = np.random.randn(512).astype(np.float32)
            base_embedding = base_embedding / np.linalg.norm(base_embedding)
            
            # Add to gallery
            gallery_embeddings[identity_id] = base_embedding
            
            # Generate query variations
            for _ in range(samples_per_identity):
                # Add small noise to simulate variations
                noise = np.random.randn(512).astype(np.float32) * 0.1
                query_emb = base_embedding + noise
                query_emb = query_emb / np.linalg.norm(query_emb)
                query_embeddings.append((query_emb, identity_id))
        
        # Add some unknown identities (not in gallery)
        for _ in range(10):
            unknown_emb = np.random.randn(512).astype(np.float32)
            unknown_emb = unknown_emb / np.linalg.norm(unknown_emb)
            query_embeddings.append((unknown_emb, -1))  # -1 indicates unknown
        
        return gallery_embeddings, query_embeddings
    
    def save_results(self, output_dir: str = "reports"):
        """Save evaluation results"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = output_path / f"evaluation_results_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✅ Results saved to: {json_file}")
        return str(json_file)
    
    def generate_report(self, output_dir: str = "reports"):
        """Generate comprehensive evaluation report"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = output_path / f"evaluation_report_{timestamp}.md"
        
        with open(md_file, 'w') as f:
            f.write("# Face Recognition System - Evaluation Report\n\n")
            f.write(f"**Generated:** {self.results['timestamp']}\n\n")
            
            # Detection Metrics
            if self.results.get("detection_metrics"):
                f.write("## Detection Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["detection_metrics"].items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # Recognition Metrics
            if self.results.get("recognition_metrics"):
                f.write("## Recognition Performance\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for key, value in self.results["recognition_metrics"].items():
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else value
                    f.write(f"| {key.replace('_', ' ').title()} | {formatted_value} |\n")
                f.write("\n")
            
            # Identification Metrics
            if self.results.get("identification_metrics"):
                f.write("## Identification Rates\n\n")
                rates = self.results["identification_metrics"].get("identification_rates", {})
                f.write("| Rank | Identification Rate |\n")
                f.write("|------|--------------------|\n")
                for key, value in rates.items():
                    f.write(f"| {key.replace('_', '-')} | {value*100:.2f}% |\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*Report generated by FRS Evaluation Suite*\n")
        
        print(f"✅ Report saved to: {md_file}")
        return str(md_file)


def main():
    """Run evaluation"""
    print("\n" + "="*60)
    print("FACE RECOGNITION SYSTEM - EVALUATION")
    print("="*60)
    
    evaluator = FRSEvaluator()
    
    # Generate synthetic test data
    gallery_embeddings, query_embeddings = evaluator.generate_synthetic_test_data(
        num_identities=50,
        samples_per_identity=10
    )
    
    # Evaluate recognition accuracy
    evaluator.evaluate_recognition_accuracy(gallery_embeddings, query_embeddings, threshold=0.6)
    
    # Evaluate identification rates
    evaluator.evaluate_identification_rate(gallery_embeddings, query_embeddings, k_values=[1, 5, 10, 20])
    
    # Save results
    evaluator.save_results()
    evaluator.generate_report()
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
