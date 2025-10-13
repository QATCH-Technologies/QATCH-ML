import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm
from pathlib import Path

# Import the required modules
from v4_fusion_dataprocessor import FusionDataprocessor as DP
from v4_fusion import V4ClfPredictor

warnings.filterwarnings('ignore')


class V4ClfBenchmark:
    """
    Benchmark for evaluating V4 Classification predictor performance.
    Applies normalization factors to account for resolution changes.
    """

    def __init__(self, predictor: V4ClfPredictor):
        """
        Initialize benchmark with a predictor instance.

        Args:
            predictor: Initialized V4ClfPredictor
        """
        self.predictor = predictor

        # Normalization factors for POIs with reduced resolution
        self.normalization_factors = {
            1: 1,    # POI1 - no normalization
            2: 1,    # POI2 - no normalization
            4: 25,   # POI4 - normalize by 25
            5: 100,  # POI5 - normalize by 100
            6: 200   # POI6 - normalize by 200
        }

        # POI mapping (skipping index 2 which is POI3)
        self.poi_indices = {
            1: 0,  # Index 0 -> POI1
            2: 1,  # Index 1 -> POI2
            4: 3,  # Index 3 -> POI4 (skipping index 2)
            5: 4,  # Index 4 -> POI5
            6: 5   # Index 5 -> POI6
        }

    def normalize_position(self, position: float, poi_num: int) -> float:
        """
        Apply normalization factor to POI position.

        Args:
            position: Raw position value
            poi_num: POI number (1, 2, 4, 5, or 6)

        Returns:
            Normalized position
        """
        factor = self.normalization_factors.get(poi_num, 1)
        return position / factor

    def load_ground_truth(self, poi_file: str) -> Dict[int, float]:
        """
        Load ground truth POI positions from file.

        Args:
            poi_file: Path to POI file (headerless CSV with 6 indices)

        Returns:
            Dictionary of POI number to normalized position
        """
        poi_data = pd.read_csv(poi_file, header=None).values.flatten()

        ground_truth = {}
        for poi_num, idx in self.poi_indices.items():
            if idx < len(poi_data) and not pd.isna(poi_data[idx]):
                # Apply normalization
                raw_position = poi_data[idx]
                normalized_position = self.normalize_position(
                    raw_position, poi_num)
                ground_truth[poi_num] = normalized_position

        return ground_truth

    def run_benchmark(self,
                      data_dir: str,
                      num_files: int = None,
                      adaptive_thresholds: Dict[int, float] = None,
                      enforce_constraints: bool = False,
                      verbose: bool = True) -> Dict[str, any]:
        """
        Run benchmark on multiple files and compute metrics.

        Args:
            data_dir: Directory containing data files
            num_files: Number of files to process (None for all)
            adaptive_thresholds: Custom thresholds for each POI
            enforce_constraints: Whether to enforce sequential constraints
            verbose: Whether to print progress

        Returns:
            Dictionary containing benchmark results and metrics
        """
        # Load file pairs
        file_pairs = DP.load_content(data_dir, num_files)

        if verbose:
            print(f"\nLoaded {len(file_pairs)} file pairs for benchmarking")
            print("="*60)

        # Storage for results
        all_predictions = []
        all_ground_truths = []
        per_poi_predictions = {poi: [] for poi in [1, 2, 4, 5, 6]}
        per_poi_ground_truths = {poi: [] for poi in [1, 2, 4, 5, 6]}

        # Tracking
        detection_stats = {
            'total_files': len(file_pairs),
            'successful_predictions': 0,
            'failed_predictions': 0,
            'per_poi_detections': {poi: 0 for poi in [1, 2, 4, 5, 6]},
            'per_poi_misses': {poi: 0 for poi in [1, 2, 4, 5, 6]},
            'method_usage': {'classification': 0, 'regression': 0}
        }

        # Process each file pair
        for i, (data_file, poi_file) in enumerate(tqdm(file_pairs, desc="Processing files")):
            try:
                # Load data
                df = pd.read_csv(data_file)

                # Load ground truth
                ground_truth = self.load_ground_truth(poi_file)

                # Run prediction
                result = self.predictor.predict(
                    df=df,
                    adaptive_thresholds=adaptive_thresholds,
                    enforce_constraints=enforce_constraints
                )

                predicted_positions = result['poi_locations']

                # Normalize predicted positions and compare with ground truth
                for poi_num in [1, 2, 4, 5, 6]:
                    if poi_num in ground_truth:
                        gt_pos = ground_truth[poi_num]

                        if poi_num in predicted_positions:
                            # Normalize prediction
                            pred_pos = self.normalize_position(
                                predicted_positions[poi_num], poi_num
                            )

                            # Store for metrics
                            all_predictions.append(pred_pos)
                            all_ground_truths.append(gt_pos)
                            per_poi_predictions[poi_num].append(pred_pos)
                            per_poi_ground_truths[poi_num].append(gt_pos)

                            # Update stats
                            detection_stats['per_poi_detections'][poi_num] += 1
                            # Classification method is always used for V4ClfPredictor
                            detection_stats['method_usage']['classification'] += 1
                        else:
                            # POI missed
                            detection_stats['per_poi_misses'][poi_num] += 1

                detection_stats['successful_predictions'] += 1

            except Exception as e:
                if verbose:
                    print(f"\nError processing {Path(data_file).name}: {e}")
                detection_stats['failed_predictions'] += 1

        # Calculate metrics
        metrics = self.calculate_metrics(
            all_predictions, all_ground_truths,
            per_poi_predictions, per_poi_ground_truths
        )

        # Compile results
        results = {
            'metrics': metrics,
            'detection_stats': detection_stats,
            'predictions': all_predictions,
            'ground_truths': all_ground_truths,
            'per_poi_predictions': per_poi_predictions,
            'per_poi_ground_truths': per_poi_ground_truths
        }

        if verbose:
            self.print_results(results)

        return results

    def calculate_metrics(self,
                          all_preds: List[float],
                          all_truths: List[float],
                          per_poi_preds: Dict[int, List[float]],
                          per_poi_truths: Dict[int, List[float]]) -> Dict:
        """
        Calculate performance metrics.

        Returns:
            Dictionary containing MAE, RMSE, R2, and error rate
        """
        metrics = {}

        # Overall metrics
        if len(all_preds) > 0:
            all_preds = np.array(all_preds)
            all_truths = np.array(all_truths)

            metrics['overall'] = {
                'mae': mean_absolute_error(all_truths, all_preds),
                'rmse': np.sqrt(mean_squared_error(all_truths, all_preds)),
                'r2': r2_score(all_truths, all_preds) if len(all_preds) > 1 else 0,
                'mean_error': np.mean(all_preds - all_truths),
                'std_error': np.std(all_preds - all_truths),
                'n_samples': len(all_preds)
            }

            # Error rate (percentage of predictions with >5% relative error)
            relative_errors = np.abs(
                (all_preds - all_truths) / (all_truths + 1e-10))
            metrics['overall']['error_rate'] = np.mean(
                relative_errors > 0.05) * 100
        else:
            metrics['overall'] = {
                'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                'mean_error': np.nan, 'std_error': np.nan,
                'error_rate': np.nan, 'n_samples': 0
            }

        # Per-POI metrics
        metrics['per_poi'] = {}
        for poi_num in [1, 2, 4, 5, 6]:
            if len(per_poi_preds[poi_num]) > 0:
                preds = np.array(per_poi_preds[poi_num])
                truths = np.array(per_poi_truths[poi_num])

                metrics['per_poi'][f'POI{poi_num}'] = {
                    'mae': mean_absolute_error(truths, preds),
                    'rmse': np.sqrt(mean_squared_error(truths, preds)),
                    'r2': r2_score(truths, preds) if len(preds) > 1 else 0,
                    'mean_error': np.mean(preds - truths),
                    'std_error': np.std(preds - truths),
                    'n_samples': len(preds)
                }

                # Error rate
                relative_errors = np.abs((preds - truths) / (truths + 1e-10))
                metrics['per_poi'][f'POI{poi_num}']['error_rate'] = np.mean(
                    relative_errors > 0.05) * 100
            else:
                metrics['per_poi'][f'POI{poi_num}'] = {
                    'mae': np.nan, 'rmse': np.nan, 'r2': np.nan,
                    'mean_error': np.nan, 'std_error': np.nan,
                    'error_rate': np.nan, 'n_samples': 0
                }

        return metrics

    def print_results(self, results: Dict):
        """Print formatted benchmark results."""
        print("\n" + "="*70)
        print("V4 CLASSIFICATION BENCHMARK RESULTS")
        print("="*70)

        # Detection statistics
        stats = results['detection_stats']
        print("\nDetection Statistics:")
        print(f"  Total files processed: {stats['total_files']}")
        print(f"  Successful predictions: {stats['successful_predictions']}")
        print(f"  Failed predictions: {stats['failed_predictions']}")

        print("\nPer-POI Detection Rates:")
        for poi_num in [1, 2, 4, 5, 6]:
            detections = stats['per_poi_detections'][poi_num]
            misses = stats['per_poi_misses'][poi_num]
            total = detections + misses
            if total > 0:
                rate = (detections / total) * 100
                print(f"  POI{poi_num}: {detections}/{total} ({rate:.1f}%)")

        print(f"\nMethod Usage:")
        print(f"  Classification: {stats['method_usage']['classification']}")
        print(f"  Regression: {stats['method_usage']['regression']}")

        # Overall metrics
        print("\n" + "-"*70)
        print("OVERALL METRICS (Normalized)")
        print("-"*70)
        overall = results['metrics']['overall']
        if overall['n_samples'] > 0:
            print(f"  MAE:   {overall['mae']:.2f}")
            print(f"  RMSE:  {overall['rmse']:.2f}")
            print(f"  R²:    {overall['r2']:.4f}")
            print(f"  ERR:   {overall['error_rate']:.1f}%")
            print(
                f"  Mean Error: {overall['mean_error']:.2f} ± {overall['std_error']:.2f}")
            print(f"  Samples: {overall['n_samples']}")
        else:
            print("  No valid predictions")

        # Per-POI metrics
        print("\n" + "-"*70)
        print("PER-POI METRICS (Normalized)")
        print("-"*70)

        for poi_name, poi_metrics in results['metrics']['per_poi'].items():
            if poi_metrics['n_samples'] > 0:
                poi_num = int(poi_name[-1])
                norm_factor = self.normalization_factors.get(poi_num, 1)

                print(f"\n{poi_name} (Normalization factor: {norm_factor}):")
                print(f"  MAE:   {poi_metrics['mae']:.2f}")
                print(f"  RMSE:  {poi_metrics['rmse']:.2f}")
                print(f"  R²:    {poi_metrics['r2']:.4f}")
                print(f"  ERR:   {poi_metrics['error_rate']:.1f}%")
                print(
                    f"  Mean Error: {poi_metrics['mean_error']:.2f} ± {poi_metrics['std_error']:.2f}")
                print(f"  Samples: {poi_metrics['n_samples']}")

        print("\n" + "="*70)


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    model_path = 'QModel/SavedModels/qmodel_v4_fusion/v4_model_pytorch.pth'
    scaler_path = 'QModel/SavedModels/qmodel_v4_fusion/v4_scaler_pytorch.joblib'
    config_path = 'QModel/SavedModels/qmodel_v4_fusion/v4_config_pytorch.json'

    predictor = V4ClfPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        config_path=config_path
    )

    # Create benchmark
    benchmark = V4ClfBenchmark(predictor)

    # Run benchmark
    results = benchmark.run_benchmark(
        data_dir='content/PROTEIN',  # or 'content/valid'
        num_files=200,  # Process first 200 files, or None for all
        adaptive_thresholds=None,  # Use default thresholds from predictor
        enforce_constraints=False,
        verbose=True
    )

    # Save results to JSON for further analysis
    import json
    with open('v4_clf_benchmark_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        json.dump(convert_to_serializable(results['metrics']), f, indent=2)
        print("\nResults saved to v4_clf_benchmark_results.json")
