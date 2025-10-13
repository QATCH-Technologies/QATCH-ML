from v4_fusion_dataprocessor import FusionDataprocessor
from qmodel_v4_predictor import QModelPredictorV4
from v4_fusion import V4Fusion as QModelV4Fusion
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
from pathlib import Path
import time
from dataclasses import dataclass
import json
warnings.filterwarnings('ignore')


@dataclass
class PredictionResult:
    """Store prediction results for a single file."""
    filename: str
    predictor_name: str
    predictions: Dict[str, int]  # POI name -> predicted index
    ground_truth: Dict[str, int]  # POI name -> true index
    relative_times: np.ndarray  # Array of relative times from data file
    inference_time: float

    def get_errors(self) -> Dict[str, float]:
        """Calculate absolute time errors for each POI."""
        errors = {}
        for poi in self.predictions:
            pred_idx = self.predictions[poi]
            true_idx = self.ground_truth[poi]

            if pred_idx != -1 and true_idx != -1:
                # Check indices are within bounds
                if pred_idx < len(self.relative_times) and true_idx < len(self.relative_times):
                    pred_time = self.relative_times[pred_idx]
                    true_time = self.relative_times[true_idx]
                    errors[poi] = abs(pred_time - true_time)
                else:
                    errors[poi] = np.nan
            else:
                errors[poi] = np.nan
        return errors


class POIBenchmark:
    """Benchmark suite for comparing POI predictors."""

    # Normalization factors for each POI
    NORMALIZATION_FACTORS = {
        1: 1,    # POI1 - no normalization
        2: 1,    # POI2 - no normalization
        4: 25,   # POI4 - normalize by 25
        5: 100,  # POI5 - normalize by 100
        6: 200   # POI6 - normalize by 200
    }

    def __init__(self):
        self.results: List[PredictionResult] = []
        self.poi_names = ['POI1', 'POI2', 'POI4', 'POI5', 'POI6']
        self.poi_indices = [0, 1, 3, 4, 5]  # Indices in ground truth file

    def load_ground_truth(self, poi_file: str) -> Dict[str, int]:
        """Load ground truth POI indices from file."""
        gt_values = pd.read_csv(poi_file, header=None).values.flatten()

        ground_truth = {}
        for poi_name, idx in zip(self.poi_names, self.poi_indices):
            if idx < len(gt_values):
                ground_truth[poi_name] = int(gt_values[idx])
            else:
                ground_truth[poi_name] = -1

        return ground_truth

    def load_relative_times(self, data_file: str) -> np.ndarray:
        """Load Relative_time feature from data file."""
        try:
            df = pd.read_csv(data_file)
            if 'Relative_time' in df.columns:
                return df['Relative_time'].values
            else:
                print(
                    f"Warning: 'Relative_time' column not found in {data_file}")
                return np.array([])
        except Exception as e:
            print(f"Error loading Relative_time from {data_file}: {e}")
            return np.array([])

    def extract_predictions(self, pred_dict: Dict, predictor_name: str) -> Dict[str, int]:
        """Extract first prediction index for each POI."""
        predictions = {}

        for poi_name in self.poi_names:
            if poi_name in pred_dict:
                indices = pred_dict[poi_name].get('indices', [-1])
                predictions[poi_name] = indices[0] if indices else -1
            else:
                predictions[poi_name] = -1

        return predictions

    def run_predictor(self, predictor, data_file: str, predictor_name: str,
                      poi_file: str, **kwargs) -> PredictionResult:
        """Run a single predictor on a data file."""
        # Load ground truth
        ground_truth = self.load_ground_truth(poi_file)

        # Load relative times
        relative_times = self.load_relative_times(data_file)

        # Time the prediction
        start_time = time.time()
        try:
            pred_dict = predictor.predict(file_buffer=data_file, **kwargs)
            inference_time = time.time() - start_time

            # Extract predictions
            predictions = self.extract_predictions(pred_dict, predictor_name)

        except Exception as e:
            print(f"Error predicting {data_file} with {predictor_name}: {e}")
            inference_time = -1
            predictions = {poi: -1 for poi in self.poi_names}

        return PredictionResult(
            filename=Path(data_file).name,
            predictor_name=predictor_name,
            predictions=predictions,
            ground_truth=ground_truth,
            relative_times=relative_times,
            inference_time=inference_time
        )

    def benchmark(self, data_dir: str, num_datasets: int,
                  fusion_predictor: QModelV4Fusion,
                  v4_predictor: QModelPredictorV4,
                  fusion_kwargs: Optional[Dict] = None,
                  v4_kwargs: Optional[Dict] = None):
        """Run benchmark on both predictors."""
        fusion_kwargs = fusion_kwargs or {}
        v4_kwargs = v4_kwargs or {}

        print(f"Loading {num_datasets} datasets from {data_dir}...")
        dataset_pairs = FusionDataprocessor.load_content(
            data_dir, num_datasets)
        print(f"Loaded {len(dataset_pairs)} dataset pairs\n")

        self.results = []

        for i, (data_file, poi_file) in enumerate(dataset_pairs, 1):
            print(f"[{i}/{len(dataset_pairs)}] Processing: {Path(data_file).name}")

            # Run V4 Fusion
            result_fusion = self.run_predictor(
                fusion_predictor, data_file, "V4_Fusion", poi_file, **fusion_kwargs
            )
            self.results.append(result_fusion)
            print(f"  V4_Fusion: {result_fusion.inference_time:.3f}s")

            # Run V4 Predictor
            result_v4 = self.run_predictor(
                v4_predictor, data_file, "V4_Predictor", poi_file, **v4_kwargs
            )
            self.results.append(result_v4)
            print(f"  V4_Predictor: {result_v4.inference_time:.3f}s\n")

    def compute_metrics(self) -> pd.DataFrame:
        """Compute comprehensive metrics for all predictions."""
        metrics_data = []

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            predictor_results = [
                r for r in self.results if r.predictor_name == predictor_name]

            for poi_name in self.poi_names:
                errors = []
                squared_errors = []
                valid_predictions = 0
                total_predictions = 0

                for result in predictor_results:
                    pred = result.predictions[poi_name]
                    true = result.ground_truth[poi_name]
                    total_predictions += 1

                    if pred != -1 and true != -1:
                        error_dict = result.get_errors()
                        error = error_dict[poi_name]
                        if not np.isnan(error):
                            errors.append(error)
                            squared_errors.append(error ** 2)
                            valid_predictions += 1

                if errors:
                    mae = np.mean(errors)
                    mse = np.mean(squared_errors)
                    rmse = np.sqrt(mse)
                    median_ae = np.median(errors)
                    std_ae = np.std(errors)
                else:
                    mae = mse = rmse = median_ae = std_ae = np.nan

                metrics_data.append({
                    'Predictor': predictor_name,
                    'POI': poi_name,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'Median_AE': median_ae,
                    'Std_AE': std_ae,
                    'Valid_Predictions': valid_predictions,
                    'Total_Predictions': total_predictions,
                    'Detection_Rate': valid_predictions / total_predictions if total_predictions > 0 else 0
                })

        return pd.DataFrame(metrics_data)

    def compute_normalized_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute normalized metrics using predefined normalization factors."""
        normalized_data = []

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            predictor_results = [
                r for r in self.results if r.predictor_name == predictor_name]

            for poi_name in self.poi_names:
                # Get normalization factor for this POI
                poi_num = int(poi_name.replace('POI', ''))
                norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)

                normalized_errors = []

                for result in predictor_results:
                    error_dict = result.get_errors()
                    error = error_dict[poi_name]

                    if not np.isnan(error):
                        # Normalize error by the predefined factor
                        normalized_error = error / norm_factor
                        normalized_errors.append(normalized_error)

                if normalized_errors:
                    norm_mae = np.mean(normalized_errors)
                    norm_mse = np.mean([e**2 for e in normalized_errors])
                    norm_rmse = np.sqrt(norm_mse)
                else:
                    norm_mae = norm_mse = norm_rmse = np.nan

                normalized_data.append({
                    'Predictor': predictor_name,
                    'POI': poi_name,
                    'Norm_Factor': norm_factor,
                    'Normalized_MAE': norm_mae,
                    'Normalized_MSE': norm_mse,
                    'Normalized_RMSE': norm_rmse
                })

        return pd.DataFrame(normalized_data)

    def plot_results(self, metrics_df: pd.DataFrame, normalized_df: pd.DataFrame,
                     save_path: Optional[str] = None):
        """Create comprehensive visualization of benchmark results."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. MAE Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        mae_pivot = metrics_df.pivot(
            index='POI', columns='Predictor', values='MAE')
        mae_pivot.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Mean Absolute Error (MAE)',
                      fontweight='bold', fontsize=12)
        ax1.set_ylabel('MAE (time units)', fontweight='bold')
        ax1.set_xlabel('')
        ax1.legend(title='Predictor')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. MSE Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        mse_pivot = metrics_df.pivot(
            index='POI', columns='Predictor', values='MSE')
        mse_pivot.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'])
        ax2.set_title('Mean Squared Error (MSE)',
                      fontweight='bold', fontsize=12)
        ax2.set_ylabel('MSE (time units²)', fontweight='bold')
        ax2.set_xlabel('')
        ax2.legend(title='Predictor')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. RMSE Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        rmse_pivot = metrics_df.pivot(
            index='POI', columns='Predictor', values='RMSE')
        rmse_pivot.plot(kind='bar', ax=ax3, color=['#2E86AB', '#A23B72'])
        ax3.set_title('Root Mean Squared Error (RMSE)',
                      fontweight='bold', fontsize=12)
        ax3.set_ylabel('RMSE (time units)', fontweight='bold')
        ax3.set_xlabel('')
        ax3.legend(title='Predictor')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 4. Normalized MAE
        ax4 = fig.add_subplot(gs[1, 0])
        norm_mae_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='Normalized_MAE')
        norm_mae_pivot.plot(kind='bar', ax=ax4, color=['#2E86AB', '#A23B72'])
        ax4.set_title('Normalized MAE', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Normalized MAE', fontweight='bold')
        ax4.set_xlabel('')
        ax4.legend(title='Predictor')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add normalization factor labels
        for idx, poi in enumerate(self.poi_names):
            poi_num = int(poi.replace('POI', ''))
            norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)
            ax4.text(idx, ax4.get_ylim()[1] * 0.95, f'÷{norm_factor}',
                     ha='center', va='top', fontsize=8, style='italic', alpha=0.7)

        # 5. Normalized MSE
        ax5 = fig.add_subplot(gs[1, 1])
        norm_mse_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='Normalized_MSE')
        norm_mse_pivot.plot(kind='bar', ax=ax5, color=['#2E86AB', '#A23B72'])
        ax5.set_title('Normalized MSE', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Normalized MSE', fontweight='bold')
        ax5.set_xlabel('')
        ax5.legend(title='Predictor')
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add normalization factor labels
        for idx, poi in enumerate(self.poi_names):
            poi_num = int(poi.replace('POI', ''))
            norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)
            ax5.text(idx, ax5.get_ylim()[1] * 0.95, f'÷{norm_factor}',
                     ha='center', va='top', fontsize=8, style='italic', alpha=0.7)

        # 6. Normalized RMSE
        ax6 = fig.add_subplot(gs[1, 2])
        norm_rmse_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='Normalized_RMSE')
        norm_rmse_pivot.plot(kind='bar', ax=ax6, color=['#2E86AB', '#A23B72'])
        ax6.set_title('Normalized RMSE', fontweight='bold', fontsize=12)
        ax6.set_ylabel('Normalized RMSE', fontweight='bold')
        ax6.set_xlabel('')
        ax6.legend(title='Predictor')
        ax6.grid(axis='y', alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add normalization factor labels
        for idx, poi in enumerate(self.poi_names):
            poi_num = int(poi.replace('POI', ''))
            norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)
            ax6.text(idx, ax6.get_ylim()[1] * 0.95, f'÷{norm_factor}',
                     ha='center', va='top', fontsize=8, style='italic', alpha=0.7)

        # 7. Detection Rate
        ax7 = fig.add_subplot(gs[1, 3])
        detection_pivot = metrics_df.pivot(
            index='POI', columns='Predictor', values='Detection_Rate')
        detection_pivot.plot(kind='bar', ax=ax7, color=['#2E86AB', '#A23B72'])
        ax7.set_title('Detection Rate', fontweight='bold', fontsize=12)
        ax7.set_ylabel('Detection Rate', fontweight='bold')
        ax7.set_xlabel('')
        ax7.set_ylim([0, 1.1])
        ax7.legend(title='Predictor')
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 8. Error Distribution Box Plot
        ax8 = fig.add_subplot(gs[2, :3])
        error_data = []
        for result in self.results:
            errors = result.get_errors()
            for poi, error in errors.items():
                if not np.isnan(error):
                    error_data.append({
                        'Predictor': result.predictor_name,
                        'POI': poi,
                        'Error': error
                    })

        if error_data:
            error_df = pd.DataFrame(error_data)
            sns.boxplot(data=error_df, x='POI', y='Error', hue='Predictor', ax=ax8,
                        palette=['#2E86AB', '#A23B72'])
            ax8.set_title('Error Distribution by POI',
                          fontweight='bold', fontsize=12)
            ax8.set_ylabel('Absolute Error (time units)', fontweight='bold')
            ax8.set_xlabel('')
            ax8.grid(axis='y', alpha=0.3)
            plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 9. Summary Statistics Table
        ax9 = fig.add_subplot(gs[2, 3])
        ax9.axis('off')

        summary_text = "Overall Summary\n" + "="*30 + "\n\n"

        # Add normalization factors info
        summary_text += "Norm Factors:\n"
        for poi_name in self.poi_names:
            poi_num = int(poi_name.replace('POI', ''))
            norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)
            summary_text += f"  {poi_name}: ÷{norm_factor}\n"
        summary_text += "\n"

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            pred_results = [
                r for r in self.results if r.predictor_name == predictor_name]
            pred_norm = normalized_df[normalized_df['Predictor']
                                      == predictor_name]

            # Overall MAE
            all_errors = []
            for r in pred_results:
                for err in r.get_errors().values():
                    if not np.isnan(err):
                        all_errors.append(err)

            if all_errors:
                overall_mae = np.mean(all_errors)
                overall_mse = np.mean([e**2 for e in all_errors])
                overall_norm_mae = pred_norm['Normalized_MAE'].mean()

                # Inference time
                valid_times = [
                    r.inference_time for r in pred_results if r.inference_time > 0]
                avg_time = np.mean(valid_times) if valid_times else 0

                summary_text += f"{predictor_name}:\n"
                summary_text += f"  Raw MAE: {overall_mae:.4f}\n"
                summary_text += f"  Raw MSE: {overall_mse:.4f}\n"
                summary_text += f"  Norm MAE: {overall_norm_mae:.3f}\n"
                summary_text += f"  Avg Time: {avg_time:.3f}s\n"
                summary_text += f"  N: {len(pred_results)}\n\n"

        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        fig.suptitle('POI Predictor Benchmark Results (Time-Based Errors)',
                     fontsize=16, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")

        plt.show()

    def save_detailed_log(self, output_path: str = "benchmark_detailed_log.json"):
        """Save detailed JSON log of all predictions."""
        log_data = {
            "benchmark_metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "num_files": len(self.results) // 2,  # Divided by 2 predictors
                "poi_names": self.poi_names,
                "normalization_factors": self.NORMALIZATION_FACTORS
            },
            "predictions": []
        }

        for result in self.results:
            # Get time-based errors
            errors = result.get_errors()

            # Get predicted and true times
            pred_times = {}
            true_times = {}
            for poi in self.poi_names:
                pred_idx = result.predictions[poi]
                true_idx = result.ground_truth[poi]

                if pred_idx != -1 and pred_idx < len(result.relative_times):
                    pred_times[poi] = float(result.relative_times[pred_idx])
                else:
                    pred_times[poi] = None

                if true_idx != -1 and true_idx < len(result.relative_times):
                    true_times[poi] = float(result.relative_times[true_idx])
                else:
                    true_times[poi] = None

            prediction_entry = {
                "filename": result.filename,
                "predictor": result.predictor_name,
                "inference_time_seconds": result.inference_time,
                "data_length": len(result.relative_times),
                "poi_predictions": {}
            }

            for poi in self.poi_names:
                prediction_entry["poi_predictions"][poi] = {
                    "predicted_index": int(result.predictions[poi]) if result.predictions[poi] != -1 else None,
                    "true_index": int(result.ground_truth[poi]) if result.ground_truth[poi] != -1 else None,
                    "predicted_time": pred_times[poi],
                    "true_time": true_times[poi],
                    "time_error": float(errors[poi]) if not np.isnan(errors[poi]) else None,
                    "index_error": abs(result.predictions[poi] - result.ground_truth[poi])
                    if result.predictions[poi] != -1 and result.ground_truth[poi] != -1
                    else None
                }

            log_data["predictions"].append(prediction_entry)

        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"\nDetailed log saved to: {output_path}")
        return log_data

    def generate_report(self, metrics_df: pd.DataFrame, normalized_df: pd.DataFrame,
                        output_path: Optional[str] = None):
        """Generate a detailed text report."""
        report = []
        report.append("="*80)
        report.append("POI PREDICTOR BENCHMARK REPORT (TIME-BASED ERRORS)")
        report.append("="*80)
        report.append("")

        # Normalization factors
        report.append("NORMALIZATION FACTORS")
        report.append("-"*80)
        for poi_name in self.poi_names:
            poi_num = int(poi_name.replace('POI', ''))
            norm_factor = self.NORMALIZATION_FACTORS.get(poi_num, 1)
            report.append(f"  {poi_name}: ÷{norm_factor}")
        report.append("")

        # Raw Metrics
        report.append("RAW METRICS (in time units)")
        report.append("-"*80)
        report.append(metrics_df.to_string(index=False))
        report.append("")

        # Normalized Metrics
        report.append("NORMALIZED METRICS (using predefined factors)")
        report.append("-"*80)
        report.append(normalized_df.to_string(index=False))
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-"*80)

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            pred_metrics = metrics_df[metrics_df['Predictor']
                                      == predictor_name]
            pred_norm_metrics = normalized_df[normalized_df['Predictor']
                                              == predictor_name]
            pred_results = [
                r for r in self.results if r.predictor_name == predictor_name]

            report.append(f"\n{predictor_name}:")
            report.append(f"  Raw Metrics:")
            report.append(
                f"    Average MAE: {pred_metrics['MAE'].mean():.4f} time units")
            report.append(
                f"    Average MSE: {pred_metrics['MSE'].mean():.4f} time units²")
            report.append(
                f"    Average RMSE: {pred_metrics['RMSE'].mean():.4f} time units")
            report.append(f"  Normalized Metrics:")
            report.append(
                f"    Average Norm MAE: {pred_norm_metrics['Normalized_MAE'].mean():.4f}")
            report.append(
                f"    Average Norm MSE: {pred_norm_metrics['Normalized_MSE'].mean():.4f}")
            report.append(
                f"    Average Norm RMSE: {pred_norm_metrics['Normalized_RMSE'].mean():.4f}")

            valid_times = [
                r.inference_time for r in pred_results if r.inference_time > 0]
            if valid_times:
                report.append(f"  Performance:")
                report.append(
                    f"    Avg Inference Time: {np.mean(valid_times):.3f}s")
                report.append(f"    Total Time: {np.sum(valid_times):.3f}s")

        report.append("")
        report.append("="*80)

        report_text = "\n".join(report)
        print(report_text)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            print(f"\nReport saved to: {output_path}")

        return report_text


# Main execution
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "content/XANTHAM"  # Update this
    NUM_DATASETS = np.inf  # Number of datasets to benchmark

    # Model paths - Update these to your actual model paths
    FUSION_REG_PATH_1 = "poi_model_mini_window_0.pth"
    FUSION_REG_PATH_2 = "poi_model_mini_window_1.pth"
    FUSION_CLF_PATH = "v4_model_pytorch.pth"

    V4_MODEL_PATH = "QModel/SavedModels/qmode_v4_tf/v4_model_mini.h5"
    V4_SCALER_PATH = "QModel/SavedModels/qmode_v4_tf/v4_scaler_mini.joblib"

    # Initialize predictors
    print("Initializing predictors...")
    fusion_predictor = QModelV4Fusion(
        reg_path_1=FUSION_REG_PATH_1,
        reg_path_2=FUSION_REG_PATH_2,
        clf_path=FUSION_CLF_PATH,
        reg_batch_size=512,
        clf_batch_size=256
    )

    v4_predictor = QModelPredictorV4(
        model_path=V4_MODEL_PATH,
        scaler_path=V4_SCALER_PATH,
        window_size=128,
        stride=16,
        tolerance=64
    )

    # Run benchmark
    print("\nStarting benchmark...")
    benchmark = POIBenchmark()

    benchmark.benchmark(
        data_dir=DATA_DIR,
        num_datasets=NUM_DATASETS,
        fusion_predictor=fusion_predictor,
        v4_predictor=v4_predictor,
        fusion_kwargs={'visualize': False},
        v4_kwargs={'top_k': 3, 'apply_constraints': True}
    )

    # Compute metrics
    print("\nComputing metrics...")
    metrics_df = benchmark.compute_metrics()
    normalized_df = benchmark.compute_normalized_metrics(metrics_df)

    # Generate report
    benchmark.generate_report(
        metrics_df,
        normalized_df,
        output_path="benchmark_report.txt"
    )

    # Plot results
    print("\nGenerating plots...")
    benchmark.plot_results(
        metrics_df,
        normalized_df,
        save_path="benchmark_results.png"
    )

    # Save metrics to CSV
    metrics_df.to_csv("benchmark_metrics.csv", index=False)
    normalized_df.to_csv("benchmark_normalized_metrics.csv", index=False)
    print("\nMetrics saved to CSV files")

    print("\n✓ Benchmark complete!")
