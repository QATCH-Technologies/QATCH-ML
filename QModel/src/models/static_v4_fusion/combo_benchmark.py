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
    relative_times: np.ndarray  # Array of relative time values
    inference_time: float

    def get_index_errors(self) -> Dict[str, float]:
        """Calculate absolute index errors for each POI."""
        errors = {}
        for poi in self.predictions:
            pred = self.predictions[poi]
            true = self.ground_truth[poi]
            if pred != -1 and true != -1:
                errors[poi] = abs(pred - true)
            else:
                errors[poi] = np.nan
        return errors

    def get_time_errors(self) -> Dict[str, float]:
        """Calculate absolute Relative_time errors for each POI."""
        errors = {}
        for poi in self.predictions:
            pred_idx = self.predictions[poi]
            true_idx = self.ground_truth[poi]

            if pred_idx != -1 and true_idx != -1:
                if pred_idx < len(self.relative_times) and true_idx < len(self.relative_times):
                    pred_time = self.relative_times[pred_idx]
                    true_time = self.relative_times[true_idx]
                    errors[poi] = abs(pred_time - true_time)
                else:
                    errors[poi] = np.nan
            else:
                errors[poi] = np.nan
        return errors

    def get_normalized_time_errors(self) -> Dict[str, float]:
        """Calculate normalized time errors (relative to POI1-to-POI distance)."""
        errors = {}
        time_errors = self.get_time_errors()

        # Get POI1's actual time
        poi1_idx = self.ground_truth.get('POI1', -1)
        if poi1_idx == -1 or poi1_idx >= len(self.relative_times):
            # If POI1 not available, return NaN for all
            return {poi: np.nan for poi in self.predictions}

        poi1_time = self.relative_times[poi1_idx]

        for poi in self.predictions:
            time_error = time_errors.get(poi, np.nan)

            if np.isnan(time_error):
                errors[poi] = np.nan
                continue

            # Get actual time for this POI
            poi_idx = self.ground_truth.get(poi, -1)
            if poi_idx == -1 or poi_idx >= len(self.relative_times):
                errors[poi] = np.nan
                continue

            poi_time = self.relative_times[poi_idx]

            # Normalization factor: time difference from POI1
            norm_factor = abs(poi_time - poi1_time)

            # Special case for POI1: normalization factor would be 0
            if norm_factor < 1e-6:  # essentially zero
                # For POI1, we can use the raw error or skip normalization
                errors[poi] = time_error  # or could set to 0 or 1
            else:
                # Normalized error as percentage
                errors[poi] = time_error / norm_factor

        return errors


class POIBenchmark:
    """Benchmark suite for comparing POI predictors using Relative_time errors."""

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
        """Load Relative_time column from data file."""
        try:
            df = pd.read_csv(data_file)
            if 'Relative_time' in df.columns:
                return df['Relative_time'].values
            else:
                print(f"Warning: No Relative_time column in {data_file}")
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
        # Load ground truth and relative times
        ground_truth = self.load_ground_truth(poi_file)
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

    def compute_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Compute metrics for index-based, time-based, and normalized time-based errors."""
        index_metrics = []
        time_metrics = []
        normalized_metrics = []

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            predictor_results = [
                r for r in self.results if r.predictor_name == predictor_name]

            for poi_name in self.poi_names:
                # Collect errors
                index_errors = []
                time_errors = []
                normalized_errors = []
                valid_predictions = 0
                total_predictions = 0

                for result in predictor_results:
                    pred = result.predictions[poi_name]
                    true = result.ground_truth[poi_name]
                    total_predictions += 1

                    if pred != -1 and true != -1:
                        # Index error
                        index_error = abs(pred - true)
                        index_errors.append(index_error)

                        # Time error
                        time_err_dict = result.get_time_errors()
                        time_err = time_err_dict.get(poi_name, np.nan)

                        # Normalized time error
                        norm_err_dict = result.get_normalized_time_errors()
                        norm_err = norm_err_dict.get(poi_name, np.nan)

                        if not np.isnan(time_err):
                            time_errors.append(time_err)
                            valid_predictions += 1

                        if not np.isnan(norm_err):
                            normalized_errors.append(norm_err)

                # Index metrics
                if index_errors:
                    index_metrics.append({
                        'Predictor': predictor_name,
                        'POI': poi_name,
                        'MAE_Index': np.mean(index_errors),
                        'MSE_Index': np.mean([e**2 for e in index_errors]),
                        'RMSE_Index': np.sqrt(np.mean([e**2 for e in index_errors])),
                        'Median_AE_Index': np.median(index_errors),
                        'Std_AE_Index': np.std(index_errors),
                        'Valid_Predictions': valid_predictions,
                        'Total_Predictions': total_predictions,
                        'Detection_Rate': valid_predictions / total_predictions if total_predictions > 0 else 0
                    })

                # Time metrics
                if time_errors:
                    time_metrics.append({
                        'Predictor': predictor_name,
                        'POI': poi_name,
                        'MAE_Time': np.mean(time_errors),
                        'MSE_Time': np.mean([e**2 for e in time_errors]),
                        'RMSE_Time': np.sqrt(np.mean([e**2 for e in time_errors])),
                        'Median_AE_Time': np.median(time_errors),
                        'Std_AE_Time': np.std(time_errors),
                        'Min_Error_Time': np.min(time_errors),
                        'Max_Error_Time': np.max(time_errors)
                    })

                # Normalized time metrics
                if normalized_errors:
                    normalized_metrics.append({
                        'Predictor': predictor_name,
                        'POI': poi_name,
                        'MAE_Normalized': np.mean(normalized_errors),
                        'MSE_Normalized': np.mean([e**2 for e in normalized_errors]),
                        'RMSE_Normalized': np.sqrt(np.mean([e**2 for e in normalized_errors])),
                        'Median_AE_Normalized': np.median(normalized_errors),
                        'Std_AE_Normalized': np.std(normalized_errors),
                        'Min_Error_Normalized': np.min(normalized_errors),
                        'Max_Error_Normalized': np.max(normalized_errors)
                    })

        return (pd.DataFrame(index_metrics),
                pd.DataFrame(time_metrics),
                pd.DataFrame(normalized_metrics))

    def plot_results(self, index_df: pd.DataFrame, time_df: pd.DataFrame,
                     normalized_df: pd.DataFrame, save_path: Optional[str] = None):
        """Create comprehensive visualization of benchmark results."""
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.3)

        # === INDEX-BASED METRICS (Row 0) ===
        # 1. Index MAE
        ax1 = fig.add_subplot(gs[0, 0])
        mae_pivot = index_df.pivot(
            index='POI', columns='Predictor', values='MAE_Index')
        mae_pivot.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72'])
        ax1.set_title('Index MAE (samples)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('MAE (samples)', fontweight='bold')
        ax1.set_xlabel('')
        ax1.legend(title='Predictor')
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 2. Index RMSE
        ax2 = fig.add_subplot(gs[0, 1])
        rmse_pivot = index_df.pivot(
            index='POI', columns='Predictor', values='RMSE_Index')
        rmse_pivot.plot(kind='bar', ax=ax2, color=['#2E86AB', '#A23B72'])
        ax2.set_title('Index RMSE (samples)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('RMSE (samples)', fontweight='bold')
        ax2.set_xlabel('')
        ax2.legend(title='Predictor')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 3. Detection Rate
        ax3 = fig.add_subplot(gs[0, 2])
        detection_pivot = index_df.pivot(
            index='POI', columns='Predictor', values='Detection_Rate')
        detection_pivot.plot(kind='bar', ax=ax3, color=['#2E86AB', '#A23B72'])
        ax3.set_title('Detection Rate', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Detection Rate', fontweight='bold')
        ax3.set_xlabel('')
        ax3.set_ylim([0, 1.1])
        ax3.legend(title='Predictor')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # === ABSOLUTE TIME METRICS (Row 1) ===
        # 4. Time MAE
        ax4 = fig.add_subplot(gs[1, 0])
        time_mae_pivot = time_df.pivot(
            index='POI', columns='Predictor', values='MAE_Time')
        time_mae_pivot.plot(kind='bar', ax=ax4, color=['#2E86AB', '#A23B72'])
        ax4.set_title('Absolute Time MAE', fontweight='bold', fontsize=12)
        ax4.set_ylabel('MAE (seconds)', fontweight='bold')
        ax4.set_xlabel('')
        ax4.legend(title='Predictor')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 5. Time RMSE
        ax5 = fig.add_subplot(gs[1, 1])
        time_rmse_pivot = time_df.pivot(
            index='POI', columns='Predictor', values='RMSE_Time')
        time_rmse_pivot.plot(kind='bar', ax=ax5, color=['#2E86AB', '#A23B72'])
        ax5.set_title('Absolute Time RMSE', fontweight='bold', fontsize=12)
        ax5.set_ylabel('RMSE (seconds)', fontweight='bold')
        ax5.set_xlabel('')
        ax5.legend(title='Predictor')
        ax5.grid(axis='y', alpha=0.3)
        plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 6. Time MSE
        ax6 = fig.add_subplot(gs[1, 2])
        time_mse_pivot = time_df.pivot(
            index='POI', columns='Predictor', values='MSE_Time')
        time_mse_pivot.plot(kind='bar', ax=ax6, color=['#2E86AB', '#A23B72'])
        ax6.set_title('Absolute Time MSE', fontweight='bold', fontsize=12)
        ax6.set_ylabel('MSE (seconds²)', fontweight='bold')
        ax6.set_xlabel('')
        ax6.legend(title='Predictor')
        ax6.grid(axis='y', alpha=0.3)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # === NORMALIZED TIME METRICS (Row 2) ===
        # 7. Normalized MAE
        ax7 = fig.add_subplot(gs[2, 0])
        norm_mae_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='MAE_Normalized')
        norm_mae_pivot.plot(kind='bar', ax=ax7, color=['#2E86AB', '#A23B72'])
        ax7.set_title('Normalized Time MAE', fontweight='bold', fontsize=12)
        ax7.set_ylabel('MAE (relative to POI1)', fontweight='bold')
        ax7.set_xlabel('')
        ax7.legend(title='Predictor')
        ax7.grid(axis='y', alpha=0.3)
        plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 8. Normalized RMSE
        ax8 = fig.add_subplot(gs[2, 1])
        norm_rmse_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='RMSE_Normalized')
        norm_rmse_pivot.plot(kind='bar', ax=ax8, color=['#2E86AB', '#A23B72'])
        ax8.set_title('Normalized Time RMSE', fontweight='bold', fontsize=12)
        ax8.set_ylabel('RMSE (relative to POI1)', fontweight='bold')
        ax8.set_xlabel('')
        ax8.legend(title='Predictor')
        ax8.grid(axis='y', alpha=0.3)
        plt.setp(ax8.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 9. Normalized MSE
        ax9 = fig.add_subplot(gs[2, 2])
        norm_mse_pivot = normalized_df.pivot(
            index='POI', columns='Predictor', values='MSE_Normalized')
        norm_mse_pivot.plot(kind='bar', ax=ax9, color=['#2E86AB', '#A23B72'])
        ax9.set_title('Normalized Time MSE', fontweight='bold', fontsize=12)
        ax9.set_ylabel('MSE (relative to POI1)', fontweight='bold')
        ax9.set_xlabel('')
        ax9.legend(title='Predictor')
        ax9.grid(axis='y', alpha=0.3)
        plt.setp(ax9.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # === DISTRIBUTION PLOTS (Row 3) ===
        # 10. Normalized Error Distribution Box Plot
        ax10 = fig.add_subplot(gs[3, :2])
        norm_error_data = []
        for result in self.results:
            norm_errors = result.get_normalized_time_errors()
            for poi, error in norm_errors.items():
                # Exclude POI1 for better visualization
                if not np.isnan(error) and poi != 'POI1':
                    norm_error_data.append({
                        'Predictor': result.predictor_name,
                        'POI': poi,
                        'Normalized_Error': error
                    })

        if norm_error_data:
            norm_error_df = pd.DataFrame(norm_error_data)
            sns.boxplot(data=norm_error_df, x='POI', y='Normalized_Error', hue='Predictor', ax=ax10,
                        palette=['#2E86AB', '#A23B72'])
            ax10.set_title('Normalized Error Distribution by POI',
                           fontweight='bold', fontsize=12)
            ax10.set_ylabel(
                'Normalized Error (relative to POI1)', fontweight='bold')
            ax10.set_xlabel('')
            ax10.grid(axis='y', alpha=0.3)
            plt.setp(ax10.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # 11. Summary Statistics
        ax11 = fig.add_subplot(gs[3, 2])
        ax11.axis('off')

        summary_text = "Summary Statistics\n" + "="*35 + "\n\n"

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            pred_results = [
                r for r in self.results if r.predictor_name == predictor_name]
            pred_normalized = normalized_df[normalized_df['Predictor']
                                            == predictor_name]

            # Overall normalized error (excluding POI1)
            all_norm_errors = []
            for r in pred_results:
                for poi, err in r.get_normalized_time_errors().items():
                    if not np.isnan(err) and poi != 'POI1':
                        all_norm_errors.append(err)

            if all_norm_errors:
                overall_norm_mae = np.mean(all_norm_errors)
                overall_norm_rmse = np.sqrt(
                    np.mean([e**2 for e in all_norm_errors]))

                # Inference time
                valid_times = [
                    r.inference_time for r in pred_results if r.inference_time > 0]
                avg_time = np.mean(valid_times) if valid_times else 0

                summary_text += f"{predictor_name}:\n"
                summary_text += f"  Norm MAE: {overall_norm_mae:.4f}\n"
                summary_text += f"  Norm RMSE: {overall_norm_rmse:.4f}\n"
                summary_text += f"  Inference: {avg_time:.3f}s\n"
                summary_text += f"  Samples: {len(pred_results)}\n\n"

        ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # === COMPARISON PLOT (Row 4) ===
        # 12. Absolute vs Normalized Error Comparison
        ax12 = fig.add_subplot(gs[4, :])

        comparison_data = []
        for result in self.results:
            time_errors = result.get_time_errors()
            norm_errors = result.get_normalized_time_errors()

            for poi in self.poi_names:
                if poi != 'POI1':  # Exclude POI1
                    if poi in time_errors and poi in norm_errors:
                        if not np.isnan(time_errors[poi]) and not np.isnan(norm_errors[poi]):
                            comparison_data.append({
                                'Predictor': result.predictor_name,
                                'POI': poi,
                                'Absolute_Error': time_errors[poi],
                                'Normalized_Error': norm_errors[poi]
                            })

        if comparison_data:
            comp_df = pd.DataFrame(comparison_data)
            for predictor in ['V4_Fusion', 'V4_Predictor']:
                pred_data = comp_df[comp_df['Predictor'] == predictor]
                color = '#2E86AB' if predictor == 'V4_Fusion' else '#A23B72'
                ax12.scatter(pred_data['Absolute_Error'], pred_data['Normalized_Error'],
                             alpha=0.5, label=predictor, color=color, s=30)

            ax12.set_title('Absolute vs Normalized Time Error',
                           fontweight='bold', fontsize=12)
            ax12.set_xlabel('Absolute Time Error (seconds)', fontweight='bold')
            ax12.set_ylabel(
                'Normalized Error (relative to POI1)', fontweight='bold')
            ax12.legend()
            ax12.grid(True, alpha=0.3)

        fig.suptitle('POI Predictor Benchmark - Normalized Time Analysis',
                     fontsize=16, fontweight='bold', y=0.997)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")

        plt.show()

    def generate_report(self, index_df: pd.DataFrame, time_df: pd.DataFrame,
                        normalized_df: pd.DataFrame, output_path: Optional[str] = None):
        """Generate a detailed text report."""
        report = []
        report.append("="*80)
        report.append(
            "POI PREDICTOR BENCHMARK REPORT - NORMALIZED TIME ANALYSIS")
        report.append("="*80)
        report.append("")

        # Index-based Metrics
        report.append("INDEX-BASED METRICS (samples)")
        report.append("-"*80)
        report.append(index_df.to_string(index=False))
        report.append("")

        # Absolute Time Metrics
        report.append("ABSOLUTE TIME METRICS (seconds)")
        report.append("-"*80)
        report.append(time_df.to_string(index=False))
        report.append("")

        # Normalized Time Metrics
        report.append("NORMALIZED TIME METRICS (relative to POI1)")
        report.append("-"*80)
        report.append(normalized_df.to_string(index=False))
        report.append("")

        # Summary
        report.append("SUMMARY")
        report.append("-"*80)

        for predictor_name in ['V4_Fusion', 'V4_Predictor']:
            pred_index = index_df[index_df['Predictor'] == predictor_name]
            pred_time = time_df[time_df['Predictor'] == predictor_name]
            pred_norm = normalized_df[normalized_df['Predictor']
                                      == predictor_name]
            pred_results = [
                r for r in self.results if r.predictor_name == predictor_name]

            report.append(f"\n{predictor_name}:")
            report.append(f"  Index-based Metrics:")
            report.append(
                f"    Average MAE: {pred_index['MAE_Index'].mean():.2f} samples")
            report.append(
                f"    Average RMSE: {pred_index['RMSE_Index'].mean():.2f} samples")
            report.append(
                f"    Detection Rate: {pred_index['Detection_Rate'].mean():.2%}")

            report.append(f"  Absolute Time Metrics:")
            report.append(
                f"    Average MAE: {pred_time['MAE_Time'].mean():.4f} seconds")
            report.append(
                f"    Average RMSE: {pred_time['RMSE_Time'].mean():.4f} seconds")

            report.append(f"  Normalized Time Metrics:")
            report.append(
                f"    Average MAE: {pred_norm['MAE_Normalized'].mean():.4f}")
            report.append(
                f"    Average RMSE: {pred_norm['RMSE_Normalized'].mean():.4f}")

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
    FUSION_CLF_PATH = "v4_model_pytorch_.pth"

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
    index_df, time_df, normalized_df = benchmark.compute_metrics()

    # Generate report
    benchmark.generate_report(
        index_df,
        time_df,
        normalized_df,
        output_path="benchmark_report_normalized.txt"
    )

    # Plot results
    print("\nGenerating plots...")
    benchmark.plot_results(
        index_df,
        time_df,
        normalized_df,
        save_path="benchmark_results_normalized.png"
    )

    # Save metrics to CSV
    index_df.to_csv("benchmark_index_metrics.csv", index=False)
    time_df.to_csv("benchmark_time_metrics.csv", index=False)
    normalized_df.to_csv("benchmark_normalized_metrics.csv", index=False)
    print("\nMetrics saved to CSV files")

    print("\n✓ Benchmark complete!")
