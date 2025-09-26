import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import warnings

# Import the existing predictors
from v4_clf import POIPredictor as ClassificationPredictor
from v4_reg import RegressionPOIPredictor

warnings.filterwarnings('ignore')


class HybridPOIPredictor:
    """
    Hybrid predictor that combines classification and regression approaches.
    Uses classification for approximate multi-POI detection, then refines positions
    using regression models for precise localization.
    """

    def __init__(self,
                 classification_model_path: str,
                 classification_scaler_path: str,
                 classification_config_path: str,
                 regression_model_paths: Dict[str, str],
                 device: str = None):
        """
        Initialize the hybrid predictor with both classification and regression models.

        Args:
            classification_model_path: Path to classification model
            classification_scaler_path: Path to classification scaler
            classification_config_path: Path to classification config
            regression_model_paths: Dict mapping POI names to regression model paths
                                   e.g., {'POI1': 'model_poi1.pth', 'POI2': 'model_poi2.pth'}
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        print("="*60)
        print("INITIALIZING HYBRID POI PREDICTOR")
        print("="*60)

        # Load classification predictor
        print("\nLoading classification model...")
        self.clf_predictor = ClassificationPredictor(
            model_path=classification_model_path,
            scaler_path=classification_scaler_path,
            config_path=classification_config_path,
            device=device
        )

        # Load regression predictors for each POI
        print("\nLoading regression models...")
        self.reg_predictors = {}
        for poi_name, model_path in regression_model_paths.items():
            print(f"  Loading {poi_name} regression model...")
            self.reg_predictors[poi_name] = RegressionPOIPredictor(
                model_path=model_path,
                poi_type=poi_name,
                device=device
            )

        print("\n✓ Hybrid predictor initialized successfully")

    def predict(self,
                data_path: str = None,
                df: pd.DataFrame = None,
                window_margin: int = 100,
                use_regression_threshold: float = 0.5,
                enforce_constraints: bool = True) -> Dict[str, any]:
        """
        Perform hybrid prediction combining both approaches.

        Args:
            data_path: Path to CSV file (if df not provided)
            df: DataFrame with sensor data (if data_path not provided)
            window_margin: Margin around classified POI to search for regression peak
            use_regression_threshold: Minimum confidence to use regression refinement
            enforce_constraints: Whether to apply sequential/temporal constraints

        Returns:
            Dictionary containing:
                - 'final_positions': Dict of POI numbers and their final positions
                - 'methods_used': Dict indicating which method was used for each POI
                - 'classification_results': Original classification results
                - 'regression_refinements': Regression results for refined POIs
                - 'confidence_scores': Confidence for each POI
                - 'dataframe': The processed dataframe
        """
        # Load data if needed
        if data_path is not None:
            df = pd.read_csv(data_path)
            print(f"\nLoaded data: {len(df)} samples")
        elif df is None:
            raise ValueError("Either data_path or df must be provided")

        print("\n" + "="*60)
        print("PERFORMING HYBRID PREDICTION")
        print("="*60)

        # Step 1: Get classification predictions
        print("\nStep 1: Running classification model...")
        clf_results = self.clf_predictor.predict(
            df=df,
            enforce_constraints=enforce_constraints
        )

        clf_positions = clf_results['poi_locations']
        clf_probabilities = clf_results['probabilities']
        window_positions = clf_results['window_positions']

        print(
            f"Classification found {len(clf_positions)} POIs: {list(clf_positions.keys())}")

        # Step 2: Refine positions using regression
        print("\nStep 2: Refining positions with regression models...")
        final_positions = {}
        methods_used = {}
        regression_refinements = {}
        confidence_scores = {}

        print("\nStep 2: Running regression models on full dataset...")
        all_regression_results = {}
        poi_map = {1: 'POI1', 2: 'POI2', 4: 'POI4', 5: 'POI5', 6: 'POI6'}

        # First, run all regression models on the full data
        for poi_name, predictor in self.reg_predictors.items():
            print(f"  Running {poi_name} regression model...")
            reg_result = predictor.predict(
                df=df,
                apply_smoothing=True
            )
            all_regression_results[poi_name] = reg_result

        # Step 3: Match regression peaks with classification windows
        print("\nStep 3: Matching regression peaks with classification windows...")
        final_positions = {}
        methods_used = {}
        confidence_scores = {}

        for poi_num, clf_position in clf_positions.items():
            poi_name = poi_map.get(poi_num)

            if poi_name and poi_name in all_regression_results:
                reg_result = all_regression_results[poi_name]

                # Check ALL peaks to find one within the window
                best_peak = None
                best_distance = float('inf')

                for peak_pos, peak_conf in reg_result.get('all_peaks', []):
                    distance = abs(peak_pos - clf_position)
                    if distance <= window_margin and peak_conf >= use_regression_threshold:
                        if distance < best_distance:
                            best_peak = (peak_pos, peak_conf)
                            best_distance = distance

                if best_peak:
                    final_positions[poi_num] = best_peak[0]
                    methods_used[poi_num] = 'regression'
                    confidence_scores[poi_num] = best_peak[1]
                else:
                    # Fall back to classification
                    final_positions[poi_num] = clf_position
                    methods_used[poi_num] = 'classification'

                # Get confidence from classification
                poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
                if clf_probabilities is not None and poi_num in poi_indices:
                    closest_window_idx = np.argmin(
                        np.abs(np.array(window_positions) - clf_position))
                    confidence_scores[poi_num] = float(
                        clf_probabilities[closest_window_idx, poi_indices[poi_num]])
                else:
                    confidence_scores[poi_num] = 0.5

                print(
                    f"\n  {poi_name}: No regression model available, using classification")
        # Create binary output
        poi_binary = np.zeros(5)
        for poi_num in [1, 2, 4, 5, 6]:
            if poi_num in final_positions:
                idx = {1: 0, 2: 1, 4: 2, 5: 3, 6: 4}[poi_num]
                poi_binary[idx] = 1

        # Print summary
        print("\n" + "="*60)
        print("HYBRID PREDICTION RESULTS")
        print("="*60)
        print(f"Total POIs detected: {len(final_positions)}")
        print(
            f"Binary output [POI1, POI2, POI4, POI5, POI6]: {poi_binary.astype(int)}")
        print("\nFinal positions:")
        for poi_num in sorted(final_positions.keys()):
            method = methods_used[poi_num]
            confidence = confidence_scores.get(poi_num, 0)
            print(f"  POI-{poi_num}: Position {final_positions[poi_num]:5d} "
                  f"(Method: {method:14s}, Confidence: {confidence:.3f})")

        return {
            'final_positions': final_positions,
            'methods_used': methods_used,
            'classification_results': clf_results,
            'regression_refinements': regression_refinements,
            'confidence_scores': confidence_scores,
            'poi_binary': poi_binary,
            'poi_count': len(final_positions),
            'dataframe': df
        }

    def visualize(self,
                  prediction_result: Dict,
                  save_path: str = None,
                  show_plot: bool = True,
                  figsize: Tuple[int, int] = (16, 14)) -> None:
        """
        Visualize hybrid prediction results showing both methods.

        Args:
            prediction_result: Output from predict() method
            save_path: Path to save the figure
            show_plot: Whether to display the plot
            figsize: Figure size
        """
        df = prediction_result['dataframe']
        final_positions = prediction_result['final_positions']
        methods_used = prediction_result['methods_used']
        clf_results = prediction_result['classification_results']
        confidence_scores = prediction_result['confidence_scores']

        # Determine x-axis
        use_time = 'Relative_time' in df.columns
        x_values = df['Relative_time'].values if use_time else np.arange(
            len(df))
        x_label = 'Relative Time' if use_time else 'Sample Index'

        # Create figure with 5 subplots
        fig, axes = plt.subplots(5, 1, figsize=figsize, sharex=True)

        # Colors for POIs
        poi_colors = {1: 'red', 2: 'orange', 4: 'gold', 5: 'green', 6: 'blue'}

        # Plot 1: Dissipation with final POI positions
        if 'Dissipation' in df.columns:
            axes[0].plot(x_values, df['Dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
        elif 'dissipation' in df.columns:
            axes[0].plot(x_values, df['dissipation'].values,
                         'b-', alpha=0.7, lw=0.5)
        axes[0].set_ylabel('Dissipation')
        axes[0].set_title('Hybrid POI Detection - Final Results')
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Resonance Frequency
        if 'Resonance_Frequency' in df.columns:
            axes[1].plot(x_values, df['Resonance_Frequency'].values,
                         'g-', alpha=0.7, lw=0.5)
            axes[1].set_ylabel('Resonance Frequency')
            axes[1].grid(True, alpha=0.3)

        # Plot 3: Classification probabilities
        if clf_results['probabilities'] is not None:
            window_positions = clf_results['window_positions']
            probabilities = clf_results['probabilities']

            # Convert window positions to x-axis values
            window_x_values = []
            for pos in window_positions:
                if use_time and pos < len(df):
                    window_x_values.append(df['Relative_time'].iloc[pos])
                else:
                    window_x_values.append(pos)

            poi_indices = {1: 1, 2: 2, 4: 3, 5: 4, 6: 5}
            for poi_num, idx in poi_indices.items():
                axes[2].plot(window_x_values, probabilities[:, idx],
                             color=poi_colors[poi_num], alpha=0.6,
                             label=f'POI-{poi_num}', lw=1)

            axes[2].set_ylabel('Classification Prob.')
            axes[2].legend(loc='upper right', ncol=5, fontsize=8)
            axes[2].grid(True, alpha=0.3)
            axes[2].axhline(0.5, color='black', linestyle=':', alpha=0.5)

        # Plot 4: Method comparison
        axes[3].set_ylabel('Detection Method')
        axes[3].set_ylim([-0.5, 1.5])
        axes[3].set_yticks([0, 1])
        axes[3].set_yticklabels(['Classification', 'Regression'])
        axes[3].grid(True, alpha=0.3)

        # Plot 5: Confidence scores
        axes[4].set_ylabel('Confidence')
        axes[4].set_ylim([0, 1.1])
        axes[4].grid(True, alpha=0.3)
        axes[4].set_xlabel(x_label)

        # Add POI markers and information
        for poi_num, position in final_positions.items():
            method = methods_used[poi_num]
            confidence = confidence_scores.get(poi_num, 0)
            color = poi_colors.get(poi_num, 'black')

            # Get x position
            if use_time and position < len(df):
                x_pos = df['Relative_time'].iloc[position]
            else:
                x_pos = position

            # Add vertical lines on all plots
            for ax in axes[:3]:
                ax.axvline(x_pos, color=color, linestyle='--' if method == 'classification' else '-',
                           alpha=0.6, linewidth=1.5)

            # Add shaded region on dissipation plot
            half_win = 64
            start_idx = max(0, position - half_win)
            end_idx = min(len(df) - 1, position + half_win)

            if use_time:
                x_start = df['Relative_time'].iloc[start_idx]
                x_end = df['Relative_time'].iloc[end_idx]
            else:
                x_start, x_end = start_idx, end_idx

            axes[0].axvspan(x_start, x_end, color=color, alpha=0.1)

            # Add label on top plot
            axes[0].text(x_pos, axes[0].get_ylim()[1] * 0.95,
                         f'POI{poi_num}\n{method[0].upper()}',
                         ha='center', va='top', fontsize=8,
                         bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

            # Plot method indicator
            y_val = 1 if method == 'regression' else 0
            axes[3].scatter(x_pos, y_val, color=color, s=100, zorder=5)
            axes[3].text(x_pos, y_val + 0.15, f'POI{poi_num}',
                         ha='center', fontsize=8, color=color)

            # Plot confidence
            axes[4].bar(x_pos, confidence, width=20 if use_time else 50,
                        color=color, alpha=0.6, label=f'POI{poi_num}')
            axes[4].text(x_pos, confidence + 0.02, f'{confidence:.2f}',
                         ha='center', fontsize=8)

        # Add legends
        axes[0].text(0.02, 0.98, 'Solid line = Regression refined\nDashed line = Classification only',
                     transform=axes[0].transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")

        if show_plot:
            plt.show()

        # Print position comparison
        print("\n" + "="*60)
        print("POSITION COMPARISON")
        print("="*60)
        clf_positions = clf_results['poi_locations']
        for poi_num in sorted(final_positions.keys()):
            if poi_num in clf_positions:
                clf_pos = clf_positions[poi_num]
                final_pos = final_positions[poi_num]
                diff = final_pos - clf_pos
                method = methods_used[poi_num]

                print(f"POI-{poi_num}:")
                print(f"  Classification: {clf_pos:5d}")
                print(f"  Final position: {final_pos:5d}")
                print(f"  Difference:     {diff:+5d}")
                print(f"  Method used:    {method}")


# Example usage
if __name__ == "__main__":
    # Define paths to models
    classification_paths = {
        'model': 'QModel/SavedModels/qmodel_v4_fusion/v4_model_pytorch.pth',
        'scaler': 'QModel/SavedModels/qmodel_v4_fusion/v4_scaler_pytorch.joblib',
        'config': 'QModel/SavedModels/qmodel_v4_fusion/v4_config_pytorch.json'
    }

    regression_paths = {
        'POI1': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_mini_window_0.pth',
        'POI2': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_small_window_1.pth',
        'POI4': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_med_window_3.pth',
        'POI5': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_large_window_4.pth',
        'POI6': 'QModel/SavedModels/qmodel_v4_fusion/poi_model_small_window_5.pth'
    }

    # Initialize hybrid predictor
    predictor = HybridPOIPredictor(
        classification_model_path=classification_paths['model'],
        classification_scaler_path=classification_paths['scaler'],
        classification_config_path=classification_paths['config'],
        regression_model_paths=regression_paths
    )
    from v4_dp import DP
    content = DP.load_content('content/PROTEIN')
    for data, poi in content:
        result = predictor.predict(
            data_path=data,
            window_margin=128,  # Search within ±100 points of classification result
            use_regression_threshold=0.25,  # Minimum confidence to use regression
            enforce_constraints=True
        )

        # Visualize results
        predictor.visualize(result)

        # Access results
        print(f"\nFinal POI positions: {result['final_positions']}")
        print(f"Methods used: {result['methods_used']}")
        print(f"Confidence scores: {result['confidence_scores']}")
