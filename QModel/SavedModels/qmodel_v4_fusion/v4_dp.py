import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Union, Dict
from tqdm import tqdm
import numpy as np
import os
import random
from sklearn.svm import OneClassSVM
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from collections import defaultdict
import matplotlib.patches as mpatches


def visualize_balanced_dataset(dataset_poi_info: Dict[str, Dict],
                               loaded_content: List[Tuple[str, str]],
                               balance_method: str = 'position_bins',
                               save_path: str = None):
    """
    Create comprehensive visualization of the balanced dataset distribution.

    Args:
        dataset_poi_info: Dictionary containing all dataset information with POI statistics
        loaded_content: List of selected (balanced) dataset tuples
        balance_method: The balancing method used
        save_path: Optional path to save the figure
    """
    # Extract statistics for all datasets and selected datasets
    all_stats = extract_statistics(dataset_poi_info)
    selected_paths = set([path for path, _ in loaded_content])
    selected_stats = extract_statistics(
        {k: v for k, v in dataset_poi_info.items() if k in selected_paths})

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Balanced Dataset Distribution Analysis\nMethod: {balance_method.replace("_", " ").title()}\n'
                 f'Selected: {len(loaded_content)} / Total: {len(dataset_poi_info)}',
                 fontsize=14, fontweight='bold')

    # 1. POI Mean Distribution Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_distribution_comparison(ax1, all_stats['mean'], selected_stats['mean'],
                                 'POI Mean Position', 'Mean Position Value')

    # 2. POI Std Distribution Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_distribution_comparison(ax2, all_stats['std'], selected_stats['std'],
                                 'POI Standard Deviation', 'Std Value')

    # 3. POI Range Distribution Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    plot_distribution_comparison(ax3, all_stats['range'], selected_stats['range'],
                                 'POI Range (Max - Min)', 'Range Value')

    # 4. 2D Scatter: Mean vs Std
    ax4 = fig.add_subplot(gs[1, 0])
    plot_2d_scatter(ax4, all_stats, selected_stats, 'mean', 'std',
                    'POI Mean vs Standard Deviation')

    # 5. 2D Scatter: Mean vs Range
    ax5 = fig.add_subplot(gs[1, 1])
    plot_2d_scatter(ax5, all_stats, selected_stats, 'mean', 'range',
                    'POI Mean vs Range')

    # 6. Coverage Heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    plot_coverage_heatmap(ax6, all_stats, selected_stats)

    # 7. Cumulative Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    plot_cumulative_distribution(ax7, all_stats, selected_stats)

    # 8. Balance Quality Metrics
    ax8 = fig.add_subplot(gs[2, 1])
    plot_balance_metrics(ax8, all_stats, selected_stats)

    # 9. POI Position Boxplots
    ax9 = fig.add_subplot(gs[2, 2])
    plot_poi_boxplots(ax9, dataset_poi_info, selected_paths)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    return fig


def extract_statistics(dataset_poi_info: Dict[str, Dict]) -> Dict[str, np.ndarray]:
    """Extract statistics from dataset POI info."""
    stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'median': [],
        'range': [],
        'count': []
    }

    for info in dataset_poi_info.values():
        stats['mean'].append(info['poi_stats']['mean'])
        stats['std'].append(info['poi_stats']['std'])
        stats['min'].append(info['poi_stats']['min'])
        stats['max'].append(info['poi_stats']['max'])
        stats['median'].append(info['poi_stats']['median'])
        stats['range'].append(info['poi_stats']['max'] -
                              info['poi_stats']['min'])
        stats['count'].append(info['poi_stats']['count'])

    return {k: np.array(v) for k, v in stats.items()}


def plot_distribution_comparison(ax, all_data, selected_data, title, xlabel):
    """Plot histogram comparison of all vs selected data."""
    bins = np.linspace(min(all_data.min(), selected_data.min()),
                       max(all_data.max(), selected_data.max()), 20)

    ax.hist(all_data, bins=bins, alpha=0.5, label='All Datasets',
            color='gray', edgecolor='black', density=True)
    ax.hist(selected_data, bins=bins, alpha=0.7, label='Selected (Balanced)',
            color='blue', edgecolor='black', density=True)

    # Add KDE curves
    from scipy.stats import gaussian_kde
    kde_all = gaussian_kde(all_data)
    kde_selected = gaussian_kde(selected_data)
    x_range = np.linspace(bins[0], bins[-1], 100)
    ax.plot(x_range, kde_all(x_range), 'k--', alpha=0.5, label='All KDE')
    ax.plot(x_range, kde_selected(x_range), 'b-',
            alpha=0.8, label='Selected KDE')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_2d_scatter(ax, all_stats, selected_stats, x_key, y_key, title):
    """Plot 2D scatter comparison."""
    # Plot all datasets
    ax.scatter(all_stats[x_key], all_stats[y_key],
               c='lightgray', s=30, alpha=0.5, label='All Datasets')

    # Plot selected datasets
    ax.scatter(selected_stats[x_key], selected_stats[y_key],
               c='blue', s=40, alpha=0.8, label='Selected', edgecolors='darkblue')

    ax.set_xlabel(f'POI {x_key.capitalize()}')
    ax.set_ylabel(f'POI {y_key.capitalize()}')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_coverage_heatmap(ax, all_stats, selected_stats):
    """Plot coverage heatmap showing distribution coverage."""
    # Create 2D histogram for coverage analysis
    H_all, xedges, yedges = np.histogram2d(
        all_stats['mean'], all_stats['std'], bins=10)
    H_selected, _, _ = np.histogram2d(selected_stats['mean'], selected_stats['std'],
                                      bins=[xedges, yedges])

    # Calculate coverage ratio
    coverage = np.zeros_like(H_all)
    mask = H_all > 0
    coverage[mask] = H_selected[mask] / H_all[mask]

    im = ax.imshow(coverage.T, origin='lower', aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])

    ax.set_xlabel('POI Mean')
    ax.set_ylabel('POI Std')
    ax.set_title('Coverage Heatmap\n(Selected / All Ratio)', fontweight='bold')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coverage Ratio', rotation=270, labelpad=15)


def plot_cumulative_distribution(ax, all_stats, selected_stats):
    """Plot cumulative distribution functions."""
    # Sort values for CDF
    all_sorted = np.sort(all_stats['mean'])
    selected_sorted = np.sort(selected_stats['mean'])

    # Calculate CDF
    all_cdf = np.arange(1, len(all_sorted) + 1) / len(all_sorted)
    selected_cdf = np.arange(
        1, len(selected_sorted) + 1) / len(selected_sorted)

    ax.plot(all_sorted, all_cdf, 'gray', linewidth=2, label='All Datasets')
    ax.plot(selected_sorted, selected_cdf,
            'blue', linewidth=2, label='Selected')

    # Add KS statistic
    from scipy.stats import ks_2samp
    ks_stat, p_value = ks_2samp(all_stats['mean'], selected_stats['mean'])

    ax.text(0.05, 0.95, f'KS Statistic: {ks_stat:.3f}\np-value: {p_value:.3f}',
            transform=ax.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('POI Mean Position')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Comparison', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_balance_metrics(ax, all_stats, selected_stats):
    """Plot balance quality metrics."""
    metrics = {}

    # Calculate various balance metrics
    for stat_name in ['mean', 'std', 'range']:
        all_data = all_stats[stat_name]
        selected_data = selected_stats[stat_name]

        # Coefficient of variation
        cv_all = np.std(all_data) / \
            np.mean(all_data) if np.mean(all_data) != 0 else 0
        cv_selected = np.std(
            selected_data) / np.mean(selected_data) if np.mean(selected_data) != 0 else 0

        # Interquartile range ratio
        iqr_all = np.percentile(all_data, 75) - np.percentile(all_data, 25)
        iqr_selected = np.percentile(
            selected_data, 75) - np.percentile(selected_data, 25)

        metrics[stat_name] = {
            'CV Improvement': (cv_all - cv_selected) / cv_all * 100 if cv_all != 0 else 0,
            'IQR Ratio': iqr_selected / iqr_all if iqr_all != 0 else 1,
            'Coverage': len(np.unique(np.digitize(selected_data, np.percentile(all_data, [20, 40, 60, 80])))) / 5 * 100
        }

    # Plot metrics
    metric_names = list(metrics['mean'].keys())
    x = np.arange(len(metric_names))
    width = 0.25

    for i, stat in enumerate(['mean', 'std', 'range']):
        values = [metrics[stat][m] for m in metric_names]
        ax.bar(x + i * width, values, width, label=f'POI {stat.capitalize()}')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value (%)')
    ax.set_title('Balance Quality Metrics', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)


def plot_poi_boxplots(ax, dataset_poi_info, selected_paths):
    """Plot boxplots comparing POI distributions."""
    # Sample some datasets for visualization (max 20)
    sample_size = min(20, len(selected_paths))
    sampled_selected = np.random.choice(
        list(selected_paths), sample_size, replace=False)

    all_poi_positions = []
    selected_poi_positions = []

    # Collect POI positions
    for path, info in dataset_poi_info.items():
        if path in sampled_selected:
            selected_poi_positions.extend(info['poi_positions'])
        all_poi_positions.extend(info['poi_positions'])

    # Create boxplot
    bp = ax.boxplot([all_poi_positions, selected_poi_positions],
                    labels=['All Datasets', 'Selected (Balanced)'],
                    patch_artist=True, notch=True)

    # Color the boxes
    colors = ['lightgray', 'lightblue']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add mean markers
    means = [np.mean(all_poi_positions), np.mean(selected_poi_positions)]
    ax.plot([1, 2], means, 'r^', markersize=8, label='Mean')

    ax.set_ylabel('POI Position Values')
    ax.set_title('POI Position Distribution Summary', fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"All: μ={np.mean(all_poi_positions):.1f}, σ={np.std(all_poi_positions):.1f}\n"
    stats_text += f"Sel: μ={np.mean(selected_poi_positions):.1f}, σ={np.std(selected_poi_positions):.1f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


class DP:
    DROP = ["Date", "Time", "Ambient", "Peak Magnitude (RAW)", "Temperature"]
    BASELINE_WIN = 500
    ROLLING_WIN = 50

    @staticmethod
    def load_content(data_dir: str, num_datasets: Union[int, float] = np.inf,
                     balance_method: str = 'position_bins') -> List[Tuple[str, str]]:
        """
        Load datasets with balanced distribution of POI positions.

        Args:
            data_dir: Directory containing the datasets
            num_datasets: Maximum number of datasets to load
            balance_method: Method for balancing ('position_bins', 'clustering', or 'stratified')

        Returns:
            List of tuples containing (data_file_path, poi_file_path)
        """
        if not isinstance(data_dir, str) or not data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
        if not os.path.isdir(data_dir):
            raise ValueError(f"Directory '{data_dir}' does not exist.")

        # First, collect all valid datasets with their POI characteristics
        dataset_poi_info: Dict[str, Dict] = {}

        for root, _, files in tqdm(os.walk(data_dir), desc='Scanning files...'):
            for f in files:
                if f.endswith(".csv") and not f.endswith("_poi.csv") and not f.endswith("_lower.csv"):
                    poi_file = f.replace(".csv", "_poi.csv")
                    poi_path = os.path.join(root, poi_file)
                    if not os.path.exists(poi_path):
                        continue
                    try:
                        poi_df = pd.read_csv(poi_path, header=None)
                    except Exception as e:
                        continue

                    poi_values = poi_df.values.flatten()
                    if len(poi_values) != len(np.unique(poi_values)):
                        continue

                    # Store dataset info with POI characteristics
                    data_path = os.path.join(root, f)
                    dataset_poi_info[data_path] = {
                        'poi_path': poi_path,
                        'poi_positions': poi_values,
                        'poi_stats': {
                            'mean': np.mean(poi_values),
                            'std': np.std(poi_values),
                            'min': np.min(poi_values),
                            'max': np.max(poi_values),
                            'median': np.median(poi_values),
                            'count': len(poi_values)
                        }
                    }

        if not dataset_poi_info:
            return []

        # Apply balancing based on selected method
        if balance_method == 'position_bins':
            loaded_content = DP._balance_by_position_bins(
                dataset_poi_info, num_datasets)
        elif balance_method == 'clustering':
            loaded_content = DP._balance_by_clustering(
                dataset_poi_info, num_datasets)
        elif balance_method == 'stratified':
            loaded_content = DP._balance_by_stratified_sampling(
                dataset_poi_info, num_datasets)
        else:
            # Fallback to random sampling
            loaded_content = [(path, info['poi_path'])
                              for path, info in dataset_poi_info.items()]
            random.shuffle(loaded_content)
            if num_datasets != np.inf:
                loaded_content = loaded_content[:int(num_datasets)]
        return loaded_content

    @staticmethod
    def _balance_by_position_bins(dataset_poi_info: Dict[str, Dict],
                                  num_datasets: Union[int, float]) -> List[Tuple[str, str]]:
        """
        Balance datasets by binning POI positions into ranges.
        """
        # Determine global min and max POI positions
        all_positions = []
        for info in dataset_poi_info.values():
            all_positions.extend(info['poi_positions'])

        if not all_positions:
            return []

        global_min = min(all_positions)
        global_max = max(all_positions)

        # Create bins based on POI position ranges
        n_bins = min(10, len(dataset_poi_info))  # Adaptive number of bins
        bin_edges = np.linspace(global_min, global_max, n_bins + 1)

        # Assign datasets to bins based on their mean POI position
        bins = defaultdict(list)
        for data_path, info in dataset_poi_info.items():
            mean_pos = info['poi_stats']['mean']
            bin_idx = np.digitize(mean_pos, bin_edges) - 1
            bin_idx = min(bin_idx, n_bins - 1)  # Ensure within range
            bins[bin_idx].append((data_path, info['poi_path']))

        # Sample equally from each bin
        loaded_content = []
        if num_datasets == np.inf:
            samples_per_bin = max(len(b) for b in bins.values())
        else:
            samples_per_bin = int(num_datasets) // len(bins)
            remainder = int(num_datasets) % len(bins)

        for bin_idx, bin_data in sorted(bins.items()):
            random.shuffle(bin_data)
            if num_datasets == np.inf:
                loaded_content.extend(bin_data)
            else:
                n_samples = samples_per_bin
                if remainder > 0:
                    n_samples += 1
                    remainder -= 1
                loaded_content.extend(bin_data[:n_samples])

        random.shuffle(loaded_content)
        return loaded_content

    @staticmethod
    def _balance_by_clustering(dataset_poi_info: Dict[str, Dict],
                               num_datasets: Union[int, float]) -> List[Tuple[str, str]]:
        """
        Balance datasets using K-means clustering on POI statistics.
        """
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            print("sklearn not available, falling back to position bins method")
            return DP._balance_by_position_bins(dataset_poi_info, num_datasets)

        # Create feature matrix from POI statistics
        features = []
        dataset_paths = []
        for data_path, info in dataset_poi_info.items():
            features.append([
                info['poi_stats']['mean'],
                info['poi_stats']['std'],
                info['poi_stats']['median'],
                info['poi_stats']['max'] - info['poi_stats']['min'],  # range
                info['poi_stats']['count']
            ])
            dataset_paths.append((data_path, info['poi_path']))

        features = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Determine number of clusters
        n_clusters = min(int(np.sqrt(len(dataset_paths))), 20)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Group datasets by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(dataset_paths[idx])

        # Sample equally from each cluster
        loaded_content = []
        if num_datasets == np.inf:
            samples_per_cluster = max(len(c) for c in clusters.values())
        else:
            samples_per_cluster = int(num_datasets) // len(clusters)
            remainder = int(num_datasets) % len(clusters)

        for cluster_id, cluster_data in sorted(clusters.items()):
            random.shuffle(cluster_data)
            if num_datasets == np.inf:
                loaded_content.extend(cluster_data)
            else:
                n_samples = samples_per_cluster
                if remainder > 0:
                    n_samples += 1
                    remainder -= 1
                loaded_content.extend(cluster_data[:n_samples])

        random.shuffle(loaded_content)
        return loaded_content

    @staticmethod
    def _balance_by_stratified_sampling(dataset_poi_info: Dict[str, Dict],
                                        num_datasets: Union[int, float]) -> List[Tuple[str, str]]:
        """
        Balance datasets using stratified sampling based on POI distribution quartiles.
        """
        # Calculate quartiles for each statistic
        stats_quartiles = {
            'mean': [],
            'std': [],
            'range': []
        }

        for info in dataset_poi_info.values():
            stats_quartiles['mean'].append(info['poi_stats']['mean'])
            stats_quartiles['std'].append(info['poi_stats']['std'])
            stats_quartiles['range'].append(
                info['poi_stats']['max'] - info['poi_stats']['min'])

        # Calculate quartile boundaries
        quartile_bounds = {}
        for stat_name, values in stats_quartiles.items():
            quartile_bounds[stat_name] = np.percentile(values, [25, 50, 75])

        # Assign each dataset to a stratum based on its characteristics
        strata = defaultdict(list)
        for data_path, info in dataset_poi_info.items():
            # Create a stratum key based on quartiles
            stratum_key = []

            mean_val = info['poi_stats']['mean']
            std_val = info['poi_stats']['std']
            range_val = info['poi_stats']['max'] - info['poi_stats']['min']

            # Assign quartile for each statistic
            for stat_name, val in [('mean', mean_val), ('std', std_val), ('range', range_val)]:
                bounds = quartile_bounds[stat_name]
                if val <= bounds[0]:
                    stratum_key.append(0)
                elif val <= bounds[1]:
                    stratum_key.append(1)
                elif val <= bounds[2]:
                    stratum_key.append(2)
                else:
                    stratum_key.append(3)

            stratum_key = tuple(stratum_key)
            strata[stratum_key].append((data_path, info['poi_path']))

        # Sample from each stratum proportionally
        loaded_content = []
        total_datasets = sum(len(s) for s in strata.values())

        for stratum_key, stratum_data in strata.items():
            random.shuffle(stratum_data)
            if num_datasets == np.inf:
                loaded_content.extend(stratum_data)
            else:
                # Calculate proportional sample size
                proportion = len(stratum_data) / total_datasets
                n_samples = max(1, int(num_datasets * proportion))
                n_samples = min(n_samples, len(stratum_data))
                loaded_content.extend(stratum_data[:n_samples])

        # Adjust if we haven't reached the target number
        if num_datasets != np.inf and len(loaded_content) < int(num_datasets):
            remaining_data = []
            for stratum_data in strata.values():
                remaining_data.extend(
                    [d for d in stratum_data if d not in loaded_content])
            random.shuffle(remaining_data)
            n_additional = int(num_datasets) - len(loaded_content)
            loaded_content.extend(remaining_data[:n_additional])

        random.shuffle(loaded_content)
        return loaded_content[:int(num_datasets)] if num_datasets != np.inf else loaded_content

    @staticmethod
    def gen_features(df: pd.DataFrame):
        def compute_ocsvm_score(shift_series: pd.Series):
            if shift_series.empty:
                raise ValueError("shift_series is empty.")
            X = shift_series.values.reshape(-1, 1)
            ocsvm = OneClassSVM(nu=0.05, kernel='rbf',
                                gamma='scale', shrinking=False)
            ocsvm.fit(X)
            # get raw decision scores and flip so negative spikes become positive
            scores = ocsvm.decision_function(X)
            scores = -scores
            # baseline at zero
            scores = scores - np.min(scores)
            return scores

        def compute_DoG(df: pd.DataFrame, col: str, sigma: float = 2):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
            if sigma <= 0:
                raise ValueError("sigma must be a positive number.")
            result = gaussian_filter1d(df[col], sigma=sigma, order=1)
            return pd.Series(result, index=df.index)

        def compute_rolling_baseline_and_shift(dog_series: pd.Series, window: int):
            if window <= 0:
                raise ValueError("window must be a positive integer.")
            baseline = dog_series.rolling(
                window=window, center=True, min_periods=1).median()
            shift = dog_series - baseline
            return baseline, shift

        def compute_difference_curve(df: pd.DataFrame, difference_factor: int = 2) -> pd.Series:
            required_cols = ["Relative_time",
                             "Resonance_Frequency", "Dissipation"]
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(
                        f"Column '{col}' is missing from DataFrame.")

            xs = df["Relative_time"]

            i = next((x for x, t in enumerate(xs) if t > 0.5), 0)
            j = next((x for x, t in enumerate(xs) if t > 2.5), 1)
            if i == j:
                j = next((x for x, t in enumerate(
                    xs) if t > xs[j] + 2.0), j + 1)

            avg_res_freq = df["Resonance_Frequency"].iloc[i:j].mean()
            avg_diss = df["Dissipation"].iloc[i:j].mean()
            ys_diss = (df["Dissipation"] - avg_diss) * avg_res_freq / 2
            ys_freq = avg_res_freq - df["Resonance_Frequency"]
            return ys_freq - difference_factor * ys_diss

        df = df.copy()
        df = df.drop(columns=DP.DROP)
        smooth_win = int(
            0.005 * len(df["Relative_time"].values))
        if smooth_win % 2 == 0:
            smooth_win += 1
        if smooth_win <= 1:
            smooth_win = 2
        polyorder = 3
        # Rebaseline
        df["Difference"] = compute_difference_curve(df, difference_factor=2)
        base_d = df["Dissipation"].iloc[:DP.BASELINE_WIN].mean()
        base_rf = df["Resonance_Frequency"].iloc[:DP.BASELINE_WIN].mean()
        base_diff = df["Difference"].iloc[:DP.BASELINE_WIN].mean()
        df["Dissipation"] = df["Dissipation"] - base_d
        df["Difference"] = df["Difference"] - base_diff
        df["Resonance_Frequency"] = -(df["Resonance_Frequency"] - base_rf)

        # Smooth and compute difference
        df["Difference"] = savgol_filter(
            df["Difference"], smooth_win, polyorder)
        df["Dissipation"] = savgol_filter(
            df["Dissipation"], smooth_win, polyorder)
        df["Resonance_Frequency"] = savgol_filter(
            df["Resonance_Frequency"], smooth_win, polyorder)

        # `Dissipation` DoG processing
        df['Dissipation_DoG'] = compute_DoG(
            df, col='Dissipation')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Dissipation_DoG_baseline'], df['Dissipation_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Dissipation_DoG'], baseline_window
        )
        df['Dissipation_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Dissipation_DoG_shift'])

        # `Resonance_Frequency` DoG processing
        df['Resonance_Frequency_DoG'] = compute_DoG(
            df, col='Resonance_Frequency')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Resonance_Frequency_DoG_baseline'], df['Resonance_Frequency_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Resonance_Frequency_DoG'], baseline_window
        )
        df['Resonance_Frequency_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Resonance_Frequency_DoG_shift'])

        # `Difference` DoG processing
        df['Difference_DoG'] = compute_DoG(
            df, col='Difference')
        baseline_window = max(3, int(np.ceil(0.05 * len(df))))
        df['Difference_DoG_baseline'], df['Difference_DoG_shift'] = compute_rolling_baseline_and_shift(
            df['Difference_DoG'], baseline_window
        )
        df['Difference_DoG_SVM_Score'] = compute_ocsvm_score(
            df['Difference_DoG_shift'])

        # 1) Compute slopes as Δy/Δt
        dt = df['Relative_time'].diff().replace(0, np.nan)
        df['Diss_slope'] = df['Dissipation'].diff() / dt
        df['RF_slope'] = df['Resonance_Frequency'].diff() / dt
        df[['Diss_slope', 'RF_slope']] = df[[
            'Diss_slope', 'RF_slope']].fillna(0)

        # 2) Rolling aggregates for Dissipation
        df['Diss_roll_mean'] = df['Dissipation'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).mean()
        df['Diss_roll_std'] = df['Dissipation'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).std().fillna(0)
        df['Diss_roll_min'] = df['Dissipation'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).min()
        df['Diss_roll_max'] = df['Dissipation'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).max()

        # 3) Rolling aggregates for Resonance_Frequency
        df['RF_roll_mean'] = df['Resonance_Frequency'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).mean()
        df['RF_roll_std'] = df['Resonance_Frequency'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).std().fillna(0)
        df['RF_roll_min'] = df['Resonance_Frequency'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).min()
        df['RF_roll_max'] = df['Resonance_Frequency'].rolling(
            window=DP.ROLLING_WIN, min_periods=1).max()

        df['Diss_x_RF'] = df['Dissipation'] * df['Resonance_Frequency']
        df['slope_DxRF'] = df['Diss_slope'] * df['RF_slope']
        df['rollmean_DxrollRF'] = df['Diss_roll_mean'] * df['RF_roll_mean']
        df['Diss_over_RF'] = df['Dissipation'] / \
            (df['Resonance_Frequency'] + 1e-6)
        df['slope_ratio'] = df['Diss_slope'] / (df['RF_slope'] + 1e-6)
        df['rollstd_ratio'] = df['Diss_roll_std'] * \
            0  # placeholder if you compute RF_roll_std
        df['time_x_Diss'] = df['Relative_time'] * df['Dissipation']
        df['time_x_slope_sum'] = df['Relative_time'] * \
            (df['Diss_slope'] + df['RF_slope'])
        df['Diss_t_x_RF_t1'] = df['Dissipation'] * \
            df['Resonance_Frequency'].shift(1)
        df['Diss_t1_x_RF_t'] = df['Dissipation'].shift(
            1) * df['Resonance_Frequency']

        window = df  # or the slice of df you’re summarizing
        df['range_Dx_range_RF'] = (window['Dissipation'].max() - window['Dissipation'].min()) \
            * (window['Resonance_Frequency'].max() - window['Resonance_Frequency'].min())
        # if you compute area under the curve:
        df['area_Dx_area_RF'] = (
            window['Dissipation'].sum()) * (window['Resonance_Frequency'].sum())

        df.fillna(0, inplace=True)

        return df


if __name__ == "__main__":
    DP.load_content('content/static/train', num_datasets=300,
                    balance_method='stratified')
