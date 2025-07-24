import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from scipy import stats
from scipy.stats import bootstrap, jarque_bera, shapiro, anderson
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class AnalysisConfig:
    """Configuration for the analysis"""
    max_runs: int = np.inf
    want_pois: Tuple[int, ...] = (1, 2, 4, 5, 6)
    bootstrap_samples: int = 10_000
    confidence_level: float = 0.95
    random_seed: int = 42
    outlier_threshold: float = 3.0  # IQR multiplier for outlier detection
    min_sample_size: int = 3
    plot_dpi: int = 300

# ----------------- Data loading and validation -----------------


def validate_data_structure(data_dir: str) -> bool:
    """Validate that the data directory exists and contains expected files"""
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return False

    csv_files = []
    for root, _, files in os.walk(data_dir):
        csv_files.extend([f for f in files if f.endswith(
            '.csv') and not f.endswith(('_poi.csv', '_lower.csv'))])

    if not csv_files:
        logger.error(f"No CSV files found in {data_dir}")
        return False

    logger.info(f"Found {len(csv_files)} potential data files")
    return True


def load_all_runs(data_dir: str, config: AnalysisConfig) -> List[Tuple[str, str]]:
    """Load all valid run pairs with improved error handling"""
    if not validate_data_structure(data_dir):
        return []

    pairs = []
    invalid_pairs = 0

    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.endswith('.csv') and not f.endswith(('_poi.csv', '_lower.csv')):
                data_path = os.path.join(root, f)
                poi_path = os.path.join(root, f.replace('.csv', '_poi.csv'))

                if os.path.exists(poi_path):
                    # Validate file readability
                    try:
                        pd.read_csv(data_path, nrows=1)
                        np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1)
                        pairs.append((data_path, poi_path))
                    except Exception as e:
                        logger.warning(
                            f"Skipping invalid pair {data_path}: {e}")
                        invalid_pairs += 1
                else:
                    invalid_pairs += 1

    if invalid_pairs > 0:
        logger.warning(f"Skipped {invalid_pairs} invalid file pairs")

    # Set random seed for reproducibility
    random.seed(config.random_seed)
    random.shuffle(pairs)

    max_runs = int(
        config.max_runs) if config.max_runs != np.inf else len(pairs)
    selected_pairs = pairs[:max_runs]

    logger.info(f"Selected {len(selected_pairs)} run pairs for analysis")
    return selected_pairs


def poi_times_from_idx(df: pd.DataFrame, poi_path: str) -> np.ndarray:
    """Extract POI times with better error handling"""
    try:
        idx = np.loadtxt(poi_path, delimiter=",", dtype=int, ndmin=1).ravel()

        # Handle both 0-based and 1-based indexing
        if idx.min() >= 1 and idx.max() > len(df):
            idx = idx - 1
        elif idx.max() >= len(df):
            raise ValueError(
                f"POI indices out of range: max={idx.max()}, df_len={len(df)}")

        if not all(0 <= i < len(df) for i in idx):
            raise ValueError("Some POI indices are out of bounds")

        times = df.loc[idx, "Relative_time"].to_numpy(float)

        if np.any(np.isnan(times)):
            raise ValueError("NaN values found in POI times")

        return times

    except Exception as e:
        logger.error(f"Error processing POI file {poi_path}: {e}")
        return np.array([])

# ----------------- Enhanced core analysis -----------------


def detect_outliers(data: np.ndarray, method: str = 'iqr', threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using various methods"""
    if len(data) < 4:
        return np.zeros(len(data), dtype=bool)

    if method == 'iqr':
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (data < lower_bound) | (data > upper_bound)

    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(data))
        return z_scores > threshold

    else:
        raise ValueError(f"Unknown outlier detection method: {method}")


def build_ratio_df(pairs: List[Tuple[str, str]], config: AnalysisConfig) -> pd.DataFrame:
    """
    Build ratio dataframe with enhanced validation and error handling
    """
    rows = []
    skipped_runs = {'insufficient_pois': 0, 'invalid_t': 0, 'read_error': 0}

    for data_csv, poi_csv in pairs:
        try:
            df = pd.read_csv(data_csv)

            # Validate required columns
            if "Relative_time" not in df.columns:
                logger.warning(f"Missing 'Relative_time' column in {data_csv}")
                skipped_runs['read_error'] += 1
                continue

            df = df.sort_values("Relative_time").reset_index(drop=True)
            times = poi_times_from_idx(df, poi_csv)

            if len(times) == 0:
                skipped_runs['read_error'] += 1
                continue

            if max(config.want_pois) > len(times):
                skipped_runs['insufficient_pois'] += 1
                continue

            p1, p2 = times[config.want_pois[0]-1], times[config.want_pois[1]-1]
            t = p2 - p1

            if t <= 0:
                skipped_runs['invalid_t'] += 1
                continue

            row = {
                "run": Path(data_csv).stem,
                "t": t,
                "poi1_time": p1,
                "poi2_time": p2
            }

            # Calculate gaps and ratios
            for k in config.want_pois[2:]:
                gap = times[k-1] - p1
                ratio = gap / t
                row[f"gap{k}"] = gap
                row[f"r{k}"] = ratio
                row[f"poi{k}_time"] = times[k-1]

            rows.append(row)

        except Exception as e:
            logger.error(f"Error processing {data_csv}: {e}")
            skipped_runs['read_error'] += 1
            continue

    # Log skipped runs
    total_skipped = sum(skipped_runs.values())
    if total_skipped > 0:
        logger.info(f"Skipped runs: {skipped_runs}")

    df_ratios = pd.DataFrame(rows)

    if df_ratios.empty:
        logger.error("No valid runs found!")
        return df_ratios

    # Clean infinite values and validate
    numeric_cols = [
        c for c in df_ratios.columns if c.startswith(('r', 'gap', 't'))]
    df_ratios[numeric_cols] = df_ratios[numeric_cols].replace(
        [np.inf, -np.inf], np.nan)

    logger.info(f"Successfully processed {len(df_ratios)} runs")
    return df_ratios


def enhanced_bootstrap_ci(x: np.ndarray, statistic=np.mean, n_resamples: int = 10_000,
                          confidence_level: float = 0.95, method: str = "bca",
                          seed: int = 42) -> Tuple[float, float]:
    """Enhanced bootstrap confidence intervals with bias-correction"""
    x = pd.to_numeric(
        pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)

    if x.size < 2:
        return (np.nan, np.nan)

    try:
        rng = np.random.default_rng(seed)
        res = bootstrap((x,), statistic, n_resamples=n_resamples,
                        method=method, confidence_level=confidence_level,
                        random_state=rng)
        return res.confidence_interval.low, res.confidence_interval.high
    except Exception as e:
        logger.warning(
            f"Bootstrap failed with method {method}, falling back to basic: {e}")
        try:
            res = bootstrap((x,), statistic, n_resamples=n_resamples,
                            method="basic", confidence_level=confidence_level,
                            random_state=rng)
            return res.confidence_interval.low, res.confidence_interval.high
        except Exception as e2:
            logger.error(f"Bootstrap completely failed: {e2}")
            return (np.nan, np.nan)


def test_normality(x: np.ndarray) -> Dict[str, Any]:
    """Test normality using multiple methods"""
    if len(x) < 3:
        return {"shapiro_p": np.nan, "jarque_bera_p": np.nan, "anderson_stat": np.nan}

    results = {}

    # Shapiro-Wilk test (good for small samples)
    if len(x) <= 5000:
        try:
            _, results["shapiro_p"] = shapiro(x)
        except Exception:
            results["shapiro_p"] = np.nan
    else:
        results["shapiro_p"] = np.nan

    # Jarque-Bera test
    try:
        _, results["jarque_bera_p"] = jarque_bera(x)
    except Exception:
        results["jarque_bera_p"] = np.nan

    # Anderson-Darling test
    try:
        ad_result = anderson(x)
        results["anderson_stat"] = ad_result.statistic
    except Exception:
        results["anderson_stat"] = np.nan

    return results


def summarize_ratios(df_ratios: pd.DataFrame, config: AnalysisConfig) -> pd.DataFrame:
    """Enhanced summary statistics with normality tests and robust estimators"""
    stats_list = []
    ratio_cols = [c for c in df_ratios.columns if c.startswith("r")]

    for col in ratio_cols:
        x = pd.to_numeric(
            df_ratios[col], errors="coerce").dropna().to_numpy(float)

        if x.size == 0:
            stats_list.append({
                "ratio": col, "n": 0, "mean": np.nan, "std": np.nan,
                "median": np.nan, "iqr": np.nan, "cv": np.nan,
                "ci95_low": np.nan, "ci95_high": np.nan,
                "skewness": np.nan, "kurtosis": np.nan,
                "mad": np.nan, "shapiro_p": np.nan, "jarque_bera_p": np.nan,
                "n_outliers": 0, "outlier_pct": 0.0
            })
            continue

        # Basic statistics
        mean_val = x.mean()
        std_val = x.std(ddof=1)
        median_val = np.median(x)
        q1, q3 = np.percentile(x, [25, 75])
        iqr_val = q3 - q1

        # Robust statistics
        mad_val = stats.median_abs_deviation(x)  # Median Absolute Deviation

        # Distribution shape
        skew_val = stats.skew(x)
        kurt_val = stats.kurtosis(x)

        # Confidence intervals
        ci_low, ci_high = enhanced_bootstrap_ci(x, n_resamples=config.bootstrap_samples,
                                                confidence_level=config.confidence_level)

        # Normality tests
        normality_results = test_normality(x)

        # Outlier detection
        outliers = detect_outliers(x, threshold=config.outlier_threshold)
        n_outliers = np.sum(outliers)
        outlier_pct = 100 * n_outliers / len(x)

        stats_list.append({
            "ratio": col,
            "n": x.size,
            "mean": mean_val,
            "std": std_val,
            "median": median_val,
            "iqr": iqr_val,
            "cv": std_val / mean_val if mean_val != 0 else np.nan,
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "skewness": skew_val,
            "kurtosis": kurt_val,
            "mad": mad_val,
            "shapiro_p": normality_results.get("shapiro_p", np.nan),
            "jarque_bera_p": normality_results.get("jarque_bera_p", np.nan),
            "n_outliers": n_outliers,
            "outlier_pct": outlier_pct
        })

    return pd.DataFrame(stats_list)


def robust_correlation_analysis(df_ratios: pd.DataFrame) -> pd.DataFrame:
    """Enhanced correlation analysis with robust methods"""
    rows = []
    t = pd.to_numeric(df_ratios["t"], errors="coerce").dropna()
    ratio_cols = [c for c in df_ratios.columns if c.startswith("r")]

    for col in ratio_cols:
        y = pd.to_numeric(df_ratios[col], errors="coerce")
        mask = t.notna() & y.notna()
        xv, yv = t[mask].to_numpy(), y[mask].to_numpy()

        if xv.size < 3:
            rows.append({
                "ratio": col, "n": xv.size,
                "pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan,
                "kendall_r": np.nan, "kendall_p": np.nan,
                "slope": np.nan, "intercept": np.nan, "stderr": np.nan
            })
            continue

        # Pearson correlation (parametric)
        pearson_r, pearson_p = stats.pearsonr(xv, yv)

        # Spearman correlation (non-parametric, rank-based)
        spearman_r, spearman_p = stats.spearmanr(xv, yv)

        # Kendall's tau (non-parametric, robust to outliers)
        kendall_r, kendall_p = stats.kendalltau(xv, yv)

        # Linear regression
        slope, intercept, _, _, stderr = stats.linregress(xv, yv)

        rows.append({
            "ratio": col,
            "n": xv.size,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "kendall_r": kendall_r,
            "kendall_p": kendall_p,
            "slope": slope,
            "intercept": intercept,
            "stderr": stderr
        })

    return pd.DataFrame(rows)

# ----------------- Enhanced plotting -----------------


def create_publication_ready_plots(df_ratios: pd.DataFrame, config: AnalysisConfig,
                                   outdir: str = "plots") -> None:
    """Create publication-ready plots with improved aesthetics and statistical annotations"""
    os.makedirs(outdir, exist_ok=True)

    ratio_cols = [c for c in df_ratios.columns if c.startswith("r")]
    ratio_cols = [c for c in ratio_cols
                  if pd.to_numeric(df_ratios[c], errors='coerce').dropna().size >= config.min_sample_size]

    if not ratio_cols:
        logger.warning(
            f"No ratios with >={config.min_sample_size} points. Skipping plots.")
        return

    # 1. Enhanced distribution plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Box plot with outliers
    ax1 = axes[0, 0]
    data_clean = []
    labels_clean = []
    for c in ratio_cols:
        values = pd.to_numeric(df_ratios[c], errors='coerce').dropna()
        if len(values) >= config.min_sample_size:
            data_clean.append(values)
            labels_clean.append(c)

    if data_clean:
        bp = ax1.boxplot(data_clean, labels=labels_clean, patch_artist=True,
                         showfliers=True, flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.6})
        colors = sns.color_palette("husl", len(data_clean))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

    ax1.set_ylabel("Ratio = (POIk - POI1) / t")
    ax1.set_title("Distribution of Ratios")
    ax1.grid(True, alpha=0.3)

    # Violin plot
    ax2 = axes[0, 1]
    if data_clean:
        parts = ax2.violinplot(data_clean, showmeans=True,
                               showextrema=True, showmedians=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        ax2.set_xticks(range(1, len(labels_clean) + 1))
        ax2.set_xticklabels(labels_clean)

    ax2.set_ylabel("Ratio = (POIk - POI1) / t")
    ax2.set_title("Distribution of Ratios")
    ax2.grid(True, alpha=0.3)

    # Q-Q plots for normality assessment
    ax3 = axes[1, 0]
    # Limit to 3 for clarity
    for i, (c, color) in enumerate(zip(ratio_cols[:3], colors)):
        values = pd.to_numeric(df_ratios[c], errors='coerce').dropna()
        if len(values) >= config.min_sample_size:
            stats.probplot(values, dist="norm", plot=ax3)
            ax3.get_lines()[-1].set_color(color)
            ax3.get_lines()[-1].set_label(c)

    ax3.set_title("Q-Q")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Histogram with density overlay
    ax4 = axes[1, 1]
    for c, color in zip(ratio_cols[:3], colors):
        values = pd.to_numeric(df_ratios[c], errors='coerce').dropna()
        if len(values) >= config.min_sample_size:
            ax4.hist(values, bins=20, alpha=0.5, density=True,
                     color=color, label=c, edgecolor='black')

    ax4.set_xlabel("Ratio Value")
    ax4.set_ylabel("Density")
    ax4.set_title("Histogram of Ratios")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "comprehensive_distributions.png"),
                dpi=config.plot_dpi, bbox_inches='tight')
    plt.close()

    # 2. Correlation plots with confidence intervals
    n_ratios = len(ratio_cols)
    fig, axes = plt.subplots(
        1, min(n_ratios, 3), figsize=(5*min(n_ratios, 3), 4))
    if n_ratios == 1:
        axes = [axes]

    for i, c in enumerate(ratio_cols[:3]):  # Limit to 3 subplots
        ax = axes[i] if len(axes) > 1 else axes[0]

        t_vals = pd.to_numeric(df_ratios["t"], errors='coerce')
        r_vals = pd.to_numeric(df_ratios[c], errors='coerce')
        mask = t_vals.notna() & r_vals.notna()

        x, y = t_vals[mask], r_vals[mask]

        if len(x) >= config.min_sample_size:
            # Scatter plot
            ax.scatter(x, y, alpha=0.6, s=30)

            # Regression line with confidence interval
            try:
                slope, intercept, r_val, p_val, stderr = stats.linregress(x, y)
                line_x = np.linspace(x.min(), x.max(), 100)
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r-', linewidth=2,
                        label=f'r={r_val:.3f}, p={p_val:.3f}')

                # Confidence band (approximate)
                se_line = stderr * \
                    np.sqrt(1/len(x) + (line_x - x.mean())
                            ** 2 / np.sum((x - x.mean())**2))
                ax.fill_between(line_x, line_y - 1.96*se_line, line_y + 1.96*se_line,
                                alpha=0.2, color='red')
                ax.legend()

            except Exception as e:
                logger.warning(f"Could not fit regression for {c}: {e}")

        ax.set_xlabel("t = POI2 - POI1 (s)")
        ax.set_ylabel(f"{c}")
        ax.set_title(f"{c} vs t")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "correlation_analysis.png"),
                dpi=config.plot_dpi, bbox_inches='tight')
    plt.close()

    logger.info(f"Enhanced plots saved to {outdir}")

# ----------------- Main analysis runner -----------------


def run_comprehensive_analysis(data_dir: str, config: AnalysisConfig, output_dir: str = ".") -> Dict[str, pd.DataFrame]:
    """Run comprehensive statistical analysis with all improvements"""

    logger.info("Starting comprehensive POI ratio analysis")

    # Set random seed for reproducibility
    np.random.seed(config.random_seed)

    # Load data
    pairs = load_all_runs(data_dir, config)
    if not pairs:
        logger.error("No valid data pairs found!")
        return {}

    # Build ratio dataframe
    df_ratios = build_ratio_df(pairs, config)
    if df_ratios.empty:
        logger.error("No valid ratios computed!")
        return {}

    # Clean and validate data
    numeric_cols = [
        c for c in df_ratios.columns if c.startswith(('r', 'gap', 't'))]
    for col in numeric_cols:
        df_ratios[col] = pd.to_numeric(df_ratios[col], errors='coerce')

    # Remove extreme outliers (optional - can be configured)
    initial_count = len(df_ratios)
    for col in [c for c in df_ratios.columns if c.startswith('r')]:
        outliers = detect_outliers(df_ratios[col].dropna(),
                                   threshold=config.outlier_threshold)
        if np.any(outliers):
            logger.info(f"Found {np.sum(outliers)} outliers in {col}")

    # Save raw data
    output_files = {}
    raw_file = os.path.join(output_dir, "poi_ratio_table_enhanced.csv")
    df_ratios.to_csv(raw_file, index=False)
    output_files['raw_data'] = df_ratios
    logger.info(f"Raw data saved to {raw_file}")

    # Statistical summaries
    df_summary = summarize_ratios(df_ratios, config)
    df_corr = robust_correlation_analysis(df_ratios)

    summary_file = os.path.join(output_dir, "poi_ratio_summary_enhanced.csv")
    corr_file = os.path.join(output_dir, "poi_ratio_correlation_enhanced.csv")

    df_summary.to_csv(summary_file, index=False)
    df_corr.to_csv(corr_file, index=False)

    output_files['summary'] = df_summary
    output_files['correlation'] = df_corr

    # Display results
    print("\n" + "="*60)
    print("RATIO SUMMARY")
    print("="*60)
    print(df_summary.round(4))

    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print(df_corr.round(4))

    # Create enhanced plots
    plot_dir = os.path.join(output_dir, "enhanced_plots")
    create_publication_ready_plots(df_ratios, config, plot_dir)

    logger.info("Comprehensive analysis completed successfully!")
    return output_files

# ----------------- Example usage -----------------


if __name__ == "__main__":
    # Configure analysis
    config = AnalysisConfig(
        max_runs=np.inf,
        want_pois=(1, 2, 4, 5, 6),
        bootstrap_samples=10_000,
        confidence_level=0.95,
        random_seed=42,
        outlier_threshold=3.0,
        min_sample_size=5,
        plot_dpi=300
    )

    # Run analysis
    results = run_comprehensive_analysis(
        data_dir="content/dropbox_dump",
        config=config,
        output_dir="."
    )
