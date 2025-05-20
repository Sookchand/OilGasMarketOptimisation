"""
Advanced drift detection module for monitoring data and model drift.
This module provides tools to detect and quantify drift between datasets.
"""

import os
import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Union, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedDriftDetector:
    """
    Advanced drift detector for monitoring data and model drift.

    This class provides methods to detect and quantify drift between datasets
    using multiple statistical tests and distance metrics.
    """

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        drift_metrics: Optional[List[str]] = None,
        thresholds: Optional[Dict[str, float]] = None,
        feature_importance: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the drift detector.

        Parameters
        ----------
        reference_data : pd.DataFrame, optional
            Reference data to compare against, by default None
        drift_metrics : List[str], optional
            List of drift metrics to calculate, by default ['ks_test', 'js_divergence', 'wasserstein']
        thresholds : Dict[str, float], optional
            Thresholds for each drift metric, by default None
        feature_importance : Dict[str, float], optional
            Importance of each feature for weighting drift detection, by default None
        """
        self.reference_data = reference_data
        self.drift_metrics = drift_metrics or ['ks_test', 'js_divergence', 'wasserstein']

        # Default thresholds
        self.thresholds = thresholds or {
            'ks_test': 0.05,  # p-value threshold
            'js_divergence': 0.1,  # Jensen-Shannon divergence threshold
            'wasserstein': 0.2,  # Wasserstein distance threshold
            'psi': 0.2  # Population Stability Index threshold
        }

        self.feature_importance = feature_importance
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)

        # Create directories for reports
        os.makedirs('reports/drift', exist_ok=True)

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect drift between reference and current data.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to compare against reference data
        feature_names : List[str], optional
            List of feature names to check for drift, by default None (all columns)
        categorical_features : List[str], optional
            List of categorical features, by default None

        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            A tuple containing:
            - Boolean indicating if significant drift was detected
            - Dictionary with detailed drift results
        """
        if self.reference_data is None:
            logger.info("No reference data available. Setting current data as reference.")
            self.reference_data = current_data
            return False, {}

        feature_names = feature_names or self.reference_data.columns
        categorical_features = categorical_features or []

        drift_results = {}
        significant_drift = False
        drift_scores = {}

        # Check for schema drift
        schema_drift = self._check_schema_drift(current_data)
        if schema_drift['has_drift']:
            logger.warning("Schema drift detected!")
            drift_results['schema_drift'] = schema_drift
            significant_drift = True

        # Check for data drift in each feature
        for feature in feature_names:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            # Handle categorical features differently
            is_categorical = feature in categorical_features

            # Calculate drift metrics
            metrics = {}
            feature_has_drift = False

            # Kolmogorov-Smirnov test (for numerical features)
            if 'ks_test' in self.drift_metrics and not is_categorical:
                try:
                    statistic, p_value = stats.ks_2samp(ref_values, cur_values)
                    metrics['ks_test'] = {'statistic': statistic, 'p_value': p_value}
                    if p_value < self.thresholds['ks_test']:
                        feature_has_drift = True
                except Exception as e:
                    logger.error(f"Error calculating KS test for {feature}: {e}")

            # Jensen-Shannon divergence
            if 'js_divergence' in self.drift_metrics:
                try:
                    js_div = self._calculate_js_divergence(ref_values, cur_values, is_categorical)
                    metrics['js_divergence'] = js_div
                    if js_div > self.thresholds['js_divergence']:
                        feature_has_drift = True
                except Exception as e:
                    logger.error(f"Error calculating JS divergence for {feature}: {e}")

            # Wasserstein distance (for numerical features)
            if 'wasserstein' in self.drift_metrics and not is_categorical:
                try:
                    w_dist = self._calculate_wasserstein(ref_values, cur_values)
                    metrics['wasserstein'] = w_dist
                    if w_dist > self.thresholds['wasserstein']:
                        feature_has_drift = True
                except Exception as e:
                    logger.error(f"Error calculating Wasserstein distance for {feature}: {e}")

            # Population Stability Index (PSI)
            if 'psi' in self.drift_metrics:
                try:
                    psi = self._calculate_psi(ref_values, cur_values, is_categorical)
                    metrics['psi'] = psi
                    if psi > self.thresholds['psi']:
                        feature_has_drift = True
                except Exception as e:
                    logger.error(f"Error calculating PSI for {feature}: {e}")

            # Store results for this feature
            drift_results[feature] = {
                'metrics': metrics,
                'has_drift': feature_has_drift
            }

            # Calculate overall drift score for this feature
            if feature_has_drift:
                # Weight by feature importance if available
                importance = 1.0
                if self.feature_importance and feature in self.feature_importance:
                    importance = self.feature_importance[feature]

                drift_scores[feature] = importance
                significant_drift = True

        # Calculate overall drift score
        if drift_scores:
            total_importance = sum(drift_scores.values())
            weighted_score = sum(drift_scores.values()) / total_importance if total_importance > 0 else 0
            drift_results['overall_drift_score'] = weighted_score
        else:
            drift_results['overall_drift_score'] = 0.0

        drift_results['has_significant_drift'] = significant_drift

        return significant_drift, drift_results

    def _check_schema_drift(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check for schema drift between reference and current data.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to check for schema drift

        Returns
        -------
        Dict[str, Any]
            Dictionary with schema drift results
        """
        ref_columns = set(self.reference_data.columns)
        cur_columns = set(current_data.columns)

        missing_columns = ref_columns - cur_columns
        new_columns = cur_columns - ref_columns

        has_drift = len(missing_columns) > 0 or len(new_columns) > 0

        return {
            'has_drift': has_drift,
            'missing_columns': list(missing_columns),
            'new_columns': list(new_columns)
        }

    def _calculate_js_divergence(
        self,
        dist1: pd.Series,
        dist2: pd.Series,
        is_categorical: bool = False
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.

        Parameters
        ----------
        dist1 : pd.Series
            First distribution
        dist2 : pd.Series
            Second distribution
        is_categorical : bool, optional
            Whether the distributions are categorical, by default False

        Returns
        -------
        float
            Jensen-Shannon divergence
        """
        if is_categorical:
            # For categorical features, calculate JS divergence based on frequency distributions
            freq1 = dist1.value_counts(normalize=True).to_dict()
            freq2 = dist2.value_counts(normalize=True).to_dict()

            # Get all unique categories
            all_categories = set(freq1.keys()) | set(freq2.keys())

            # Create probability distributions with zeros for missing categories
            p = np.array([freq1.get(cat, 0) for cat in all_categories])
            q = np.array([freq2.get(cat, 0) for cat in all_categories])
        else:
            # For numerical features, create histograms with the same bins
            min_val = min(dist1.min(), dist2.min())
            max_val = max(dist1.max(), dist2.max())

            bins = np.linspace(min_val, max_val, 20)

            hist1, _ = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)

            # Add small epsilon to avoid division by zero
            p = hist1 + 1e-10
            q = hist2 + 1e-10

            # Normalize
            p = p / p.sum()
            q = q / q.sum()

        # Calculate JS divergence
        m = 0.5 * (p + q)

        # Calculate KL divergence: KL(p||m) and KL(q||m)
        kl_pm = np.sum(p * np.log(p / m))
        kl_qm = np.sum(q * np.log(q / m))

        # JS divergence is the average of the two KL divergences
        js_divergence = 0.5 * (kl_pm + kl_qm)

        return js_divergence

    def _calculate_wasserstein(self, dist1: pd.Series, dist2: pd.Series) -> float:
        """
        Calculate Wasserstein distance between two distributions.

        Parameters
        ----------
        dist1 : pd.Series
            First distribution
        dist2 : pd.Series
            Second distribution

        Returns
        -------
        float
            Wasserstein distance
        """
        # Sort the distributions
        dist1_sorted = np.sort(dist1)
        dist2_sorted = np.sort(dist2)

        # Interpolate to make the distributions the same length
        n = 1000
        quantiles = np.linspace(0, 1, n)

        dist1_interp = np.quantile(dist1_sorted, quantiles)
        dist2_interp = np.quantile(dist2_sorted, quantiles)

        # Calculate Wasserstein distance (1-Wasserstein or Earth Mover's Distance)
        wasserstein = np.mean(np.abs(dist1_interp - dist2_interp))

        return wasserstein

    def _calculate_psi(
        self,
        dist1: pd.Series,
        dist2: pd.Series,
        is_categorical: bool = False
    ) -> float:
        """
        Calculate Population Stability Index (PSI) between two distributions.

        Parameters
        ----------
        dist1 : pd.Series
            First distribution (reference)
        dist2 : pd.Series
            Second distribution (current)
        is_categorical : bool, optional
            Whether the distributions are categorical, by default False

        Returns
        -------
        float
            Population Stability Index
        """
        if is_categorical:
            # For categorical features
            freq1 = dist1.value_counts(normalize=True).to_dict()
            freq2 = dist2.value_counts(normalize=True).to_dict()

            # Get all unique categories
            all_categories = set(freq1.keys()) | set(freq2.keys())

            # Calculate PSI
            psi = 0
            for cat in all_categories:
                p1 = freq1.get(cat, 0) + 1e-6  # Add small epsilon to avoid division by zero
                p2 = freq2.get(cat, 0) + 1e-6

                psi += (p2 - p1) * np.log(p2 / p1)

            return psi
        else:
            # For numerical features, create histograms with the same bins
            min_val = min(dist1.min(), dist2.min())
            max_val = max(dist1.max(), dist2.max())

            bins = np.linspace(min_val, max_val, 10)

            hist1, _ = np.histogram(dist1, bins=bins, density=True)
            hist2, _ = np.histogram(dist2, bins=bins, density=True)

            # Add small epsilon to avoid division by zero
            hist1 = hist1 + 1e-6
            hist2 = hist2 + 1e-6

            # Normalize
            hist1 = hist1 / hist1.sum()
            hist2 = hist2 / hist2.sum()

            # Calculate PSI
            psi = np.sum((hist2 - hist1) * np.log(hist2 / hist1))

            return psi

    def generate_drift_report(
        self,
        current_data: pd.DataFrame,
        report_path: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive drift report.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to compare against reference data
        report_path : str, optional
            Path to save the report, by default None
        feature_names : List[str], optional
            List of feature names to check for drift, by default None (all columns)
        categorical_features : List[str], optional
            List of categorical features, by default None

        Returns
        -------
        Dict[str, Any]
            Dictionary with drift report results
        """
        if self.reference_data is None:
            logger.warning("No reference data available. Cannot generate drift report.")
            return {"error": "No reference data available"}

        # Detect drift
        has_drift, drift_results = self.detect_drift(
            current_data,
            feature_names=feature_names,
            categorical_features=categorical_features
        )

        # Create report timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Default report path
        if report_path is None:
            report_path = f"reports/drift/drift_report_{timestamp}.html"

        # Generate visualizations
        visualizations = self._generate_visualizations(
            current_data,
            feature_names=feature_names,
            categorical_features=categorical_features
        )

        # Create report content
        report = {
            "timestamp": timestamp,
            "has_significant_drift": has_drift,
            "drift_results": drift_results,
            "visualizations": visualizations,
            "recommendations": self._generate_recommendations(drift_results)
        }

        # Save report
        self._save_report(report, report_path)

        return report

    def _generate_visualizations(
        self,
        current_data: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        Generate visualizations for drift report.

        Parameters
        ----------
        current_data : pd.DataFrame
            Current data to compare against reference data
        feature_names : List[str], optional
            List of feature names to check for drift, by default None (all columns)
        categorical_features : List[str], optional
            List of categorical features, by default None

        Returns
        -------
        Dict[str, str]
            Dictionary with visualization paths
        """
        if self.reference_data is None:
            return {}

        feature_names = feature_names or self.reference_data.columns
        categorical_features = categorical_features or []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(f"reports/drift/figures_{timestamp}", exist_ok=True)

        visualizations = {}

        # Distribution comparison for each feature
        for feature in feature_names:
            if feature not in self.reference_data.columns or feature not in current_data.columns:
                continue

            ref_values = self.reference_data[feature].dropna()
            cur_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(cur_values) == 0:
                continue

            is_categorical = feature in categorical_features

            # Create figure
            plt.figure(figsize=(10, 6))

            if is_categorical:
                # For categorical features, create bar plots
                ref_counts = ref_values.value_counts(normalize=True)
                cur_counts = cur_values.value_counts(normalize=True)

                # Get all categories
                all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))

                # Create DataFrame for plotting
                plot_data = pd.DataFrame({
                    'Reference': [ref_counts.get(cat, 0) for cat in all_categories],
                    'Current': [cur_counts.get(cat, 0) for cat in all_categories]
                }, index=all_categories)

                # Plot
                plot_data.plot(kind='bar', ax=plt.gca())
                plt.title(f"Distribution Comparison for {feature}")
                plt.ylabel("Frequency")
                plt.xlabel("Category")
                plt.xticks(rotation=45)
                plt.legend(["Reference", "Current"])
                plt.tight_layout()
            else:
                # For numerical features, create histograms
                plt.hist(ref_values, bins=30, alpha=0.5, label="Reference")
                plt.hist(cur_values, bins=30, alpha=0.5, label="Current")
                plt.title(f"Distribution Comparison for {feature}")
                plt.xlabel(feature)
                plt.ylabel("Frequency")
                plt.legend()
                plt.tight_layout()

            # Save figure
            fig_path = f"reports/drift/figures_{timestamp}/{feature}_distribution.png"
            plt.savefig(fig_path)
            plt.close()

            visualizations[f"{feature}_distribution"] = fig_path

        # PCA visualization for overall drift
        try:
            # Select common numerical features
            common_features = [f for f in feature_names
                              if f in self.reference_data.columns
                              and f in current_data.columns
                              and f not in categorical_features]

            if len(common_features) >= 2:
                # Extract data
                ref_data = self.reference_data[common_features].dropna()
                cur_data = current_data[common_features].dropna()

                if len(ref_data) > 0 and len(cur_data) > 0:
                    # Combine data for scaling
                    combined_data = pd.concat([ref_data, cur_data])

                    # Scale data
                    scaled_data = self.scaler.fit_transform(combined_data)

                    # Split back into reference and current
                    ref_scaled = scaled_data[:len(ref_data)]
                    cur_scaled = scaled_data[len(ref_data):]

                    # Apply PCA
                    pca = PCA(n_components=2)
                    combined_pca = pca.fit_transform(scaled_data)

                    # Split back into reference and current
                    ref_pca = combined_pca[:len(ref_data)]
                    cur_pca = combined_pca[len(ref_data):]

                    # Create figure
                    plt.figure(figsize=(10, 8))
                    plt.scatter(ref_pca[:, 0], ref_pca[:, 1], alpha=0.5, label="Reference")
                    plt.scatter(cur_pca[:, 0], cur_pca[:, 1], alpha=0.5, label="Current")
                    plt.title("PCA Visualization of Data Drift")
                    plt.xlabel("Principal Component 1")
                    plt.ylabel("Principal Component 2")
                    plt.legend()
                    plt.tight_layout()

                    # Save figure
                    fig_path = f"reports/drift/figures_{timestamp}/pca_visualization.png"
                    plt.savefig(fig_path)
                    plt.close()

                    visualizations["pca_visualization"] = fig_path
        except Exception as e:
            logger.error(f"Error generating PCA visualization: {e}")

        return visualizations

    def _generate_recommendations(self, drift_results: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on drift results.

        Parameters
        ----------
        drift_results : Dict[str, Any]
            Drift detection results

        Returns
        -------
        List[str]
            List of recommendations
        """
        recommendations = []

        if not drift_results.get('has_significant_drift', False):
            recommendations.append("No significant drift detected. Models can continue to be used without retraining.")
            return recommendations

        # General recommendation for significant drift
        recommendations.append("Significant drift detected. Consider retraining models with more recent data.")

        # Check for schema drift
        if 'schema_drift' in drift_results and drift_results['schema_drift']['has_drift']:
            schema_drift = drift_results['schema_drift']

            if schema_drift['missing_columns']:
                recommendations.append(
                    f"Schema drift detected: {len(schema_drift['missing_columns'])} columns are missing in the current data. "
                    f"Models depending on these features may not work correctly: {', '.join(schema_drift['missing_columns'])}."
                )

            if schema_drift['new_columns']:
                recommendations.append(
                    f"Schema drift detected: {len(schema_drift['new_columns'])} new columns found in the current data. "
                    f"Consider incorporating these features in model retraining: {', '.join(schema_drift['new_columns'])}."
                )

        # Find features with the most significant drift
        feature_drift = {}
        for feature, result in drift_results.items():
            if isinstance(result, dict) and 'has_drift' in result and result['has_drift']:
                # Calculate average drift score across metrics
                metrics = result.get('metrics', {})
                scores = []

                if 'ks_test' in metrics and 'p_value' in metrics['ks_test']:
                    # Convert p-value to a score (lower p-value = higher drift)
                    scores.append(1.0 - metrics['ks_test']['p_value'])

                if 'js_divergence' in metrics:
                    scores.append(metrics['js_divergence'])

                if 'wasserstein' in metrics:
                    scores.append(metrics['wasserstein'])

                if 'psi' in metrics:
                    scores.append(metrics['psi'])

                if scores:
                    feature_drift[feature] = sum(scores) / len(scores)

        # Sort features by drift score
        sorted_features = sorted(feature_drift.items(), key=lambda x: x[1], reverse=True)

        if sorted_features:
            top_features = sorted_features[:5]
            recommendations.append(
                f"The following features show the most significant drift and should be carefully examined: "
                f"{', '.join([f'{feature} (score: {score:.3f})' for feature, score in top_features])}."
            )

        # Recommendation for overall drift score
        if 'overall_drift_score' in drift_results:
            score = drift_results['overall_drift_score']
            if score > 0.5:
                recommendations.append(
                    f"Overall drift score is very high ({score:.3f}). Immediate model retraining is recommended."
                )
            elif score > 0.3:
                recommendations.append(
                    f"Overall drift score is moderate ({score:.3f}). Consider model retraining in the near future."
                )
            else:
                recommendations.append(
                    f"Overall drift score is relatively low ({score:.3f}). Monitor the situation but immediate action may not be necessary."
                )

        return recommendations

    def _save_report(self, report: Dict[str, Any], report_path: str) -> None:
        """
        Save drift report to file.

        Parameters
        ----------
        report : Dict[str, Any]
            Drift report
        report_path : str
            Path to save the report
        """
        try:
            # Create a simple HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Drift Detection Report - {report['timestamp']}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary {{ background-color: #f0f0f0; padding: 15px; border-radius: 5px; }}
                    .drift-detected {{ color: #d9534f; }}
                    .no-drift {{ color: #5cb85c; }}
                    .recommendations {{ background-color: #e8f4f8; padding: 15px; border-radius: 5px; }}
                    .visualization {{ margin: 20px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                </style>
            </head>
            <body>
                <h1>Drift Detection Report</h1>
                <p>Generated on: {report['timestamp']}</p>

                <div class="summary">
                    <h2>Summary</h2>
                    <p class="{'drift-detected' if report['has_significant_drift'] else 'no-drift'}">
                        {'Significant drift detected!' if report['has_significant_drift'] else 'No significant drift detected.'}
                    </p>
                    <p>Overall Drift Score: {report['drift_results'].get('overall_drift_score', 'N/A')}</p>
                </div>

                <div class="recommendations">
                    <h2>Recommendations</h2>
                    <ul>
                        {''.join([f'<li>{rec}</li>' for rec in report['recommendations']])}
                    </ul>
                </div>

                <h2>Visualizations</h2>
                <div class="visualizations">
                    {''.join([f'<div class="visualization"><h3>{key.replace("_", " ").title()}</h3><img src="{path}" alt="{key}" style="max-width:100%;"></div>' for key, path in report['visualizations'].items()])}
                </div>

                <h2>Detailed Drift Results</h2>
                <table>
                    <tr>
                        <th>Feature</th>
                        <th>Drift Detected</th>
                        <th>Metrics</th>
                    </tr>
                    {''.join([f'<tr><td>{feature}</td><td>{"Yes" if isinstance(result, dict) and result.get("has_drift", False) else "No"}</td><td>{str(result.get("metrics", {})) if isinstance(result, dict) else ""}</td></tr>' for feature, result in report['drift_results'].items() if feature not in ['has_significant_drift', 'overall_drift_score', 'schema_drift']])}
                </table>
            </body>
            </html>
            """

            # Save HTML report
            with open(report_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Drift report saved to {report_path}")
        except Exception as e:
            logger.error(f"Error saving drift report: {e}")