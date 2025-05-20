"""
Drift monitoring module for continuous monitoring of data and model drift.
This module provides tools to monitor drift over time and trigger alerts.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from src.monitoring.drift_detector import AdvancedDriftDetector
from src.utils.data_utils import load_processed_data, load_features_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DriftMonitor:
    """
    Drift monitor for continuous monitoring of data and model drift.
    
    This class provides methods to monitor drift over time and trigger alerts
    when significant drift is detected.
    """
    
    def __init__(
        self,
        reference_data_path: Optional[str] = None,
        monitoring_config_path: Optional[str] = None,
        email_config_path: Optional[str] = None
    ):
        """
        Initialize the drift monitor.
        
        Parameters
        ----------
        reference_data_path : str, optional
            Path to reference data, by default None
        monitoring_config_path : str, optional
            Path to monitoring configuration, by default None
        email_config_path : str, optional
            Path to email configuration, by default None
        """
        self.reference_data = None
        if reference_data_path and os.path.exists(reference_data_path):
            try:
                self.reference_data = pd.read_parquet(reference_data_path)
                logger.info(f"Loaded reference data from {reference_data_path}")
            except Exception as e:
                logger.error(f"Error loading reference data: {e}")
        
        # Default monitoring configuration
        self.monitoring_config = {
            'check_frequency': 'daily',  # 'hourly', 'daily', 'weekly'
            'drift_thresholds': {
                'ks_test': 0.05,
                'js_divergence': 0.1,
                'wasserstein': 0.2,
                'psi': 0.2
            },
            'alert_on_drift': True,
            'auto_retrain_on_drift': False,
            'commodities': ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
            'categorical_features': []
        }
        
        # Load monitoring configuration if provided
        if monitoring_config_path and os.path.exists(monitoring_config_path):
            try:
                with open(monitoring_config_path, 'r') as f:
                    config = json.load(f)
                    self.monitoring_config.update(config)
                logger.info(f"Loaded monitoring configuration from {monitoring_config_path}")
            except Exception as e:
                logger.error(f"Error loading monitoring configuration: {e}")
        
        # Default email configuration
        self.email_config = {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'recipients': [],
            'subject_prefix': '[DRIFT ALERT]'
        }
        
        # Load email configuration if provided
        if email_config_path and os.path.exists(email_config_path):
            try:
                with open(email_config_path, 'r') as f:
                    config = json.load(f)
                    self.email_config.update(config)
                logger.info(f"Loaded email configuration from {email_config_path}")
            except Exception as e:
                logger.error(f"Error loading email configuration: {e}")
        
        # Initialize drift detector
        self.drift_detector = AdvancedDriftDetector(
            reference_data=self.reference_data,
            drift_metrics=['ks_test', 'js_divergence', 'wasserstein', 'psi'],
            thresholds=self.monitoring_config['drift_thresholds']
        )
        
        # Create directories for reports
        os.makedirs('reports/drift', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Last check timestamp
        self.last_check = None
    
    def check_drift(
        self,
        commodity: str,
        current_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
        generate_report: bool = True
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check for drift in a specific commodity.
        
        Parameters
        ----------
        commodity : str
            Name of the commodity to check
        current_data : pd.DataFrame, optional
            Current data to compare against reference data, by default None (will load from processed data)
        feature_names : List[str], optional
            List of feature names to check for drift, by default None (all columns)
        categorical_features : List[str], optional
            List of categorical features, by default None
        generate_report : bool, optional
            Whether to generate a drift report, by default True
            
        Returns
        -------
        Tuple[bool, Dict[str, Any]]
            A tuple containing:
            - Boolean indicating if significant drift was detected
            - Dictionary with detailed drift results
        """
        # Load current data if not provided
        if current_data is None:
            current_data = load_processed_data(commodity)
            
            if current_data.empty:
                logger.error(f"No data found for commodity {commodity}")
                return False, {"error": f"No data found for commodity {commodity}"}
        
        # Load reference data if not already loaded
        if self.reference_data is None:
            # Try to load features data as reference
            self.reference_data = load_features_data(commodity)
            
            if self.reference_data.empty:
                logger.error(f"No reference data found for commodity {commodity}")
                return False, {"error": f"No reference data found for commodity {commodity}"}
            
            # Update drift detector with reference data
            self.drift_detector.reference_data = self.reference_data
        
        # Use provided categorical features or from config
        if categorical_features is None:
            categorical_features = self.monitoring_config.get('categorical_features', [])
        
        # Check for drift
        has_drift, drift_results = self.drift_detector.detect_drift(
            current_data,
            feature_names=feature_names,
            categorical_features=categorical_features
        )
        
        # Generate report if requested
        if generate_report and has_drift:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = f"reports/drift/{commodity}_drift_report_{timestamp}.html"
            
            report = self.drift_detector.generate_drift_report(
                current_data,
                report_path=report_path,
                feature_names=feature_names,
                categorical_features=categorical_features
            )
            
            # Send alert if configured
            if has_drift and self.monitoring_config.get('alert_on_drift', True):
                self._send_drift_alert(commodity, drift_results, report_path)
            
            # Trigger auto-retraining if configured
            if has_drift and self.monitoring_config.get('auto_retrain_on_drift', False):
                self._trigger_retraining(commodity, drift_results)
        
        # Update last check timestamp
        self.last_check = datetime.now()
        
        return has_drift, drift_results
    
    def run_scheduled_check(self) -> Dict[str, Any]:
        """
        Run scheduled drift checks for all commodities.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with drift check results for all commodities
        """
        # Check if it's time to run the check based on frequency
        if self.last_check is not None:
            frequency = self.monitoring_config.get('check_frequency', 'daily')
            
            if frequency == 'hourly':
                if datetime.now() - self.last_check < timedelta(hours=1):
                    logger.info("Skipping scheduled check (less than 1 hour since last check)")
                    return {"status": "skipped", "reason": "Too soon since last check"}
            
            elif frequency == 'daily':
                if datetime.now() - self.last_check < timedelta(days=1):
                    logger.info("Skipping scheduled check (less than 1 day since last check)")
                    return {"status": "skipped", "reason": "Too soon since last check"}
            
            elif frequency == 'weekly':
                if datetime.now() - self.last_check < timedelta(weeks=1):
                    logger.info("Skipping scheduled check (less than 1 week since last check)")
                    return {"status": "skipped", "reason": "Too soon since last check"}
        
        # Run checks for all commodities
        commodities = self.monitoring_config.get('commodities', ['crude_oil'])
        results = {}
        
        for commodity in commodities:
            logger.info(f"Running drift check for {commodity}")
            
            try:
                has_drift, drift_results = self.check_drift(commodity)
                results[commodity] = {
                    "has_drift": has_drift,
                    "drift_score": drift_results.get('overall_drift_score', 0),
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"Drift check for {commodity}: {'Drift detected' if has_drift else 'No drift detected'}")
            
            except Exception as e:
                logger.error(f"Error checking drift for {commodity}: {e}")
                results[commodity] = {"error": str(e)}
        
        return results
    
    def _send_drift_alert(self, commodity: str, drift_results: Dict[str, Any], report_path: str) -> bool:
        """
        Send drift alert via email.
        
        Parameters
        ----------
        commodity : str
            Name of the commodity
        drift_results : Dict[str, Any]
            Drift detection results
        report_path : str
            Path to the drift report
            
        Returns
        -------
        bool
            Whether the alert was sent successfully
        """
        if not self.email_config.get('enabled', False):
            logger.info("Email alerts are disabled")
            return False
        
        if not self.email_config.get('recipients'):
            logger.warning("No email recipients configured")
            return False
        
        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            msg['Subject'] = f"{self.email_config['subject_prefix']} Drift Detected in {commodity.replace('_', ' ').title()}"
            
            # Email body
            body = f"""
            <html>
            <body>
                <h2>Drift Alert: {commodity.replace('_', ' ').title()}</h2>
                <p>Significant drift has been detected in the {commodity} data.</p>
                <p>Overall Drift Score: {drift_results.get('overall_drift_score', 'N/A')}</p>
                <p>Please review the attached drift report for details.</p>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach drift report
            if os.path.exists(report_path):
                with open(report_path, 'rb') as f:
                    attachment = MIMEApplication(f.read(), _subtype='html')
                    attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(report_path))
                    msg.attach(attachment)
            
            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
            
            logger.info(f"Drift alert sent for {commodity}")
            return True
        
        except Exception as e:
            logger.error(f"Error sending drift alert: {e}")
            return False
    
    def _trigger_retraining(self, commodity: str, drift_results: Dict[str, Any]) -> bool:
        """
        Trigger model retraining.
        
        Parameters
        ----------
        commodity : str
            Name of the commodity
        drift_results : Dict[str, Any]
            Drift detection results
            
        Returns
        -------
        bool
            Whether retraining was triggered successfully
        """
        # This is a placeholder for the actual retraining logic
        # In a real implementation, this would trigger a retraining pipeline
        logger.info(f"Triggering retraining for {commodity} due to drift")
        
        # TODO: Implement actual retraining logic
        
        return True
