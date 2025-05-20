"""
Script to run drift monitoring as a scheduled task.
This script can be run periodically to check for data and model drift.
"""

import os
import sys
import logging
import json
import argparse
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.monitoring.drift_monitor import DriftMonitor
from src.utils.data_utils import load_processed_data, load_features_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/drift_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run drift monitoring')
    
    parser.add_argument('--config', type=str, default='config/monitoring_config.json',
                        help='Path to monitoring configuration file')
    
    parser.add_argument('--email-config', type=str, default='config/email_config.json',
                        help='Path to email configuration file')
    
    parser.add_argument('--reference-data', type=str, default=None,
                        help='Path to reference data file')
    
    parser.add_argument('--commodities', nargs='+', default=None,
                        help='List of commodities to check for drift')
    
    parser.add_argument('--force', action='store_true',
                        help='Force drift check even if it\'s too soon since the last check')
    
    parser.add_argument('--report-dir', type=str, default='reports/drift',
                        help='Directory to save drift reports')
    
    return parser.parse_args()

def main():
    """Run drift monitoring."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs('config', exist_ok=True)
    
    # Create default monitoring configuration if it doesn't exist
    if not os.path.exists(args.config):
        default_config = {
            'check_frequency': 'daily',
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
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.info(f"Created default monitoring configuration at {args.config}")
    
    # Create default email configuration if it doesn't exist
    if not os.path.exists(args.email_config):
        default_email_config = {
            'enabled': False,
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'recipients': [],
            'subject_prefix': '[DRIFT ALERT]'
        }
        
        os.makedirs(os.path.dirname(args.email_config), exist_ok=True)
        
        with open(args.email_config, 'w') as f:
            json.dump(default_email_config, f, indent=4)
        
        logger.info(f"Created default email configuration at {args.email_config}")
    
    # Initialize drift monitor
    drift_monitor = DriftMonitor(
        reference_data_path=args.reference_data,
        monitoring_config_path=args.config,
        email_config_path=args.email_config
    )
    
    # Override commodities if specified
    if args.commodities:
        drift_monitor.monitoring_config['commodities'] = args.commodities
    
    # Run drift checks
    if args.force:
        # Reset last check timestamp to force a check
        drift_monitor.last_check = None
    
    results = drift_monitor.run_scheduled_check()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.report_dir, f"drift_check_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Drift check results saved to {results_path}")
    
    # Print summary
    print("\nDrift Monitoring Summary:")
    print("========================")
    
    if results.get('status') == 'skipped':
        print(f"Check skipped: {results.get('reason')}")
    else:
        for commodity, result in results.items():
            if 'error' in result:
                print(f"{commodity}: ERROR - {result['error']}")
            else:
                drift_status = "DRIFT DETECTED" if result.get('has_drift', False) else "No drift"
                drift_score = result.get('drift_score', 'N/A')
                print(f"{commodity}: {drift_status} (Score: {drift_score})")
    
    print("\nComplete. Check logs for details.")

if __name__ == "__main__":
    main()
