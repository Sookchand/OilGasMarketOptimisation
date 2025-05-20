"""
Script to demonstrate the usage of the Online Learning Framework.
This script updates models with new data if significant drift is detected.
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.models.online_learning import OnlineLearningManager, ModelRegistry, EvaluationMetrics, DriftDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/online_learning.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run online learning')
    
    parser.add_argument('--config', type=str, default='config/online_learning_config.json',
                        help='Path to online learning configuration file')
    
    parser.add_argument('--commodities', nargs='+', default=None,
                        help='List of commodities to update')
    
    parser.add_argument('--model-types', nargs='+', default=None,
                        help='List of model types to update')
    
    parser.add_argument('--force', action='store_true',
                        help='Force update regardless of drift')
    
    parser.add_argument('--registry-dir', type=str, default='models/registry',
                        help='Directory for model registry')
    
    return parser.parse_args()

def main():
    """Run online learning."""
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs('logs', exist_ok=True)
    os.makedirs('config', exist_ok=True)
    os.makedirs(args.registry_dir, exist_ok=True)
    
    # Create default configuration if it doesn't exist
    if not os.path.exists(args.config):
        default_config = {
            'update_frequency': 'daily',
            'drift_threshold': 0.05,
            'improvement_threshold': 5.0,
            'auto_update': True,
            'commodities': ['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
            'model_types': ['arima', 'xgboost', 'lstm', 'price_drivers'],
            'categorical_features': []
        }
        
        os.makedirs(os.path.dirname(args.config), exist_ok=True)
        
        with open(args.config, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        logger.info(f"Created default configuration at {args.config}")
    
    # Initialize components
    model_registry = ModelRegistry(registry_dir=args.registry_dir)
    evaluation_metrics = EvaluationMetrics()
    drift_detector = DriftDetector()
    
    # Initialize online learning manager
    online_learning_manager = OnlineLearningManager(
        model_registry=model_registry,
        evaluation_metrics=evaluation_metrics,
        drift_detector=drift_detector,
        config_path=args.config
    )
    
    # Update models
    results = online_learning_manager.update_models(
        commodities=args.commodities,
        model_types=args.model_types,
        force_update=args.force
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join('logs', f"online_learning_results_{timestamp}.json")
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Online learning results saved to {results_path}")
    
    # Print summary
    print("\nOnline Learning Summary:")
    print("=======================")
    
    if isinstance(results, dict) and 'status' in results:
        print(f"Status: {results['status']}")
        if 'reason' in results:
            print(f"Reason: {results['reason']}")
    else:
        for commodity, commodity_results in results.items():
            print(f"\n{commodity.replace('_', ' ').title()}:")
            
            if isinstance(commodity_results, dict) and 'status' in commodity_results:
                print(f"  Status: {commodity_results['status']}")
                if 'reason' in commodity_results:
                    print(f"  Reason: {commodity_results['reason']}")
            else:
                for model_type, model_results in commodity_results.items():
                    status = model_results.get('status', 'unknown')
                    
                    if status == 'updated':
                        version = model_results.get('version', 'unknown')
                        avg_improvement = model_results.get('avg_improvement', 0)
                        print(f"  {model_type}: UPDATED (version: {version}, improvement: {avg_improvement:.2f}%)")
                    elif status == 'skipped':
                        reason = model_results.get('reason', 'unknown')
                        print(f"  {model_type}: SKIPPED ({reason})")
                    elif status == 'error':
                        reason = model_results.get('reason', 'unknown')
                        print(f"  {model_type}: ERROR ({reason})")
                    else:
                        print(f"  {model_type}: {status}")
    
    print("\nComplete. Check logs for details.")

if __name__ == "__main__":
    main()
