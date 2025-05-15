"""
Trading pipeline for the Oil & Gas Market Optimization project.
This script orchestrates the trading and risk management pipeline.
"""

import os
import logging
import argparse
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import numpy as np

from src.trading.strategies.trend_following import MovingAverageCrossover, MACDStrategy
from src.trading.strategies.mean_reversion import RSIStrategy, BollingerBandsStrategy
from src.trading.strategies.volatility_breakout import DonchianChannelStrategy, ATRChannelStrategy
from src.trading.execution.backtester import run_backtest
from src.trading.execution.performance_metrics import calculate_returns_metrics, plot_performance
from src.risk.var_calculator import calculate_var, calculate_expected_shortfall
from src.risk.monte_carlo import run_monte_carlo_analysis
from src.risk.portfolio_optimizer import optimize_portfolio
from src.risk.risk_limits import create_risk_limits
from src.agentic_ai.market_analyzer_agent import run_market_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/trading_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define constants
PROCESSED_DATA_DIR = 'data/processed'
RESULTS_DIR = 'results/trading'

def load_commodity_data(
    commodity: str,
    data_dir: str = PROCESSED_DATA_DIR
) -> pd.DataFrame:
    """
    Load processed commodity data.
    
    Parameters
    ----------
    commodity : str
        Commodity name
    data_dir : str, optional
        Data directory, by default PROCESSED_DATA_DIR
    
    Returns
    -------
    pd.DataFrame
        Commodity data
    """
    file_path = os.path.join(data_dir, f"{commodity}.parquet")
    
    try:
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded {len(df)} rows for {commodity} from {file_path}")
            return df
        else:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

def prepare_data_for_trading(
    df: pd.DataFrame,
    price_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Prepare data for trading.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    price_col : str, optional
        Price column name, by default None (use first column)
    
    Returns
    -------
    pd.DataFrame
        Prepared data
    """
    # Make a copy of the data
    data = df.copy()
    
    # If price_col not specified, use the first column
    if price_col is None:
        price_col = data.columns[0]
    
    # Rename price column to 'close' for consistency
    data = data.rename(columns={price_col: 'close'})
    
    # Calculate OHLC if not available
    if 'open' not in data.columns:
        data['open'] = data['close'].shift(1)
    
    if 'high' not in data.columns:
        data['high'] = data['close']
    
    if 'low' not in data.columns:
        data['low'] = data['close']
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    
    # Drop NaN values
    data = data.dropna()
    
    return data

def create_strategy(
    strategy_type: str,
    **kwargs
) -> Any:
    """
    Create a trading strategy.
    
    Parameters
    ----------
    strategy_type : str
        Strategy type
    **kwargs : dict
        Strategy parameters
    
    Returns
    -------
    Any
        Trading strategy
    """
    if strategy_type == 'ma_crossover':
        return MovingAverageCrossover(**kwargs)
    elif strategy_type == 'macd':
        return MACDStrategy(**kwargs)
    elif strategy_type == 'rsi':
        return RSIStrategy(**kwargs)
    elif strategy_type == 'bollinger':
        return BollingerBandsStrategy(**kwargs)
    elif strategy_type == 'donchian':
        return DonchianChannelStrategy(**kwargs)
    elif strategy_type == 'atr':
        return ATRChannelStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

def run_trading_pipeline(
    commodity: str,
    strategy_type: str,
    strategy_params: Optional[Dict[str, Any]] = None,
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    risk_analysis: bool = True,
    monte_carlo_sims: int = 1000,
    save_results: bool = True,
    output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """
    Run the trading pipeline.
    
    Parameters
    ----------
    commodity : str
        Commodity name
    strategy_type : str
        Strategy type
    strategy_params : Dict[str, Any], optional
        Strategy parameters, by default None
    initial_capital : float, optional
        Initial capital, by default 10000.0
    commission : float, optional
        Commission rate, by default 0.001
    slippage : float, optional
        Slippage rate, by default 0.001
    risk_analysis : bool, optional
        Whether to perform risk analysis, by default True
    monte_carlo_sims : int, optional
        Number of Monte Carlo simulations, by default 1000
    save_results : bool, optional
        Whether to save results, by default True
    output_dir : str, optional
        Output directory, by default RESULTS_DIR
    
    Returns
    -------
    Dict[str, Any]
        Pipeline results
    """
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Load data
    df = load_commodity_data(commodity)
    
    if df.empty:
        logger.error(f"No data for {commodity}")
        return {}
    
    # Prepare data for trading
    data = prepare_data_for_trading(df)
    
    # Create strategy
    strategy_params = strategy_params or {}
    strategy = create_strategy(strategy_type, **strategy_params)
    
    logger.info(f"Created {strategy_type} strategy for {commodity}")
    
    # Run backtest
    results, metrics = run_backtest(
        strategy=strategy,
        data=data,
        price_col='close',
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage,
        save_results=save_results,
        output_dir=output_dir
    )
    
    logger.info(f"Backtest completed with metrics: {metrics}")
    
    # Initialize pipeline results
    pipeline_results = {
        'commodity': commodity,
        'strategy': strategy_type,
        'strategy_params': strategy_params,
        'backtest_metrics': metrics,
        'risk_metrics': {}
    }
    
    # Perform risk analysis
    if risk_analysis and not results.empty:
        # Calculate VaR
        var_results = calculate_var(results['net_returns'], method='all')
        pipeline_results['risk_metrics']['var'] = var_results
        
        # Calculate Expected Shortfall
        es = calculate_expected_shortfall(results['net_returns'])
        pipeline_results['risk_metrics']['expected_shortfall'] = es
        
        # Run Monte Carlo analysis
        if monte_carlo_sims > 0:
            mc_simulations, mc_stats = run_monte_carlo_analysis(
                returns=results['net_returns'],
                initial_value=initial_capital,
                num_simulations=monte_carlo_sims,
                save_plots=save_results,
                output_dir=output_dir
            )
            
            pipeline_results['risk_metrics']['monte_carlo'] = mc_stats
        
        logger.info(f"Risk analysis completed")
    
    # Save pipeline results
    if save_results:
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results to JSON
        results_file = os.path.join(output_dir, f"{commodity}_{strategy_type}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            # Convert numpy values to Python types
            json_results = json.dumps(pipeline_results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(json_results)
        
        logger.info(f"Saved pipeline results to {results_file}")
    
    return pipeline_results

def run_multi_strategy_pipeline(
    commodity: str,
    strategies: List[Dict[str, Any]],
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    risk_analysis: bool = True,
    save_results: bool = True,
    output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """
    Run the trading pipeline with multiple strategies.
    
    Parameters
    ----------
    commodity : str
        Commodity name
    strategies : List[Dict[str, Any]]
        List of strategy configurations
    initial_capital : float, optional
        Initial capital, by default 10000.0
    commission : float, optional
        Commission rate, by default 0.001
    slippage : float, optional
        Slippage rate, by default 0.001
    risk_analysis : bool, optional
        Whether to perform risk analysis, by default True
    save_results : bool, optional
        Whether to save results, by default True
    output_dir : str, optional
        Output directory, by default RESULTS_DIR
    
    Returns
    -------
    Dict[str, Any]
        Pipeline results
    """
    # Initialize results
    all_results = {
        'commodity': commodity,
        'strategies': {}
    }
    
    # Run pipeline for each strategy
    for strategy_config in strategies:
        strategy_type = strategy_config['type']
        strategy_params = strategy_config.get('params', {})
        
        logger.info(f"Running pipeline for {strategy_type} strategy")
        
        # Run pipeline
        results = run_trading_pipeline(
            commodity=commodity,
            strategy_type=strategy_type,
            strategy_params=strategy_params,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            risk_analysis=risk_analysis,
            save_results=save_results,
            output_dir=output_dir
        )
        
        # Add to results
        all_results['strategies'][strategy_type] = results
    
    # Find best strategy
    if all_results['strategies']:
        best_strategy = max(
            all_results['strategies'].keys(),
            key=lambda k: all_results['strategies'][k].get('backtest_metrics', {}).get('sharpe_ratio', -float('inf'))
        )
        
        all_results['best_strategy'] = best_strategy
        logger.info(f"Best strategy: {best_strategy}")
    
    # Save combined results
    if save_results:
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results to JSON
        results_file = os.path.join(output_dir, f"{commodity}_multi_strategy_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            # Convert numpy values to Python types
            json_results = json.dumps(all_results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(json_results)
        
        logger.info(f"Saved multi-strategy results to {results_file}")
    
    return all_results

def run_full_pipeline(
    commodities: List[str],
    strategies: List[Dict[str, Any]],
    initial_capital: float = 10000.0,
    commission: float = 0.001,
    slippage: float = 0.001,
    risk_analysis: bool = True,
    market_analysis: bool = True,
    save_results: bool = True,
    output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """
    Run the full pipeline for multiple commodities and strategies.
    
    Parameters
    ----------
    commodities : List[str]
        List of commodities
    strategies : List[Dict[str, Any]]
        List of strategy configurations
    initial_capital : float, optional
        Initial capital, by default 10000.0
    commission : float, optional
        Commission rate, by default 0.001
    slippage : float, optional
        Slippage rate, by default 0.001
    risk_analysis : bool, optional
        Whether to perform risk analysis, by default True
    market_analysis : bool, optional
        Whether to perform market analysis, by default True
    save_results : bool, optional
        Whether to save results, by default True
    output_dir : str, optional
        Output directory, by default RESULTS_DIR
    
    Returns
    -------
    Dict[str, Any]
        Pipeline results
    """
    # Initialize results
    all_results = {
        'commodities': {},
        'market_analysis': None
    }
    
    # Run pipeline for each commodity
    for commodity in commodities:
        logger.info(f"Running pipeline for {commodity}")
        
        # Run multi-strategy pipeline
        results = run_multi_strategy_pipeline(
            commodity=commodity,
            strategies=strategies,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            risk_analysis=risk_analysis,
            save_results=save_results,
            output_dir=output_dir
        )
        
        # Add to results
        all_results['commodities'][commodity] = results
    
    # Run market analysis
    if market_analysis:
        logger.info("Running market analysis")
        
        try:
            market_results = run_market_analysis()
            all_results['market_analysis'] = market_results
            
            logger.info("Market analysis completed")
            
        except Exception as e:
            logger.error(f"Error running market analysis: {e}")
    
    # Save combined results
    if save_results:
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results to JSON
        results_file = os.path.join(output_dir, f"full_pipeline_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            # Convert numpy values to Python types
            json_results = json.dumps(all_results, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)
            f.write(json_results)
        
        logger.info(f"Saved full pipeline results to {results_file}")
    
    return all_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the trading pipeline')
    
    parser.add_argument('--commodities', nargs='+', default=['crude_oil', 'regular_gasoline', 'conventional_gasoline', 'diesel'],
                        help='List of commodities to process')
    parser.add_argument('--strategy', choices=['ma_crossover', 'macd', 'rsi', 'bollinger', 'donchian', 'atr', 'all'],
                        default='all', help='Trading strategy to use')
    parser.add_argument('--initial-capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--slippage', type=float, default=0.001, help='Slippage rate')
    parser.add_argument('--no-risk-analysis', action='store_false', dest='risk_analysis',
                        help='Disable risk analysis')
    parser.add_argument('--no-market-analysis', action='store_false', dest='market_analysis',
                        help='Disable market analysis')
    parser.add_argument('--no-save-results', action='store_false', dest='save_results',
                        help='Disable saving results')
    parser.add_argument('--output-dir', default=RESULTS_DIR, help='Output directory')
    
    args = parser.parse_args()
    
    # Define strategies
    if args.strategy == 'all':
        strategies = [
            {'type': 'ma_crossover', 'params': {'fast_window': 10, 'slow_window': 50}},
            {'type': 'macd', 'params': {'fast_window': 12, 'slow_window': 26, 'signal_window': 9}},
            {'type': 'rsi', 'params': {'window': 14, 'oversold': 30, 'overbought': 70}},
            {'type': 'bollinger', 'params': {'window': 20, 'num_std': 2.0}},
            {'type': 'donchian', 'params': {'window': 20}},
            {'type': 'atr', 'params': {'window': 14, 'multiplier': 2.0}}
        ]
    else:
        strategies = [{'type': args.strategy, 'params': {}}]
    
    # Run full pipeline
    run_full_pipeline(
        commodities=args.commodities,
        strategies=strategies,
        initial_capital=args.initial_capital,
        commission=args.commission,
        slippage=args.slippage,
        risk_analysis=args.risk_analysis,
        market_analysis=args.market_analysis,
        save_results=args.save_results,
        output_dir=args.output_dir
    )
