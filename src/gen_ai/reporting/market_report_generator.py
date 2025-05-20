"""
Automated Market Report Generator.
This module provides tools to generate comprehensive market reports.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
import jinja2
import markdown
import base64
from io import BytesIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_report_generator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MarketReportGenerator:
    """
    Automated Market Report Generator.

    This class provides methods to generate comprehensive market reports
    with data visualizations, insights, and recommendations.
    """

    def __init__(
        self,
        template_dir: str = 'templates/reports',
        output_dir: str = 'reports/market',
        data_provider = None,
        model_provider = None,
        llm_provider = None
    ):
        """
        Initialize the market report generator.

        Parameters
        ----------
        template_dir : str, optional
            Directory containing report templates, by default 'templates/reports'
        output_dir : str, optional
            Directory to save generated reports, by default 'reports/market'
        data_provider : object, optional
            Provider for market data, by default None
        model_provider : object, optional
            Provider for forecasting models, by default None
        llm_provider : object, optional
            Provider for language model generation, by default None
        """
        self.template_dir = template_dir
        self.output_dir = output_dir
        self.data_provider = data_provider
        self.model_provider = model_provider
        self.llm_provider = llm_provider

        # Create directories if they don't exist
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs('logs', exist_ok=True)

        # Initialize Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

        # Create default template if it doesn't exist
        self._create_default_template()

    def _create_default_template(self) -> None:
        """Create default report template if it doesn't exist."""
        default_template_path = os.path.join(self.template_dir, 'daily_report.html')

        if not os.path.exists(default_template_path):
            default_template = """<!DOCTYPE html>
<html>
<head>
    <title>{{ title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ddd;
        }
        .section {
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #eee;
        }
        .chart {
            margin: 20px 0;
            text-align: center;
        }
        .chart img {
            max-width: 100%;
            height: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px 15px;
            border-bottom: 1px solid #ddd;
            text-align: left;
        }
        th {
            background-color: #f8f8f8;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        .neutral {
            color: orange;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            font-size: 0.8em;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on: {{ date }}</p>
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            {{ executive_summary|safe }}
        </div>

        {% for commodity in commodities %}
        <div class="section">
            <h2>{{ commodity.name }}</h2>

            <div class="chart">
                <h3>Price Trend</h3>
                <img src="data:image/png;base64,{{ commodity.price_chart }}" alt="{{ commodity.name }} Price Trend">
            </div>

            <h3>Market Analysis</h3>
            {{ commodity.analysis|safe }}

            <div class="chart">
                <h3>Price Forecast</h3>
                <img src="data:image/png;base64,{{ commodity.forecast_chart }}" alt="{{ commodity.name }} Price Forecast">
            </div>

            <h3>Key Metrics</h3>
            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Change</th>
                </tr>
                {% for metric in commodity.metrics %}
                <tr>
                    <td>{{ metric.name }}</td>
                    <td>{{ metric.value }}</td>
                    <td class="{{ metric.change_class }}">{{ metric.change }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endfor %}

        <div class="section">
            <h2>Trading Signals</h2>
            <table>
                <tr>
                    <th>Commodity</th>
                    <th>Signal</th>
                    <th>Strength</th>
                    <th>Timeframe</th>
                </tr>
                {% for signal in trading_signals %}
                <tr>
                    <td>{{ signal.commodity }}</td>
                    <td>{{ signal.signal }}</td>
                    <td>{{ signal.strength }}</td>
                    <td>{{ signal.timeframe }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>

        <div class="section">
            <h2>Risk Assessment</h2>
            {{ risk_assessment|safe }}

            <div class="chart">
                <h3>Risk Heatmap</h3>
                <img src="data:image/png;base64,{{ risk_heatmap }}" alt="Risk Heatmap">
            </div>
        </div>

        <div class="section">
            <h2>Market Insights</h2>
            {{ market_insights|safe }}
        </div>

        <div class="footer">
            <p>This report was automatically generated by the Oil & Gas Market Optimization System.</p>
            <p>© {{ current_year }} All rights reserved.</p>
        </div>
    </div>
</body>
</html>"""

            os.makedirs(os.path.dirname(default_template_path), exist_ok=True)

            with open(default_template_path, 'w') as f:
                f.write(default_template)

            logger.info(f"Created default report template at {default_template_path}")

    def generate_daily_report(
        self,
        commodities: List[str],
        date: Optional[datetime] = None,
        lookback_days: int = 30,
        forecast_days: int = 14,
        template_name: str = 'daily_report.html',
        output_format: str = 'html'
    ) -> str:
        """
        Generate a comprehensive daily market report.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to include in the report
        date : datetime, optional
            Date for the report, by default None (today)
        lookback_days : int, optional
            Number of days to look back for historical data, by default 30
        forecast_days : int, optional
            Number of days to forecast, by default 14
        template_name : str, optional
            Name of the template to use, by default 'daily_report.html'
        output_format : str, optional
            Output format ('html' or 'pdf'), by default 'html'

        Returns
        -------
        str
            Path to the generated report
        """
        # Set date to today if not provided
        if date is None:
            date = datetime.now()

        # Format date for display and filenames
        date_str = date.strftime("%Y-%m-%d")
        date_display = date.strftime("%B %d, %Y")

        # Gather data and insights
        market_data = self._gather_market_data(commodities, date, lookback_days)
        forecasts = self._generate_forecasts(commodities, market_data, forecast_days)
        trading_signals = self._analyze_trading_signals(commodities, market_data)
        risk_metrics = self._calculate_risk_metrics(commodities, market_data)

        # Generate natural language insights
        executive_summary = self._generate_executive_summary(market_data, forecasts, trading_signals)
        commodity_analyses = self._generate_commodity_analyses(commodities, market_data, forecasts)
        market_insights = self._generate_market_insights(commodities, market_data, forecasts)
        risk_assessment = self._generate_risk_assessment(commodities, risk_metrics)

        # Create visualizations
        commodity_charts = self._create_visualization_charts(commodities, market_data, forecasts)
        risk_heatmap = self._create_risk_heatmap(risk_metrics)

        # Prepare data for template
        template_data = {
            "title": f"Daily Market Report - {date_display}",
            "date": date_display,
            "executive_summary": executive_summary,
            "commodities": [],
            "trading_signals": trading_signals,
            "risk_assessment": risk_assessment,
            "risk_heatmap": risk_heatmap,
            "market_insights": market_insights,
            "current_year": datetime.now().year
        }

        # Add commodity data
        for commodity in commodities:
            commodity_data = {
                "name": commodity.replace('_', ' ').title(),
                "price_chart": commodity_charts.get(f"{commodity}_price", ""),
                "forecast_chart": commodity_charts.get(f"{commodity}_forecast", ""),
                "analysis": commodity_analyses.get(commodity, "No analysis available."),
                "metrics": self._format_metrics(commodity, market_data, forecasts)
            }

            template_data["commodities"].append(commodity_data)

        # Render template
        try:
            template = self.jinja_env.get_template(template_name)
            report_html = template.render(**template_data)

            # Save report
            output_path = os.path.join(self.output_dir, f"market_report_{date_str}.{output_format}")

            if output_format == 'html':
                with open(output_path, 'w') as f:
                    f.write(report_html)
            elif output_format == 'pdf':
                self._convert_html_to_pdf(report_html, output_path)

            logger.info(f"Generated market report saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error generating market report: {e}")
            return ""

    def _gather_market_data(
        self,
        commodities: List[str],
        date: datetime,
        lookback_days: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Gather market data for the specified commodities.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to gather data for
        date : datetime
            Date for the report
        lookback_days : int
            Number of days to look back for historical data

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        """
        market_data = {}

        # Calculate start date
        start_date = date - timedelta(days=lookback_days)

        for commodity in commodities:
            try:
                # If data provider is available, use it
                if self.data_provider:
                    data = self.data_provider.get_historical_data(
                        commodity=commodity,
                        start_date=start_date,
                        end_date=date
                    )
                else:
                    # Otherwise, generate sample data
                    data = self._generate_sample_data(commodity, start_date, date)

                market_data[commodity] = data
                logger.info(f"Gathered market data for {commodity}")

            except Exception as e:
                logger.error(f"Error gathering market data for {commodity}: {e}")
                # Generate sample data as fallback
                market_data[commodity] = self._generate_sample_data(commodity, start_date, date)

        return market_data

    def _generate_forecasts(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame],
        forecast_days: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate forecasts for the specified commodities.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to generate forecasts for
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecast_days : int
            Number of days to forecast

        Returns
        -------
        Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity
        """
        forecasts = {}

        for commodity in commodities:
            try:
                # If model provider is available, use it
                if self.model_provider:
                    forecast = self.model_provider.generate_forecast(
                        commodity=commodity,
                        data=market_data[commodity],
                        horizon=forecast_days
                    )
                else:
                    # Otherwise, generate sample forecast
                    forecast = self._generate_sample_forecast(commodity, market_data[commodity], forecast_days)

                forecasts[commodity] = forecast
                logger.info(f"Generated forecast for {commodity}")

            except Exception as e:
                logger.error(f"Error generating forecast for {commodity}: {e}")
                # Generate sample forecast as fallback
                forecasts[commodity] = self._generate_sample_forecast(commodity, market_data[commodity], forecast_days)

        return forecasts

    def _analyze_trading_signals(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        """
        Analyze trading signals for the specified commodities.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to analyze
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity

        Returns
        -------
        List[Dict[str, str]]
            List of trading signals
        """
        trading_signals = []

        for commodity in commodities:
            try:
                # Get market data
                data = market_data[commodity]

                if data.empty:
                    continue

                # Calculate simple moving averages
                data['SMA20'] = data['price'].rolling(window=20).mean()
                data['SMA50'] = data['price'].rolling(window=50).mean()

                # Get the most recent data point
                latest = data.iloc[-1]

                # Determine signal based on moving average crossover
                if latest['SMA20'] > latest['SMA50']:
                    signal = "BUY"
                    strength = "Strong" if (latest['SMA20'] / latest['SMA50'] - 1) > 0.05 else "Moderate"
                elif latest['SMA20'] < latest['SMA50']:
                    signal = "SELL"
                    strength = "Strong" if (1 - latest['SMA20'] / latest['SMA50']) > 0.05 else "Moderate"
                else:
                    signal = "HOLD"
                    strength = "Neutral"

                # Add signal to list
                trading_signals.append({
                    "commodity": commodity.replace('_', ' ').title(),
                    "signal": signal,
                    "strength": strength,
                    "timeframe": "Medium-term"
                })

                logger.info(f"Analyzed trading signal for {commodity}: {signal} ({strength})")

            except Exception as e:
                logger.error(f"Error analyzing trading signal for {commodity}: {e}")
                # Add default signal as fallback
                trading_signals.append({
                    "commodity": commodity.replace('_', ' ').title(),
                    "signal": "HOLD",
                    "strength": "Neutral",
                    "timeframe": "Medium-term"
                })

        return trading_signals

    def _calculate_risk_metrics(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk metrics for the specified commodities.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to calculate risk metrics for
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity

        Returns
        -------
        Dict[str, Dict[str, float]]
            Dictionary of risk metrics for each commodity
        """
        risk_metrics = {}

        for commodity in commodities:
            try:
                # Get market data
                data = market_data[commodity]

                if data.empty:
                    continue

                # Calculate daily returns
                data['return'] = data['price'].pct_change()

                # Calculate risk metrics
                volatility = data['return'].std() * np.sqrt(252)  # Annualized volatility
                var_95 = data['return'].quantile(0.05) * data['price'].iloc[-1]  # 95% VaR
                var_99 = data['return'].quantile(0.01) * data['price'].iloc[-1]  # 99% VaR

                # Calculate max drawdown
                data['cumulative_return'] = (1 + data['return']).cumprod()
                data['cumulative_max'] = data['cumulative_return'].cummax()
                data['drawdown'] = (data['cumulative_return'] / data['cumulative_max']) - 1
                max_drawdown = data['drawdown'].min()

                # Store risk metrics
                risk_metrics[commodity] = {
                    "volatility": volatility,
                    "var_95": var_95,
                    "var_99": var_99,
                    "max_drawdown": max_drawdown
                }

                logger.info(f"Calculated risk metrics for {commodity}")

            except Exception as e:
                logger.error(f"Error calculating risk metrics for {commodity}: {e}")
                # Add default risk metrics as fallback
                risk_metrics[commodity] = {
                    "volatility": 0.2,
                    "var_95": -0.02,
                    "var_99": -0.03,
                    "max_drawdown": -0.1
                }

        return risk_metrics

    def _generate_executive_summary(
        self,
        market_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame],
        trading_signals: List[Dict[str, str]]
    ) -> str:
        """
        Generate executive summary for the market report.

        Parameters
        ----------
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecasts : Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity
        trading_signals : List[Dict[str, str]]
            List of trading signals

        Returns
        -------
        str
            Executive summary in HTML format
        """
        try:
            # If LLM provider is available, use it
            if self.llm_provider:
                # Prepare context for LLM
                context = {
                    "market_data": {k: v.to_dict() for k, v in market_data.items()},
                    "forecasts": {k: v.to_dict() for k, v in forecasts.items()},
                    "trading_signals": trading_signals
                }

                # Generate summary using LLM
                summary = self.llm_provider.generate_text(
                    prompt="Generate an executive summary for the daily market report based on the provided data.",
                    context=context
                )

                return summary

            # Otherwise, generate a simple summary
            summary = "<p>This daily market report provides an overview of current market conditions, forecasts, and trading signals for key commodities.</p>"

            # Add summary of price movements
            summary += "<p>Key price movements:</p><ul>"

            for commodity, data in market_data.items():
                if data.empty:
                    continue

                # Calculate price change
                first_price = data['price'].iloc[0]
                last_price = data['price'].iloc[-1]
                price_change = (last_price / first_price - 1) * 100

                # Add to summary
                summary += f"<li><strong>{commodity.replace('_', ' ').title()}</strong>: "
                if price_change > 0:
                    summary += f"<span class='positive'>+{price_change:.2f}%</span>"
                else:
                    summary += f"<span class='negative'>{price_change:.2f}%</span>"
                summary += "</li>"

            summary += "</ul>"

            # Add summary of trading signals
            buy_signals = [s for s in trading_signals if s['signal'] == 'BUY']
            sell_signals = [s for s in trading_signals if s['signal'] == 'SELL']

            if buy_signals:
                summary += "<p>Buy signals detected for: "
                summary += ", ".join([s['commodity'] for s in buy_signals])
                summary += ".</p>"

            if sell_signals:
                summary += "<p>Sell signals detected for: "
                summary += ", ".join([s['commodity'] for s in sell_signals])
                summary += ".</p>"

            return summary

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "<p>Executive summary not available due to an error.</p>"

    def _generate_commodity_analyses(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame]
    ) -> Dict[str, str]:
        """
        Generate analyses for each commodity.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to analyze
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecasts : Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity

        Returns
        -------
        Dict[str, str]
            Dictionary of analyses for each commodity in HTML format
        """
        analyses = {}

        for commodity in commodities:
            try:
                # If LLM provider is available, use it
                if self.llm_provider:
                    # Prepare context for LLM
                    context = {
                        "commodity": commodity,
                        "market_data": market_data[commodity].to_dict() if commodity in market_data else {},
                        "forecast": forecasts[commodity].to_dict() if commodity in forecasts else {}
                    }

                    # Generate analysis using LLM
                    analysis = self.llm_provider.generate_text(
                        prompt=f"Generate a detailed market analysis for {commodity} based on the provided data.",
                        context=context
                    )

                    analyses[commodity] = analysis
                    continue

                # Otherwise, generate a simple analysis
                data = market_data.get(commodity, pd.DataFrame())
                forecast = forecasts.get(commodity, pd.DataFrame())

                if data.empty:
                    analyses[commodity] = "<p>No data available for analysis.</p>"
                    continue

                # Calculate price change
                first_price = data['price'].iloc[0]
                last_price = data['price'].iloc[-1]
                price_change = (last_price / first_price - 1) * 100

                # Calculate volatility
                returns = data['price'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility

                # Generate analysis
                analysis = f"<p>{commodity.replace('_', ' ').title()} prices have "

                if price_change > 5:
                    analysis += "increased significantly"
                elif price_change > 0:
                    analysis += "increased slightly"
                elif price_change < -5:
                    analysis += "decreased significantly"
                elif price_change < 0:
                    analysis += "decreased slightly"
                else:
                    analysis += "remained stable"

                analysis += f" by {abs(price_change):.2f}% over the analyzed period.</p>"

                # Add volatility analysis
                analysis += "<p>Market volatility is "

                if volatility > 0.3:
                    analysis += "very high"
                elif volatility > 0.2:
                    analysis += "high"
                elif volatility > 0.1:
                    analysis += "moderate"
                else:
                    analysis += "low"

                analysis += f" at {volatility:.2f} (annualized).</p>"

                # Add forecast analysis if available
                if not forecast.empty:
                    last_forecast = forecast['forecast'].iloc[-1]
                    forecast_change = (last_forecast / last_price - 1) * 100

                    analysis += "<p>The forecast indicates that prices are expected to "

                    if forecast_change > 5:
                        analysis += "increase significantly"
                    elif forecast_change > 0:
                        analysis += "increase slightly"
                    elif forecast_change < -5:
                        analysis += "decrease significantly"
                    elif forecast_change < 0:
                        analysis += "decrease slightly"
                    else:
                        analysis += "remain stable"

                    analysis += f" by {abs(forecast_change):.2f}% over the forecast period.</p>"

                analyses[commodity] = analysis

            except Exception as e:
                logger.error(f"Error generating analysis for {commodity}: {e}")
                analyses[commodity] = "<p>Analysis not available due to an error.</p>"

        return analyses

    def _generate_market_insights(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate market insights for the report.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to analyze
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecasts : Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity

        Returns
        -------
        str
            Market insights in HTML format
        """
        try:
            # If LLM provider is available, use it
            if self.llm_provider:
                # Prepare context for LLM
                context = {
                    "commodities": commodities,
                    "market_data": {k: v.to_dict() for k, v in market_data.items()},
                    "forecasts": {k: v.to_dict() for k, v in forecasts.items()}
                }

                # Generate insights using LLM
                insights = self.llm_provider.generate_text(
                    prompt="Generate market insights and analysis based on the provided data.",
                    context=context
                )

                return insights

            # Otherwise, generate simple insights
            insights = "<p>Market insights based on current data and forecasts:</p>"

            # Calculate correlations between commodities
            if len(commodities) > 1:
                prices = {}
                for commodity, data in market_data.items():
                    if not data.empty:
                        prices[commodity] = data['price']

                if prices:
                    price_df = pd.DataFrame(prices)
                    corr = price_df.corr()

                    insights += "<h3>Commodity Correlations</h3>"
                    insights += "<p>The following correlations were observed between commodities:</p><ul>"

                    for i in range(len(commodities)):
                        for j in range(i+1, len(commodities)):
                            c1 = commodities[i]
                            c2 = commodities[j]

                            if c1 in corr.index and c2 in corr.columns:
                                corr_value = corr.loc[c1, c2]

                                insights += f"<li><strong>{c1.replace('_', ' ').title()} and {c2.replace('_', ' ').title()}</strong>: "

                                if corr_value > 0.8:
                                    insights += "Strong positive correlation"
                                elif corr_value > 0.5:
                                    insights += "Moderate positive correlation"
                                elif corr_value > 0.2:
                                    insights += "Weak positive correlation"
                                elif corr_value > -0.2:
                                    insights += "No significant correlation"
                                elif corr_value > -0.5:
                                    insights += "Weak negative correlation"
                                elif corr_value > -0.8:
                                    insights += "Moderate negative correlation"
                                else:
                                    insights += "Strong negative correlation"

                                insights += f" ({corr_value:.2f})</li>"

                    insights += "</ul>"

            # Add trend analysis
            insights += "<h3>Market Trends</h3>"
            insights += "<p>The following trends were observed in the market:</p><ul>"

            for commodity, data in market_data.items():
                if data.empty:
                    continue

                # Calculate simple trend
                data['SMA20'] = data['price'].rolling(window=20).mean()

                if len(data) >= 20:
                    first_sma = data['SMA20'].dropna().iloc[0]
                    last_sma = data['SMA20'].iloc[-1]

                    trend_change = (last_sma / first_sma - 1) * 100

                    insights += f"<li><strong>{commodity.replace('_', ' ').title()}</strong>: "

                    if trend_change > 10:
                        insights += "Strong upward trend"
                    elif trend_change > 5:
                        insights += "Moderate upward trend"
                    elif trend_change > 1:
                        insights += "Slight upward trend"
                    elif trend_change > -1:
                        insights += "Sideways trend"
                    elif trend_change > -5:
                        insights += "Slight downward trend"
                    elif trend_change > -10:
                        insights += "Moderate downward trend"
                    else:
                        insights += "Strong downward trend"

                    insights += f" ({trend_change:.2f}%)</li>"

            insights += "</ul>"

            # Add seasonal patterns if data spans more than 90 days
            for commodity, data in market_data.items():
                if data.empty or len(data) < 90:
                    continue

                # Check if data has date index
                if isinstance(data.index, pd.DatetimeIndex):
                    # Extract month from index
                    data['month'] = data.index.month

                    # Calculate monthly averages
                    monthly_avg = data.groupby('month')['price'].mean()

                    if len(monthly_avg) > 1:
                        max_month = monthly_avg.idxmax()
                        min_month = monthly_avg.idxmin()

                        insights += f"<h3>Seasonal Patterns for {commodity.replace('_', ' ').title()}</h3>"
                        insights += "<p>Based on historical data, the following seasonal patterns were observed:</p>"
                        insights += f"<p>Highest prices tend to occur in {datetime(2000, max_month, 1).strftime('%B')}.</p>"
                        insights += f"<p>Lowest prices tend to occur in {datetime(2000, min_month, 1).strftime('%B')}.</p>"

            return insights

        except Exception as e:
            logger.error(f"Error generating market insights: {e}")
            return "<p>Market insights not available due to an error.</p>"

    def _generate_risk_assessment(
        self,
        commodities: List[str],
        risk_metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Generate risk assessment for the report.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to assess
        risk_metrics : Dict[str, Dict[str, float]]
            Dictionary of risk metrics for each commodity

        Returns
        -------
        str
            Risk assessment in HTML format
        """
        try:
            # If LLM provider is available, use it
            if self.llm_provider:
                # Prepare context for LLM
                context = {
                    "commodities": commodities,
                    "risk_metrics": risk_metrics
                }

                # Generate risk assessment using LLM
                assessment = self.llm_provider.generate_text(
                    prompt="Generate a risk assessment based on the provided risk metrics.",
                    context=context
                )

                return assessment

            # Otherwise, generate a simple risk assessment
            assessment = "<p>Risk assessment based on calculated metrics:</p>"

            # Add overall risk level
            overall_volatility = 0
            commodity_count = 0

            for commodity, metrics in risk_metrics.items():
                if 'volatility' in metrics:
                    overall_volatility += metrics['volatility']
                    commodity_count += 1

            if commodity_count > 0:
                avg_volatility = overall_volatility / commodity_count

                assessment += "<h3>Overall Market Risk</h3>"
                assessment += "<p>The overall market risk level is "

                if avg_volatility > 0.3:
                    assessment += "<span class='negative'>High</span>"
                elif avg_volatility > 0.2:
                    assessment += "<span class='neutral'>Moderate</span>"
                else:
                    assessment += "<span class='positive'>Low</span>"

                assessment += f" with an average annualized volatility of {avg_volatility:.2f}.</p>"

            # Add risk assessment for each commodity
            assessment += "<h3>Commodity Risk Levels</h3>"
            assessment += "<ul>"

            for commodity, metrics in risk_metrics.items():
                if 'volatility' not in metrics:
                    continue

                volatility = metrics['volatility']
                var_95 = metrics.get('var_95', 0)
                max_drawdown = metrics.get('max_drawdown', 0)

                assessment += f"<li><strong>{commodity.replace('_', ' ').title()}</strong>: "

                if volatility > 0.3:
                    assessment += "<span class='negative'>High risk</span>"
                elif volatility > 0.2:
                    assessment += "<span class='neutral'>Moderate risk</span>"
                else:
                    assessment += "<span class='positive'>Low risk</span>"

                assessment += f" (Volatility: {volatility:.2f}, VaR(95%): {var_95:.2f}, Max Drawdown: {max_drawdown:.2f})</li>"

            assessment += "</ul>"

            # Add risk management recommendations
            assessment += "<h3>Risk Management Recommendations</h3>"
            assessment += "<ul>"
            assessment += "<li>Diversify across multiple commodities to reduce portfolio volatility.</li>"
            assessment += "<li>Use stop-loss orders to limit potential losses.</li>"
            assessment += "<li>Consider hedging strategies for high-volatility commodities.</li>"
            assessment += "<li>Monitor market conditions regularly and adjust positions as needed.</li>"
            assessment += "</ul>"

            return assessment

        except Exception as e:
            logger.error(f"Error generating risk assessment: {e}")
            return "<p>Risk assessment not available due to an error.</p>"

    def _create_visualization_charts(
        self,
        commodities: List[str],
        market_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame]
    ) -> Dict[str, str]:
        """
        Create visualization charts for the report.

        Parameters
        ----------
        commodities : List[str]
            List of commodities to visualize
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecasts : Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity

        Returns
        -------
        Dict[str, str]
            Dictionary of base64-encoded chart images
        """
        charts = {}

        for commodity in commodities:
            try:
                data = market_data.get(commodity, pd.DataFrame())
                forecast = forecasts.get(commodity, pd.DataFrame())

                if data.empty:
                    continue

                # Create price chart
                fig, ax = plt.subplots(figsize=(10, 6))

                # Plot price
                ax.plot(data.index, data['price'], label='Price', color='blue')

                # Plot moving averages if available
                if 'SMA20' in data.columns:
                    ax.plot(data.index, data['SMA20'], label='20-day MA', color='orange')

                if 'SMA50' in data.columns:
                    ax.plot(data.index, data['SMA50'], label='50-day MA', color='green')

                ax.set_title(f"{commodity.replace('_', ' ').title()} Price Trend")
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                # Save chart as base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                charts[f"{commodity}_price"] = image_base64
                plt.close(fig)

                # Create forecast chart if available
                if not forecast.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))

                    # Plot historical price
                    ax.plot(data.index, data['price'], label='Historical', color='blue')

                    # Plot forecast
                    ax.plot(forecast.index, forecast['forecast'], label='Forecast', color='red')

                    # Plot confidence intervals if available
                    if 'lower_bound' in forecast.columns and 'upper_bound' in forecast.columns:
                        ax.fill_between(
                            forecast.index,
                            forecast['lower_bound'],
                            forecast['upper_bound'],
                            color='red',
                            alpha=0.2,
                            label='Confidence Interval'
                        )

                    ax.set_title(f"{commodity.replace('_', ' ').title()} Price Forecast")
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Price')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()

                    # Save chart as base64
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png')
                    buffer.seek(0)
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    charts[f"{commodity}_forecast"] = image_base64
                    plt.close(fig)

            except Exception as e:
                logger.error(f"Error creating visualization charts for {commodity}: {e}")

        return charts

    def _create_risk_heatmap(
        self,
        risk_metrics: Dict[str, Dict[str, float]]
    ) -> str:
        """
        Create risk heatmap visualization.

        Parameters
        ----------
        risk_metrics : Dict[str, Dict[str, float]]
            Dictionary of risk metrics for each commodity

        Returns
        -------
        str
            Base64-encoded heatmap image
        """
        try:
            # Extract risk metrics
            commodities = []
            volatilities = []
            var_values = []

            for commodity, metrics in risk_metrics.items():
                if 'volatility' in metrics and 'var_95' in metrics:
                    commodities.append(commodity.replace('_', ' ').title())
                    volatilities.append(metrics['volatility'])
                    var_values.append(abs(metrics['var_95']))

            if not commodities:
                return ""

            # Create heatmap data
            risk_data = np.column_stack([volatilities, var_values])

            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))

            # Create heatmap
            im = ax.imshow(risk_data, cmap='YlOrRd')

            # Set labels
            ax.set_yticks(np.arange(len(commodities)))
            ax.set_yticklabels(commodities)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Volatility', 'VaR (95%)'])

            # Add colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Risk Level', rotation=-90, va="bottom")

            # Add text annotations
            for i in range(len(commodities)):
                for j in range(2):
                    text = ax.text(j, i, f"{risk_data[i, j]:.2f}",
                                  ha="center", va="center", color="black")

            ax.set_title("Risk Heatmap")
            plt.tight_layout()

            # Save heatmap as base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close(fig)

            return image_base64

        except Exception as e:
            logger.error(f"Error creating risk heatmap: {e}")
            return ""

    def _format_metrics(
        self,
        commodity: str,
        market_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame]
    ) -> List[Dict[str, str]]:
        """
        Format metrics for display in the report.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        market_data : Dict[str, pd.DataFrame]
            Dictionary of market data for each commodity
        forecasts : Dict[str, pd.DataFrame]
            Dictionary of forecasts for each commodity

        Returns
        -------
        List[Dict[str, str]]
            List of formatted metrics
        """
        metrics = []

        try:
            data = market_data.get(commodity, pd.DataFrame())
            forecast = forecasts.get(commodity, pd.DataFrame())

            if data.empty:
                return metrics

            # Current price
            current_price = data['price'].iloc[-1]
            metrics.append({
                "name": "Current Price",
                "value": f"${current_price:.2f}",
                "change": "",
                "change_class": ""
            })

            # Daily change
            if len(data) >= 2:
                prev_price = data['price'].iloc[-2]
                daily_change = (current_price / prev_price - 1) * 100

                metrics.append({
                    "name": "Daily Change",
                    "value": f"${current_price - prev_price:.2f}",
                    "change": f"{daily_change:+.2f}%",
                    "change_class": "positive" if daily_change >= 0 else "negative"
                })

            # Weekly change
            if len(data) >= 5:
                week_ago_price = data['price'].iloc[-5]
                weekly_change = (current_price / week_ago_price - 1) * 100

                metrics.append({
                    "name": "Weekly Change",
                    "value": f"${current_price - week_ago_price:.2f}",
                    "change": f"{weekly_change:+.2f}%",
                    "change_class": "positive" if weekly_change >= 0 else "negative"
                })

            # Monthly change
            if len(data) >= 20:
                month_ago_price = data['price'].iloc[-20]
                monthly_change = (current_price / month_ago_price - 1) * 100

                metrics.append({
                    "name": "Monthly Change",
                    "value": f"${current_price - month_ago_price:.2f}",
                    "change": f"{monthly_change:+.2f}%",
                    "change_class": "positive" if monthly_change >= 0 else "negative"
                })

            # Volatility
            if 'return' in data.columns:
                volatility = data['return'].std() * np.sqrt(252)

                metrics.append({
                    "name": "Volatility (Annualized)",
                    "value": f"{volatility:.2f}",
                    "change": "",
                    "change_class": ""
                })

            # Forecast
            if not forecast.empty:
                last_forecast = forecast['forecast'].iloc[-1]
                forecast_change = (last_forecast / current_price - 1) * 100

                metrics.append({
                    "name": "Forecast (End of Period)",
                    "value": f"${last_forecast:.2f}",
                    "change": f"{forecast_change:+.2f}%",
                    "change_class": "positive" if forecast_change >= 0 else "negative"
                })

            return metrics

        except Exception as e:
            logger.error(f"Error formatting metrics for {commodity}: {e}")
            return metrics

    def _generate_sample_data(
        self,
        commodity: str,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Generate sample market data for demonstration.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        start_date : datetime
            Start date for the data
        end_date : datetime
            End date for the data

        Returns
        -------
        pd.DataFrame
            Sample market data
        """
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Set initial price based on commodity
        if commodity == 'crude_oil':
            initial_price = 75.0
        elif commodity == 'natural_gas':
            initial_price = 3.5
        elif commodity == 'gasoline':
            initial_price = 2.5
        elif commodity == 'diesel':
            initial_price = 3.0
        else:
            initial_price = 50.0

        # Generate random walk for price
        np.random.seed(hash(commodity) % 10000)
        returns = np.random.normal(0.0002, 0.02, size=len(date_range))
        prices = initial_price * (1 + returns).cumprod()

        # Create DataFrame
        df = pd.DataFrame({
            'price': prices
        }, index=date_range)

        return df

    def _generate_sample_forecast(
        self,
        commodity: str,
        data: pd.DataFrame,
        forecast_days: int
    ) -> pd.DataFrame:
        """
        Generate sample forecast for demonstration.

        Parameters
        ----------
        commodity : str
            Name of the commodity
        data : pd.DataFrame
            Historical market data
        forecast_days : int
            Number of days to forecast

        Returns
        -------
        pd.DataFrame
            Sample forecast
        """
        if data.empty:
            return pd.DataFrame()

        # Get last date and price
        last_date = data.index[-1]
        last_price = data['price'].iloc[-1]

        # Generate date range for forecast
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=forecast_days,
            freq='D'
        )

        # Set trend based on commodity
        if commodity == 'crude_oil':
            trend = 0.001
        elif commodity == 'natural_gas':
            trend = -0.0005
        elif commodity == 'gasoline':
            trend = 0.0008
        elif commodity == 'diesel':
            trend = 0.0012
        else:
            trend = 0.0

        # Generate random walk for forecast
        np.random.seed(hash(commodity + 'forecast') % 10000)
        forecast_returns = np.random.normal(trend, 0.015, size=forecast_days)
        forecast_prices = last_price * np.cumprod(1 + forecast_returns)

        # Calculate confidence intervals
        std_dev = data['price'].pct_change().std() * np.sqrt(np.arange(1, forecast_days + 1))
        lower_bound = forecast_prices * (1 - 1.96 * std_dev)
        upper_bound = forecast_prices * (1 + 1.96 * std_dev)

        # Create DataFrame
        df = pd.DataFrame({
            'forecast': forecast_prices,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }, index=forecast_dates)

        return df

    def _convert_html_to_pdf(self, html_content: str, output_path: str) -> None:
        """
        Convert HTML content to PDF.

        Parameters
        ----------
        html_content : str
            HTML content to convert
        output_path : str
            Path to save the PDF
        """
        try:
            # Try to import weasyprint
            from weasyprint import HTML

            # Convert HTML to PDF
            HTML(string=html_content).write_pdf(output_path)

            logger.info(f"Converted HTML to PDF: {output_path}")

        except ImportError:
            logger.error("WeasyPrint not installed. Cannot convert HTML to PDF.")

            # Save HTML instead
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Saved HTML instead: {html_path}")

        except Exception as e:
            logger.error(f"Error converting HTML to PDF: {e}")

            # Save HTML instead
            html_path = output_path.replace('.pdf', '.html')
            with open(html_path, 'w') as f:
                f.write(html_content)

            logger.info(f"Saved HTML instead: {html_path}")
