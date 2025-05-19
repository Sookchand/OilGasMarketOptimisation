#!/bin/bash

# Exit on error
set -e

# Print commands
set -x

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/raw data/processed data/features data/insights data/chroma logs results/forecasting results/backtests results/model_selection results/monte_carlo results/trading

# Generate sample data
python create_basic_data.py

# Process data
python run_data_pipeline.py

# Create a simple HTML file that links to the Streamlit app
cat > index.html << 'EOL'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Oil & Gas Market Optimization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: #333;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 1rem;
            text-align: center;
        }
        h1 {
            margin: 0;
        }
        .content {
            background-color: white;
            padding: 2rem;
            margin-top: 2rem;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .btn {
            display: inline-block;
            background-color: #3498db;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-top: 20px;
            font-weight: bold;
            transition: background-color 0.3s;
        }
        .btn:hover {
            background-color: #2980b9;
        }
        .features {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 2rem;
        }
        .feature {
            flex-basis: 30%;
            background-color: #f9f9f9;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .feature h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        @media (max-width: 768px) {
            .feature {
                flex-basis: 100%;
            }
        }
    </style>
</head>
<body>
    <header>
        <h1>📈 Oil & Gas Market Optimization</h1>
    </header>
    
    <div class="container">
        <div class="content">
            <h2>Interactive Trading and Risk Analysis Platform</h2>
            <p>
                This application provides tools for analyzing oil and gas market data, backtesting trading strategies,
                and optimizing portfolios. You can upload your own datasets or use sample data.
            </p>
            
            <p>
                Due to Netlify's limitations with running Python applications, this application is hosted on Streamlit Cloud.
                Click the button below to access the full interactive application.
            </p>
            
            <a href="https://share.streamlit.io/yourusername/oil-gas-market-optimization/main/web_app.py" class="btn" target="_blank">
                Launch Application
            </a>
            
            <div class="features">
                <div class="feature">
                    <h3>Data Management</h3>
                    <p>Upload your own datasets for each commodity or use sample data. Process and visualize the data before analysis.</p>
                </div>
                
                <div class="feature">
                    <h3>Trading Strategies</h3>
                    <p>Backtest trading strategies including Moving Average Crossover and RSI. Analyze performance metrics and visualize results.</p>
                </div>
                
                <div class="feature">
                    <h3>Risk Analysis</h3>
                    <p>Calculate Value at Risk (VaR) using multiple methods. Run Monte Carlo simulations to project future portfolio values.</p>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
EOL

echo "Build completed successfully!"
