"""
Q&A component for the Oil & Gas Market Optimization system.
This module provides a simple question and answer functionality.
"""

import streamlit as st
import re

# Dictionary of questions and answers
QA_DATABASE = {
    # Data-Related Questions
    "what data format is required": 
        "The system accepts CSV and Excel files. At minimum, you need a 'Date' column and a 'Price' column. The Date column will be used as the index.",
    
    "can i use data from different sources": 
        "Yes, as long as the data is formatted correctly. The system will clean and standardize the data during processing.",
    
    "how much historical data is recommended": 
        "For robust backtesting, at least 2-3 years of data is recommended. For risk analysis, 5+ years provides better statistical significance.",
    
    "how does the sample data generation work": 
        "The sample data generator creates synthetic price data using a combination of random walks, seasonality, and trends. It's designed to mimic real-world commodity price movements.",
    
    # Trading Strategy Questions
    "which strategy performs best": 
        "There is no universally 'best' strategy. Performance depends on the commodity, time period, and market conditions. The system allows you to compare multiple strategies to find what works best for your specific case.",
    
    "how do i interpret the backtest results": 
        "The backtest results show how the strategy would have performed historically. Key metrics include total return, Sharpe ratio, maximum drawdown, and win rate. Higher returns with lower drawdowns and higher Sharpe ratios indicate better strategies.",
    
    "can i create custom strategies": 
        "Yes, you can implement custom strategies by creating new files in the `src/trading/strategies/` directory. See the Advanced Usage section in the INSTRUCTIONS.md file.",
    
    "how often should i reoptimize strategy parameters": 
        "It's good practice to review and potentially reoptimize parameters quarterly or when market conditions change significantly.",
    
    # Risk Analysis Questions
    "what is value at risk": 
        "Value at Risk (VaR) estimates the maximum potential loss over a specified time horizon at a given confidence level. For example, a 1-day 95% VaR of 2% means there's a 95% probability that you won't lose more than 2% in one day.",
    
    "how are monte carlo simulations used": 
        "Monte Carlo simulations generate thousands of possible future price paths based on historical data characteristics. This helps estimate the range of potential outcomes and assess the probability of different scenarios.",
    
    "how should i use the risk metrics": 
        "Risk metrics should inform position sizing, stop-loss levels, and portfolio allocation. For example, you might limit position sizes so that the potential loss (based on VaR) doesn't exceed a certain percentage of your portfolio.",
    
    # Technical Questions
    "why am i getting a no module found error": 
        "This usually indicates a missing Python module. Make sure you've installed all dependencies with `pip install -r requirements.txt` and that you're running the commands from the project root directory.",
    
    "how can i speed up the data processing": 
        "Process one commodity at a time, use a smaller date range, or consider downsampling high-frequency data. You can also use the command-line interface for more efficient batch processing.",
    
    "can i run this system on a cloud server": 
        "Yes, the system can be deployed to cloud platforms like AWS, Google Cloud, or Heroku. See the deployment instructions in DEPLOYMENT.md.",
    
    # General Questions
    "what is this system": 
        "This is the Oil & Gas Market Optimization system, a comprehensive platform for analyzing commodity price data, backtesting trading strategies, and performing risk analysis for crude oil, regular gasoline, conventional gasoline, and diesel.",
    
    "how do i start": 
        "Start by generating sample data with `python create_basic_data.py`, then run the web application with `streamlit run web_app.py`. You can then explore the different features of the system through the web interface.",
    
    "where can i find documentation": 
        "Comprehensive documentation is available in the INSTRUCTIONS.md and USAGE_GUIDE.md files. For specific topics, check the docs/ directory.",
    
    "how do i update the system": 
        "Pull the latest changes from the GitHub repository and reinstall dependencies if needed: `git pull` followed by `pip install -r requirements.txt`."
}

def find_best_match(question, qa_database):
    """
    Find the best matching question in the database.
    
    Parameters:
    -----------
    question : str
        The user's question
    qa_database : dict
        Dictionary of questions and answers
        
    Returns:
    --------
    tuple
        (best_match_question, best_match_answer, match_score)
    """
    # Preprocess the question
    question = question.lower().strip()
    if question.endswith('?'):
        question = question[:-1]
    
    best_match = None
    best_score = 0
    
    for q in qa_database.keys():
        # Simple word overlap score
        q_words = set(q.split())
        question_words = set(question.split())
        common_words = q_words.intersection(question_words)
        
        if len(common_words) > 0:
            # Calculate score based on word overlap
            score = len(common_words) / max(len(q_words), len(question_words))
            
            # Boost score for key phrase matches
            for phrase in q_words:
                if phrase in question and len(phrase) > 3:  # Only consider phrases longer than 3 chars
                    score += 0.1
            
            if score > best_score:
                best_score = score
                best_match = q
    
    if best_match and best_score > 0.3:  # Threshold for a good match
        return best_match, qa_database[best_match], best_score
    else:
        return None, None, 0

def qa_interface():
    """
    Create a Streamlit interface for the Q&A component.
    """
    st.header("Questions & Answers")
    
    st.write("""
    Ask a question about the Oil & Gas Market Optimization system. 
    You can ask about data formats, trading strategies, risk analysis, or technical issues.
    """)
    
    # User input
    user_question = st.text_input("Your question:", "")
    
    if st.button("Ask") or user_question:
        if user_question:
            # Find the best match
            _, answer, score = find_best_match(user_question, QA_DATABASE)
            
            if answer:
                st.success("Answer:")
                st.write(answer)
                
                # Show related questions
                st.subheader("You might also be interested in:")
                related_questions = []
                for q in QA_DATABASE.keys():
                    if q != user_question.lower().strip():
                        _, _, rel_score = find_best_match(q, {user_question.lower().strip(): ""})
                        if rel_score > 0.2:
                            related_questions.append(q)
                
                # Display up to 3 related questions
                for i, q in enumerate(related_questions[:3]):
                    if st.button(f"{q.capitalize()}?", key=f"related_{i}"):
                        st.session_state.user_question = q
                        st.experimental_rerun()
            else:
                st.error("I don't have a specific answer for that question. Please try rephrasing or check the documentation.")
                st.info("You can find comprehensive documentation in the INSTRUCTIONS.md and USAGE_GUIDE.md files.")
    
    # Show some example questions
    with st.expander("Example questions you can ask"):
        example_questions = [
            "What data format is required?",
            "Which strategy performs best?",
            "What is Value at Risk?",
            "How can I speed up data processing?",
            "Can I create custom strategies?"
        ]
        
        for i, q in enumerate(example_questions):
            if st.button(q, key=f"example_{i}"):
                st.session_state.user_question = q
                st.experimental_rerun()

if __name__ == "__main__":
    st.set_page_config(page_title="Q&A - Oil & Gas Market Optimization", page_icon="📈")
    qa_interface()
