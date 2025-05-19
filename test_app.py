"""
Simple test application to verify Streamlit is working correctly.
"""

import streamlit as st

# Set page config with minimal settings
st.set_page_config(
    page_title="Streamlit Test",
    page_icon="🧪",
    layout="centered"
)

# Simple header and description
st.title("Streamlit Test Application")
st.write("This is a simple test to verify that Streamlit is working correctly.")

# Add a divider
st.divider()

# Simple interactive elements
st.subheader("Basic Interactivity Test")

# Test button
if st.button("Click Me"):
    st.success("Button clicked successfully!")

# Test selectbox
option = st.selectbox(
    "Select an option:",
    ["Option 1", "Option 2", "Option 3"]
)
st.write(f"You selected: {option}")

# Test slider
value = st.slider("Select a value:", 0, 100, 50)
st.write(f"Selected value: {value}")

# Add another divider
st.divider()

# Display system info
st.subheader("System Information")
import sys
import streamlit as st
import pandas as pd
import numpy as np

st.write(f"Python version: {sys.version}")
st.write(f"Streamlit version: {st.__version__}")
st.write(f"Pandas version: {pd.__version__}")
st.write(f"NumPy version: {np.__version__}")

# Add a final note
st.info("If you can see this page and interact with the elements above, Streamlit is working correctly on your system.")
