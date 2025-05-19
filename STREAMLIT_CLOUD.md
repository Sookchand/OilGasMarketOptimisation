# Deploying to Streamlit Cloud

This document provides instructions for deploying the Oil & Gas Market Optimization system to Streamlit Cloud.

## Prerequisites

1. A Streamlit Cloud account (sign up at https://streamlit.io/cloud)
2. A GitHub repository containing your project

## Deployment Steps

### 1. Prepare Your Repository

1. Make sure your repository contains the following files:
   - `minimal_app.py` (the extremely simplified version of the app)
   - `requirements.txt` (with absolute minimum dependencies)
   - `.streamlit/config.toml` (Streamlit configuration)
   - `packages.txt` (minimal system dependencies)

2. Push these files to your GitHub repository:
   ```bash
   git add minimal_app.py requirements.txt .streamlit/config.toml packages.txt
   git commit -m "Add minimal files for Streamlit Cloud deployment"
   git push
   ```

### 2. Deploy to Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Connect your GitHub repository
4. Set the main file path to `minimal_app.py`
5. Click "Deploy"

### 3. Troubleshooting Deployment Issues

If you encounter issues during deployment:

#### Common Error: "Installer returned a non-zero exit code"

This usually means there's a problem with your dependencies. Try these solutions:

1. **Simplify requirements.txt**:
   - Remove version constraints (e.g., use `pandas` instead of `pandas>=1.3.0`)
   - Remove unnecessary packages
   - Split complex dependencies into separate lines

2. **Check for system dependencies**:
   - Some Python packages require system libraries
   - Add these to `packages.txt` (one per line)

3. **Use a different Python version**:
   - Try specifying a different Python version in your app settings

4. **Check for memory issues**:
   - Streamlit Cloud has memory limits
   - Reduce the size of your data or models

#### Other Common Issues

1. **Import errors**:
   - Make sure all imports are available in requirements.txt
   - Use try/except blocks for optional dependencies

2. **File not found errors**:
   - Create directories programmatically with `os.makedirs()`
   - Use relative paths

3. **Timeout errors**:
   - Streamlit Cloud has a deployment timeout
   - Simplify your app's initialization

### 4. Updating Your App

To update your app after making changes:

1. Push changes to your GitHub repository
2. Streamlit Cloud will automatically redeploy your app

## Integrating with Your Portfolio Website

Once your app is deployed to Streamlit Cloud, you can integrate it with your portfolio website:

### Option 1: Embed as an iframe

Add this HTML to your portfolio website:

```html
<iframe
  src="https://yourusername-oil-gas-market-optimization-minimal-app-abc123.streamlit.app"
  width="100%"
  height="800px"
  frameborder="0"
  title="Oil & Gas Market Optimization"
></iframe>
```

### Option 2: Add a Link

Add a button or link to your portfolio website:

```html
<a href="https://yourusername-oil-gas-market-optimization-minimal-app-abc123.streamlit.app"
   class="button"
   target="_blank">
  Launch Oil & Gas Market Optimization App
</a>
```

## Resources

- [Streamlit Cloud Documentation](https://docs.streamlit.io/streamlit-cloud)
- [Streamlit Deployment Guide](https://docs.streamlit.io/streamlit-cloud/get-started/deploy-an-app)
