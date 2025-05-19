# Deploying to Netlify and Streamlit Cloud

This document provides instructions for deploying the Oil & Gas Market Optimization system to both Netlify and Streamlit Cloud.

## Prerequisites

1. A Netlify account
2. A Streamlit Cloud account
3. Git installed on your local machine
4. A GitHub repository for your project

## Deployment Steps

### 1. Push Your Code to GitHub

First, push your code to a GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/oil-gas-market-optimization.git
git push -u origin main
```

### 2. Deploy to Streamlit Cloud

1. Log in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Connect your GitHub repository
4. Set the main file path to `web_app.py`
5. Click "Deploy"
6. Once deployed, note the URL (e.g., `https://yourusername-oil-gas-market-optimization-web-app-abc123.streamlit.app`)

### 3. Update the Streamlit App URL

1. Open `index.html` and update the "Launch Application" button URL with your Streamlit Cloud URL:
   ```html
   <a href="https://yourusername-oil-gas-market-optimization-web-app-abc123.streamlit.app" class="btn" target="_blank">
       Launch Application
   </a>
   ```

2. Open `netlify_build.sh` and update the same URL in the HTML template

3. Commit and push these changes:
   ```bash
   git add index.html netlify_build.sh
   git commit -m "Update Streamlit app URL"
   git push
   ```

### 4. Deploy to Netlify

#### Option 1: Deploy via Netlify UI (Recommended)

1. Log in to your Netlify account
2. Click "New site from Git"
3. Choose GitHub as your Git provider
4. Authorize Netlify to access your GitHub account
5. Select your repository
6. Configure build settings:
   - Build command: `bash netlify_build.sh`
   - Publish directory: `.`
7. Click "Show advanced" and add these environment variables:
   - Key: `PYTHON_VERSION`, Value: `3.8`
8. Click "Deploy site"

#### Option 2: Deploy via Netlify CLI

1. Install the Netlify CLI:
   ```bash
   npm install netlify-cli -g
   ```

2. Log in to Netlify:
   ```bash
   netlify login
   ```

3. Initialize your site:
   ```bash
   netlify init
   ```

4. Deploy your site:
   ```bash
   netlify deploy --prod
   ```

### 5. Troubleshooting Netlify Deployment

If you encounter issues with the Netlify deployment:

1. Check the build logs in the Netlify dashboard
2. Common issues:
   - Python version compatibility: Try using Python 3.8 instead of 3.9
   - Missing dependencies: Make sure all dependencies are in requirements.txt
   - Build timeout: Simplify the build process by removing data generation steps

3. As a fallback, you can deploy just the static HTML page:
   - Build command: `echo "Static deployment"`
   - Publish directory: `.`

## Integrating with Your Portfolio Website

To integrate this application with your existing portfolio website (https://sookchandportfolio.netlify.app/):

### Option 1: Embed as an iframe

Add the following HTML to your portfolio website:

```html
<iframe
  src="https://yourusername-oil-gas-market-optimization-web-app-abc123.streamlit.app"
  width="100%"
  height="800px"
  frameborder="0"
  title="Oil & Gas Market Optimization"
></iframe>
```

### Option 2: Add as a Subdomain

1. Configure a subdomain in your Netlify DNS settings (e.g., oil-gas.sookchandportfolio.netlify.app)
2. Point the subdomain to your Netlify site
3. Add a link to the subdomain from your portfolio website

### Option 3: Add as a Page

1. Create a new page on your portfolio website
2. Add the following HTML:

```html
<div class="project-section">
  <h2>Oil & Gas Market Optimization</h2>
  <p>
    An interactive system for analyzing oil and gas market data, backtesting trading strategies,
    and performing risk analysis. Upload your own datasets or use sample data.
  </p>

  <a href="https://yourusername-oil-gas-market-optimization-web-app-abc123.streamlit.app" class="btn btn-primary" target="_blank">
    Launch Application
  </a>
</div>
```

## Troubleshooting Integration

If you encounter issues with the integration:

1. Make sure the Streamlit app is properly deployed and accessible
2. Check that the iframe or link is correctly configured
3. Test the integration in different browsers
4. Consider using a direct link instead of an iframe if you encounter cross-origin issues

## Troubleshooting

If you encounter issues during deployment:

1. Check the Netlify build logs for errors
2. Ensure all dependencies are correctly specified in requirements.txt
3. Verify that your runtime.txt and Procfile are correctly configured
4. Check that your application is compatible with the Python version specified in runtime.txt

For more help, refer to the [Netlify documentation](https://docs.netlify.com/) or contact Netlify support.
