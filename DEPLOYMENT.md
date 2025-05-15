# Deploying to Netlify

This document provides instructions for deploying the Oil & Gas Market Optimization system to Netlify.

## Prerequisites

1. A Netlify account
2. Git installed on your local machine
3. A GitHub repository for your project

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

### 2. Deploy to Netlify

#### Option 1: Deploy via Netlify UI

1. Log in to your Netlify account
2. Click "New site from Git"
3. Choose GitHub as your Git provider
4. Authorize Netlify to access your GitHub account
5. Select your repository
6. Configure build settings:
   - Build command: `pip install -r requirements.txt`
   - Publish directory: `.`
7. Click "Deploy site"

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

### 3. Configure Environment Variables

1. Go to your site settings in Netlify
2. Navigate to "Build & deploy" > "Environment"
3. Add the following environment variables:
   - `PYTHON_VERSION`: `3.9`
   - Any API keys or secrets your application needs

### 4. Configure Build Hooks (Optional)

If you want to trigger builds automatically when your data changes:

1. Go to your site settings in Netlify
2. Navigate to "Build & deploy" > "Build hooks"
3. Create a new build hook
4. Use the provided URL to trigger builds via HTTP POST requests

## Integrating with Your Portfolio Website

To integrate this application with your existing portfolio website (https://sookchandportfolio.netlify.app/):

### Option 1: Embed as an iframe

Add the following HTML to your portfolio website:

```html
<iframe 
  src="https://your-netlify-app-url.netlify.app" 
  width="100%" 
  height="800px" 
  frameborder="0"
  title="Oil & Gas Market Optimization"
></iframe>
```

### Option 2: Add as a Subdomain

1. Configure a subdomain in your Netlify DNS settings (e.g., oil-gas.sookchandportfolio.netlify.app)
2. Point the subdomain to your new Netlify site
3. Add a link to the subdomain from your portfolio website

### Option 3: Add as a Page

1. Create a new page on your portfolio website
2. Add a link to your Netlify app
3. Style the link to match your portfolio design

## Troubleshooting

If you encounter issues during deployment:

1. Check the Netlify build logs for errors
2. Ensure all dependencies are correctly specified in requirements.txt
3. Verify that your runtime.txt and Procfile are correctly configured
4. Check that your application is compatible with the Python version specified in runtime.txt

For more help, refer to the [Netlify documentation](https://docs.netlify.com/) or contact Netlify support.
