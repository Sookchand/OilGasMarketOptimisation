# Oil & Gas Market Optimization Website

This is the dedicated website for the Oil & Gas Market Optimization project. It showcases the project's features and provides an interactive demo that connects to the Streamlit web application.

## Features

- Modern, responsive design
- Interactive demo with Streamlit integration
- Detailed feature descriptions
- Sample data downloads
- Comprehensive documentation

## Directory Structure

```
website/
├── css/
│   ├── styles.css        # Main stylesheet
│   └── responsive.css    # Responsive design styles
├── js/
│   └── script.js         # JavaScript for interactivity
├── images/               # Images and screenshots
├── data/                 # Sample data files
├── index.html            # Home page
├── demo.html             # Interactive demo page
├── features.html         # Features page
├── documentation.html    # Documentation page
└── about.html            # About page
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Sookchand/OilGasMarketOptimisation.git
   cd OilGasMarketOptimisation/website
   ```

2. Open the website locally:
   - Open `index.html` in your web browser
   - Or use a local development server:
     ```bash
     # Using Python
     python -m http.server 8000
     
     # Using Node.js
     npx serve
     ```

3. Deploy to a web hosting service:
   - Upload the entire `website` directory to your web hosting service
   - Or deploy to Netlify, GitHub Pages, or Vercel

## Streamlit Integration

The website integrates with the Streamlit web application for the interactive demo. The Streamlit app is embedded in an iframe on the demo page.

To update the Streamlit app URL:
1. Deploy your Streamlit app to a hosting service (e.g., Streamlit Cloud, Heroku)
2. Update the iframe src attribute in `demo.html`:
   ```html
   <iframe id="streamlit-iframe" src="https://your-streamlit-app-url.com" frameborder="0"></iframe>
   ```

## Adding Sample Data

To add sample data files for download:
1. Place the CSV files in the `data` directory
2. Update the download links in `demo.html`:
   ```html
   <a href="data/your_file.csv" class="btn btn-secondary btn-sm" download>Download CSV</a>
   ```

## Adding Screenshots

To add screenshots of the application:
1. Place the image files in the `images` directory
2. Update the image sources in the HTML files:
   ```html
   <img src="images/your-screenshot.png" alt="Description">
   ```

## Customization

- **Colors**: Edit the CSS variables in `css/styles.css` to change the color scheme
- **Fonts**: Update the Google Fonts link in the HTML files to change the typography
- **Content**: Modify the HTML files to update the content

## Credits

Developed by Sookchand Harripersad
