# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the Oil & Gas Market Optimization project.

## Main Workflow (`main.yml`)

The main workflow is triggered on:
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main`, `master`, or `develop` branches
- Manual trigger via GitHub Actions UI

### Jobs

#### 1. Test

This job runs tests on the codebase using multiple Python versions:
- Sets up Python environment (3.8 and 3.9)
- Creates necessary directories
- Installs dependencies
- Runs linting with flake8
- Runs unit tests with pytest and generates coverage report

#### 2. Build

This job builds the Python package:
- Sets up Python 3.9
- Installs build dependencies
- Builds the package
- Archives the build artifacts

#### 3. Data Pipeline

This job runs the data processing pipeline (only on manual trigger):
- Sets up Python 3.9
- Creates necessary directories
- Downloads sample data
- Runs the data pipeline with ARIMA model
- Archives the pipeline results

#### 4. Docker

This job builds and pushes a Docker image (only on main/master branch):
- Sets up Docker Buildx
- Logs in to DockerHub
- Extracts metadata for Docker
- Builds and pushes the Docker image

#### 5. Deploy Docs

This job deploys documentation to GitHub Pages (only on main/master branch):
- Sets up Python 3.9
- Installs MkDocs
- Sets up documentation
- Deploys to GitHub Pages

## Required Secrets

The following secrets need to be set in your GitHub repository:

- `EIA_API_KEY`: API key for the U.S. Energy Information Administration (EIA)
- `DOCKERHUB_USERNAME`: Your DockerHub username
- `DOCKERHUB_TOKEN`: Your DockerHub access token

## Usage

### Manual Trigger

You can manually trigger the workflow from the GitHub Actions UI:
1. Go to the "Actions" tab in your repository
2. Select the "Oil & Gas Market Optimization CI/CD" workflow
3. Click "Run workflow"
4. Select the branch and click "Run workflow"

### Automatic Trigger

The workflow will automatically run when:
- You push to the main, master, or develop branch
- You create a pull request to the main, master, or develop branch

## Customization

You can customize the workflow by editing the `.github/workflows/main.yml` file:
- Add or remove jobs
- Change the Python versions
- Modify the test commands
- Add additional deployment steps
