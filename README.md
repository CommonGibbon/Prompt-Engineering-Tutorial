# Prompt Engineering with MLflow

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A practical tutorial and framework for optimizing AI prompts using systematic experimentation and tracking with MLflow. This project demonstrates how to improve prompt performance through iterative testing using a computer vision task: identifying a specific cat named Melvin from photos.

## Overview

This repository showcases prompt engineering best practices through a hands-on example. Instead of relying on intuition, we use MLflow to track experiments systematically, comparing different prompt strategies to find what works best.

**The Challenge**: Train an AI to identify Melvin (a specific tabby cat) from various cat photos, accepting images with Melvin and rejecting images without him.

**The Solution**: Four different prompt engineering approaches, from basic descriptions to multi-modal comparison using sample images.

## Project Structure

```
├── conf/                           # Hydra configuration files
│   ├── config.yaml                # Main configuration
│   └── prompts/                   # Prompt versions (v1-v4)
├── data/
│   ├── images/                    # Test images
│   ├── sample_images/             # Reference images of Melvin
│   └── labels.csv                 # Ground truth labels
├── notebooks/
│   └── prompt_engineering_tutorial.ipynb  # Interactive tutorial
├── scripts/
│   └── prompt_engineering_demo.py # Production script
├── src/prompt_engineering/
│   └── utils.py                   # Core functionality
└── mlruns/                        # MLflow experiment tracking
```

## Setup

### Prerequisites
- Python 3.13+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- OpenRouter API key for accessing AI models

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd PromptEngineering
   ```

2. **Install dependencies with Poetry**
   ```bash
   poetry install
   ```

3. **Set up environment variables**
   Create a `.env` file in the project root:
   ```bash
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

4. **Activate the Poetry environment**
   ```bash
   poetry shell
   ```

## Getting Started

### Interactive Tutorial (Notebook)

The Jupyter notebook provides a step-by-step walkthrough of prompt engineering concepts:

```bash
poetry run jupyter lab notebooks/prompt_engineering_tutorial.ipynb
```

**What you'll learn from the notebook:**
- How to set up MLflow for experiment tracking
- Four different prompt engineering strategies:
  - **V1**: Basic description (baseline)
  - **V2**: Detailed feature-based description
  - **V3**: Chain-of-thought reasoning with self-reflection
  - **V4**: Multi-modal approach using sample images
- How to evaluate and visualize results
- Best practices for systematic prompt optimization

### Production Script

For automated experiments and configuration management:

```bash
poetry run python scripts/prompt_engineering_demo.py
```

## Configuration Management

The project uses [Hydra](https://hydra.cc/) for clean configuration management. All settings are controlled through YAML files in the `conf/` directory.

### Main Configuration (`conf/config.yaml`)

```yaml
defaults:
  - prompts: v4  # Which prompt version to use

image_path: "data/images/"
label_path: "data/labels.csv"
sample_image_path: "data/sample_images/sample.jpg"
model: openai/gpt-5-mini

mlflow:
    tracking_uri: "mlruns"
    experiment_name: "cat_it_prompt_evaluation"
    run_name: "experiment_${now:%Y-%m-%d_%H-%M-%S}"
```

### Editing Configurations

1. **Change prompt version**: Edit the `defaults` section in `config.yaml`
2. **Modify prompts**: Edit files in `conf/prompts/` (v1.yaml through v4.yaml)
3. **Switch models**: Change the `model` parameter
4. **Adjust paths**: Update data paths as needed

### Creating New Prompt Versions

1. Create a new file: `conf/prompts/v5.yaml`
2. Add your prompt configuration:
   ```yaml
   system_prompt: |
     Your new prompt text here...
   
   version: v5
   version_description: Description of this approach
   ```
3. Update `config.yaml` to use the new version: `- prompts: v5`

## Experiment Tracking & Results

### Running Experiments

Each run automatically creates a new MLflow experiment with:
- Prompt version and description
- Model parameters
- Accuracy metrics
- Timestamps

### Viewing Results

Launch the MLflow UI to compare experiments:

```bash
poetry run mlflow ui
```

This opens a web interface (typically `http://localhost:5000`) where you can:
- Compare accuracy across different prompt versions
- View detailed logs and parameters
- Analyze experiment trends over time
- Export results for further analysis


---

