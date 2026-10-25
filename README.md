# Adversarial Robustness of Vision Transformers for Medical Image Classification

A FastAPI-based backend application that demonstrates the robustness of Vision Transformer (ViT) models against adversarial attacks in medical image classification tasks. The system classifies chest X-ray images into four categories: COVID, Lung Opacity, Normal, and Viral Pneumonia.

## Project Overview

This project is designed to evaluate how well Vision Transformers trained with different techniques perform under adversarial attacks. The application provides a REST API for image classification with optional attack generation and explainability features.







## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (optional, for GPU acceleration)
- pip package manager

### Setup

1. **Create and activate virtual environment** (optional but recommended):

   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure model files are in place**:
   ```
   models/
   ├── vit_standard.pth
   ├── vit_pgd.pth
   └── vit_fgsm.pth
   ```

## Running the Application

### Start the FastAPI Server

```bash
# Basic startup
uvicorn main:app --reload

# With specific host and port
uvicorn main:app --host 0.0.0.0 --port 8000

# Production mode (without reload)
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at `http://localhost:8000`

### Interactive API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## API Endpoints

### 1. **Health Check**

```
GET /health
```

**Response**: Server status and loaded models



---

### 2. **Image Classification**

```
POST /classify
```

**Parameters**:

- `file` (File, required): Chest X-ray image (PNG, JPEG, etc.)
- `model` (string, optional): Model selection - `"standard"`, `"pgd"`, or `"fgsm"` (default: `"standard"`)
- `attack` (boolean, optional): Apply FGSM attack to image (default: `false`)
- `epsilon` (float, optional): Attack perturbation magnitude (default: `0.01`)

**Response**: Classification result with confidence scores and heatmap generation info



---

### 3. **Retrieve Heatmap**

```
GET /heatmap/{heatmap_id}
```

**Path Parameters**:

- `heatmap_id` (string): Unique identifier from classification response

**Response** (pending):

```json
{
  "status": "pending"
}
```

**Response** (ready):

```json
{
  "status": "ready",
  "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

**Response** (error):

```json
{
  "status": "error",
  "error": "Error message describing what went wrong"
}
```






---

**Last Updated**: April 2026  
**Project Type**: Final Year Project (FYP) - Adversarial Robustness Research
