# Localized Image Quality Mapping of BigBrain Histology Data Using Patch-Based Image Analysis 

This repository contains the code and experiments for our project on quantitative image quality assessment in histological brain images using spectral-based metrics.

The primary goal is to evaluate whether frequency-domain metrics (e.g., spectral cutoff frequency, spectral energy, spatial energy) correlate with:

- Image degradation
- Downstream biological analysis reliability 
- Structural differences in brain regions

The implementation is provided as a collection of Jupyter notebooks corresponding to different stages of the pipeline.

## Notebook Descriptions

- preprocessing.ipynb  
Prepares raw BigBrain data into patches and performs initial cleaning.  
Requires full dataset

- downsampling.ipynb  
Generates multi-resolution data and simulates degradation.  
Requires full dataset

- patch_quality_reliability_analysis.ipynb  
Evaluates the relationship between quality metrics and segmentation reliability.  
Requires full dataset

- spectral_cutoff_frequency.ipynb  
Implements and computes spectral-based quality metrics on image patches.  
Runnable with provided sample patches

- image_verification.ipynb
Validates metric behavior under controlled degradations (blur and noise).  
Runnable with provided sample patches

## Setup Instructions

### 1. Environment

`pip install numpy scipy matplotlib scikit-image opencv-python notebook`

### 2. Running the Code


The following notebooks can be executed without the full dataset:

- `spectral_cutoff_frequency.ipynb`
- `image_verification.ipynb`

We provide a small set of sample patches in: `sample_patches/`. 
These are sufficient to

- Compute all quality metrics
- Reproduce degradation experiments
- Visualize metric behavior

The complete pipeline requires the full BigBrain dataset **(~20GB)**, which is not included in this repository but more data can be found [here]().

## Expected Results

Using the provided sample patches, the following behaviors should be observed:

- Blur degradation → decreases high-frequency content → decreases quality metrics
- Noise injection → increases high-frequency content → may artificially inflate metrics
- Segmentation reliability correlates positively with metrics under blur, but inversely under noise


This demonstrates that spectral metrics primarily capture high-frequency content, which aligns with structural detail but can be confounded by noise.