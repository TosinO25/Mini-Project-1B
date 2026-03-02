# Mini-Project-1B
# Emotion Recognition using EEG and Computer Games

## Project Overview

In this project, you'll work with EEG (electroencephalogram) data to understand how emotions affect brain activity. You'll use real EEG recordings collected while people played computer games and experienced different emotions. Your goal is to analyze the EEG signals, extract features, and build a system that can recognize emotions from brain data. You'll compare different computational approaches and measure their performance trade-offs.

## Background Knowledge

Before starting, you should understand:
- What is EEG and what do the different frequency bands (alpha, beta, gamma, etc.) represent?
- How do emotions change brain activity patterns?
- What features of EEG signals might indicate different emotions?

Explore the reference repository for context: https://github.com/cyneuro/CI-BioEng-Class/tree/main/emotion_recognition

## Project Structure

This project has two experiments:

**Experiment 1: EEG Analysis and Emotion Classification**
- Load and explore EEG data from the provided dataset
- Extract features from EEG signals
- Train a classifier to recognize emotions
- Answer questions about your findings

**Experiment 2: Performance Comparison**
- Compare emotion recognition performance using different methods
- Measure and document time differences
- Report which approach is faster/more accurate

## Environment Setup

**Step 1: Create Virtual Environment**

Using Conda:
```bash
conda create -n eeg_emotion python=3.9
conda activate eeg_emotion
pip install -r requirements.txt
```

Using Pip:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Step 2: Verify Installation**

Test the key packages:
```bash
python -c "import numpy; import pandas; import mne; print('Ready!')"
```

## Getting Started with Experiment 1

**Start by opening the Jupyter notebook:**
```bash
jupyter notebook Experiment_1_EEG_Analysis.ipynb
```

The notebook will guide you through:
1. Loading the EEG dataset
2. Understanding the data structure
3. Visualizing raw EEG signals
4. Computing frequency domain features (FFT)
5. Building a simple classifier

**As you work through the notebook, think about:**
- What patterns do you see in the raw EEG data?
- How do different emotions appear in the frequency domain?
- Which frequency bands (alpha, beta, gamma) change the most with emotion?

## Experiment 1: Questions to Answer

The notebook will ask you specific questions. Your task is to:

1. **Data Exploration**: Describe what you observe in the raw EEG signals. Which channels are most informative? Do you see clear differences between emotional states?

2. **Feature Analysis**: After computing features (like power spectral density), what features seem most important for distinguishing emotions? How did you determine this?

3. **Classification**: What classification accuracy did you achieve? What does this tell you about how well emotions can be recognized from EEG?

Document your answers and findings in a Word document as you work.

## Experiment 2: Time Comparison

**Your task:** Implement or compare two different approaches to emotion recognition:
- **Approach 1**: The method from the reference repository
- **Approach 2**: Your own simplified or alternative method

Measure the execution time for each approach on the same dataset:

```python
import time

start = time.time()
# Run your method here
end = time.time()
print(f"Time taken: {end - start} seconds")
```

**In your Word document, create a table showing:**
- Method name
- Processing time
- Accuracy (if applicable)
- Pros and cons of each approach

**Questions to think about:**
- Why is one method faster than the other?
- Is faster always better? What trade-offs exist?
- How would this scale if you had much more data?

## Understanding EEG Features

As you work, you'll extract features from EEG signals. Common features include:

- **Power Spectral Density (PSD)**: How much signal power exists at different frequencies
- **Frequency Bands**: Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30+ Hz) - each associated with different brain states
- **Statistical Features**: Mean, variance, entropy - characteristics of the signal over time

**Your job:** Determine which features are most useful for emotion recognition. Test different combinations and see what works best.

## Data Files

The EEG dataset should include:
- Raw EEG recordings (.csv or .edf format)
- Associated emotion labels
- Channel information

Load the data and explore its structure first before jumping into analysis.

## Deliverables Checklist

For submission, you need:

- [ ] **Word Document** containing:
  - Answers to Experiment 1 questions (questions 1-2)
  - Time comparison table from Experiment 2
  - Brief explanation of your findings

- [ ] **Completed Jupyter Notebook** with:
  - All code cells executed
  - Visualizations of EEG data
  - Feature extraction code
  - Classifier code
  - Your answers documented in markdown cells

- [ ] **Code runs on both local machine and FABRIC**

## Experiment Workflow

1. **Load Data**: Read the EEG dataset and understand its format
2. **Explore**: Visualize raw signals, check for artifacts, understand the data
3. **Process**: Extract features that represent emotional states
4. **Train**: Build a classifier using extracted features
5. **Evaluate**: Test your classifier and measure performance
6. **Optimize**: Try different features/methods and compare (Experiment 2)
7. **Document**: Write down your findings and answer the questions

## Tips for Success

- Start by just loading and plotting one EEG signal - don't try to do everything at once
- Look for clear patterns in the data before building complex classifiers
- Test with different features to see what improves classification
- Simple models often work as well as complex ones - start simple
- Document what you tried, even if it didn't work well
- Use visualization (plots, heatmaps) to understand what your classifier is learning\

## Running on FABRIC

Once your code works locally, test it on FABRIC:

1. Clone the repository onto FABRIC
2. Install dependencies: `pip install -r requirements.txt`
3. Run the notebook or Python scripts
4. FABRIC is useful for processing larger EEG datasets or testing multiple parameter combinations

## Next Steps

After completing this project, you'll understand:
- How to work with real neuroscience data (EEG)
- How to extract meaningful features from signals
- How to build and evaluate machine learning classifiers
- How to measure and compare computational performance
- How to integrate hardware (micro:bit) with data analysis
