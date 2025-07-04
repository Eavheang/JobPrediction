﻿# JobPrediction

A machine learning project for predicting job positions based on resume data. This project uses LSTM and DistilBERT models for classification.

## Project Structure

- `checkpoints/` - Contains trained model checkpoints
  - `distilbert_epoch_150.pt` - DistilBERT model checkpoint
  - `lstm_epoch_150.pt` - LSTM model checkpoint
- `Dataset/` - Contains the training data
  - `resume_dataset - gpt_dataset.csv` - Resume dataset for training
- `web.py` - Web interface for the prediction system

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the web interface:
```bash
python web.py
```
3. Download the Model here : https://drive.google.com/file/d/1uMZmEhBs7s2MD7eOUPFnYyyRVdpQjEKn/view?usp=sharing, https://drive.google.com/file/d/1Y6F9OxpvlJpG6mn-dl0rX_7AbbfcPHtT/view?usp=sharing
4. Download the Datasets : https://drive.google.com/file/d/1wbcYQKPQOgSRsMr_RPN1zFJe0JMsei3p/view?usp=sharing
