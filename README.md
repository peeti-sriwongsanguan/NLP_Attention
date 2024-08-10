# NLP Attention with Amazon Automotive Review Data

This project demonstrates an attention mechanism as an improvement over the encoder-decoder-based neural machine translation system, using Amazon Automotive Review data.

## Description
This project uses a dataset containing product reviews and metadata from Amazon, including reviews spanning May 1996 - July 2014. The dataset includes:

Reviews (ratings, text, helpfulness votes)
Product metadata (descriptions, category information, price, brand, and image features)
Links (also viewed/also bought graphs)

We use this data to build an LSTM-based sentiment analysis model.


## Features

Data preprocessing and cleaning
Sentiment analysis (positive/negative classification)
LSTM-based deep learning model
Visualization of data and results

## Requirements

Python 3.7+<br>TensorFlow 2.x<br>Keras<br>NumPy<br>Pandas<br>Matplotlib<br>Seaborn<br>NLTK<br>Scikit-learn

## Installation

Clone this repository:
```git clone https://github.com/peeti-sriwongsanguan/nlp-attention-amazon-auto.git```

Install the required packages:
```pip install -r requirements.txt```


## Usage

Prepare your data: Ensure you have the Amazon Automotive Review dataset (reviews_Automotive_5.json.gz) in the project directory.
Run the main script:
```python main.py```

When prompted, enter the path to your dataframe file.

## Model Architecture
The model uses an LSTM architecture with the following layers:
<ol>
<li>Embedding layer</li>
<li>SpatialDropout1D layer</li>
<li>LSTM layer</li>
<li>Dense layer with sigmoid activation</li>
</ol>

## Results
The model's performance can be evaluated based on the test accuracy printed at the end of the script execution.
