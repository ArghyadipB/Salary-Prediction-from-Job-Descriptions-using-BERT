# Salary-prediction-from-Job-Descriptions-using-BERT
A BERT-based deep learning model for salary prediction using job descriptions. The project is based upon predicting the normalized salaries from the 'info' column which is concatenatinations of multiple columns like 'job descriptions', 'company name', 'location' etc. The model performs good on metrics as can be seen from the later part of the notebook. The notebook also includes some result analysis for better insights of model and it's performance.


## Author(s)

**Arghyadip Bagchi**

* LinkedIn profile: https://linkedin.com/in/arghyadip-bagchi
* Kaggle profile: https://www.kaggle.com/arghyadipbagchi


## Features


* BERT for regression

* The model performs good on test dataset producing mse of 28,000



## Tech Stack

  
*  Numpy, pandas, matplotlib, pytorch and transformers


## Description of the project

### Data Preprocessing:
* The notebook uses the LinkedIn Job Posting dataset [1], focusing on US-based yearly salaries with descriptions.
* Cleans the text data by removing irrelevant characters, URLs, email addresses, and extra spaces.
* Handles missing values by replacing them with empty strings.
Filters data to keep only relevant job postings with yearly salaries in USD and above a minimum threshold.
* Combines relevant columns (company name, title, experience level, work type, location, skills, and description) into a single 'info' column for model input.
* Performs outlier removal based on the interquartile range (IQR) of the salary data to improve model robustness.

### Model Architecture:
* Uses a pre-trained BERT model (bert-base-uncased) as a feature extractor.
* Adds a regression head (a linear layer) on top of BERT to predict the salary.

### Training Details:
* Applies log transformation to the salary data to handle its skewed distribution.
* Uses a data collator for dynamic padding to handle variable-length input sequences.
* Employs training upto 6 epochs (with train batch size = 16, max_len of allowed token as 512, a learning rate of 0.5e-5 and Huber loss with delta set as 1.0 as loss function) with validation loss tracing to prevent overfitting.
* Saves the best-performing model checkpoint during training.

### Evaluation Metrics:
* RMSE is used as the primary evaluation metric to measure the accuracy of salary predictions.
* Additional analysis is performed to understand the distribution of prediction errors and identify potential areas for improvement.
* Correlation between actual and predicted salaries is calculated to assess the model's overall performance.

### Reproducibility:
* Sets random seeds to ensure consistent results across different runs.
* Provides clear instructions for loading the saved model and performing inference.

## Observations:

* The model gets underfitted for epochs greater than 5-6 epochs
* BERT limits its token length to 512 tokens which is found out to be not suitable for this application as most of the text in the job description column is of very long text type. Language models with larger token limit like longformer tend to go out of gpu memory in my case.

## Reference

[1] Linked Job Posting dataset: https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
