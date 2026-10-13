# Prompt log

## Prompt for Streamlit UI to upload 5 Excel files and select columns to train data on

I am building an MMM engine in Python.

For the first version, I want:

- The page should have a clean title: "Marketing Mix Model engine"
- Allow the user to upload 5 Excel (.xlsx) files
- Read the Excel files using pandas
- Collate data from all these files into one in ascending order by date. In case of absence of certain columns in some files, fill them in with an asterisk and highlight them in red when displaying preview data. Make sure all columns have data that is clean and validated, for example, make sure that date mentioned exists in the calendar year, values are numeric etc.
- Display this collated data in a preview table to the user.
- Allow the user to select more than 3 and less than 6 channels containing data to train the MMM engine.
- Validate that the selected column exists and contains numeric data
- Display another table containing these columns, date column and sales column
- Do NOT implement any linear regression or existing MMM or analysis model.
- Keep the code simple and explain what each block does via comments.
- Put the application code in `app.py`

## Generate synthetic data to test UI

Write code to generate 5 months of synthetic sales data for a hypothetical FMCG company; it should create 5 different files I can use to test my UI.

The key requirements are:

- In the first month, the columns should be: date (in short form), sales, TV_spend, Meta_spend, Youtube_spend, PR_spend, Youtube_spend and Instagram_spend
- In the second month, the columns should be: date (in short form), sales, TV_spend, Meta_spend, Youtube_spend, PR_spend, Instagram_spend
- In the third month, the columns should be: date (in long form), sales, TV_spend, Meta_spend, Youtube_spend, PR_spend
- In the fourth month, the columns should be: date (in short form), sales, TV_spend, Meta_spend, Youtube_spend, PR_spend (in words)
- In the fifth month, the columns should be: date (in short form), sales, TV_spend, Meta_spend, Youtube_spend, PR_spend

This is data from a hypothetical FMCG company, so make sure the data fits a logical regression concept such data would have.

- DO NOT apply any analysis, MMM models or linear regression models, only generate synthetic data to train and test a model on
- Do not alter `app.py`
- Put application code in `test_streamlit.py`

## Validate data and get a final collated Excel sheet to train and test model on

Using the final data collected from user, including the columns selected by the user, date and sales, collate this in a separate data frame to be used in the future. Call this data frame `final_data`. Validate the data obtained, display to the user:

- The number of values you have under each channel
- What channels will be considered
- If there are any missing values

Make these changes ONLY in `app.py`, do not alter anything else.

## Training a ridge regression model on historical data

Use the data frame `final_data` obtained from the user from `app.py`. In this data, split it 80-20 in training and test data.

From the training data, separate inputs (x) and sales (y). Inputs (x) will be all the columns in `final_data` except date and sales.

Split this data chronologically and scale the features.

Train Ridge Regression and return the trained model and the data needed for evaluation.

Using the test data, test the model and show all necessary parameters for its accuracy.

- Put the application code in `model_training.py`
- DO NOT make any changes to any other file.

## Show the results of training and test data on the Streamlit interface

Using `model_training.py`, alter `app.py` such that the model evaluation is shown to the user. DO NOT change anything else.

## Add seasonality and adstock to ridge regression model and check accuracy

Alter `model_training.py` for the ridge regression model to handle seasonality and adstock:

- The model should account for adstock in all channels of marketing spend and how they will affect the overall sales
- The model should account for monthly/weekly seasonality if present. The seasonality might be sine natured and the model must figure out its component.

DO NOT make any changes to any other files.

## Add graphical display

Now, using the test data, display a graph showing the actual values and predicted values to the user.

DO NOT change anything else, only plot these values so a user can see what the difference between actual and predicted values is.
