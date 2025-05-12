# Car Price Prediction - Data Preprocessing

This project contains a PySpark-based data preprocessing pipeline for car price prediction.

## Project Structure


```
car_price_prediction/
├── config/                  # Configuration files
├── data/                    # Data files
│   ├── raw/                 # Raw input data
│   ├── interim/             # Intermediate processed data
│   └── processed/           # Final processed datasets
├── notebooks/               # Jupyter notebooks
├── src/                     # Source code
│   ├── preprocessing/       # Preprocessing pipeline
│   └── utils/               # Utility functions
├── tests/                   # Unit tests
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```


## Setup

1. Clone the repository
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place your raw data file in the `data/raw/` directory

## Usage

Run the preprocessing pipeline with:

```bash
python -m src.main
```

The preprocessed data will be saved in the `data/processed/` directory.

## Pipeline Steps

1. **Data Preparation**: Load data, transform values, drop useless columns
2. **Feature Engineering**:
   - Handle missing values
   - Extract features from date
   - Split equipment column into multiple boolean columns
   - Rename columns to English
3. **Variable Encoding**:
   - Convert boolean columns to integers
   - Convert numeric columns to correct type
   - Label encode categorical columns

## Configuration

All parameters are stored in `config/config.yaml`. You can modify:
- File paths
- Spark configuration
- Columns to process
- Equipment types and mappings

## Development

To add new functionality:
1. Add utility functions in `src/utils/`
2. Update the pipeline in `src/preprocessing/pipeline.py`
3. Update the configuration in `config/config.yaml`
4. Add tests in `tests/`
