# La Liga 2025/26 Season Predictor

A machine learning-based prediction system for forecasting the final league positions in the 2025/26 La Liga season using historical performance data.

## Overview

This script analyzes historical La Liga seasons from 2019/20 to 2024/25 to predict the final league table for the 2025/26 season. The system uses a Random Forest Regressor to learn patterns from team performance statistics and predict future finishing positions.

## Features

- **Comprehensive Data Processing**: Handles missing data, duplicates, and data quality issues
- **Advanced Feature Engineering**: Includes per-game statistics and year-over-year performance trends
- **Smart Promoted Team Handling**: Automatically assigns realistic baseline statistics for newly promoted teams
- **Robust Prediction Model**: Uses Random Forest with proper feature scaling
- **Professional Output**: Clean, formatted league table with promoted team indicators

## Key Improvements Made

### 1. **Data Quality & Parsing**
- **Better missing data handling**: Drops rows with missing scores instead of treating them as 0-0 draws
- **Improved deduplication**: Removes duplicate matches and handles data quality issues
- **Robust goal parsing**: Safely converts scores to integers with proper error handling

### 2. **Feature Engineering**
- **Comprehensive statistics**: Added per-game metrics (points per game, goals per game, etc.)
- **Trend analysis**: Includes year-over-year changes in performance
- **Better team representation**: More informative features for the machine learning model

### 3. **Model Improvements**
- **Regression approach**: More appropriate for predicting continuous position values
- **Optimized algorithm**: Random Forest Regressor with tuned hyperparameters
- **Proper scaling**: StandardScaler for better model performance

### 4. **Promoted Teams Handling**
- **Realistic positioning**: Promoted teams are placed at the bottom (ranks 18-20)
- **Performance-based ordering**: Maintains relative performance differences among promoted teams
- **Default feature generation**: Creates reasonable baseline statistics for new teams

### 5. **Code Quality**
- **Professional documentation**: Comprehensive docstrings with parameters and return values
- **Better error handling**: Robust error handling and informative messages
- **Clean structure**: Well-organized functions with clear responsibilities

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Ensure all required CSV files are in the same directory as the script
2. Run the prediction:

```bash
python3 laliga_2025-26_prediction.py
```

## Input Data Format

The script expects CSV files with the following columns:
- `Date`: Match date
- `Team 1`: Home team
- `FT`: Final score (e.g., "2-1")
- `HT`: Half-time score (optional)
- `Team 2`: Away team

## Output

The script outputs a predicted league table showing:
- Final rank (1-20)
- Team name
- Predicted position (decimal value)
- Promoted team indicator (P)

## Example Output

```
============================================================
PREDICTED LA LIGA 2025/26 TABLE
============================================================
Rank Team                 Predicted Position  
------------------------------------------------------------
1    Real Madrid          2.0                 
2    Ath Madrid           2.3                 
3    Barcelona            2.5                 
...
25   Levante              18.0                 (P)
26   Oviedo               19.0                 (P)
27   Elche                20.0                 (P)

(P) = Promoted team
============================================================
```

## Model Details

- **Algorithm**: Random Forest Regressor
- **Features**: 14 comprehensive team statistics
- **Training Data**: 85 samples from 6 seasons
- **Feature Scaling**: StandardScaler for optimal performance

## Promoted Teams for 2025/26

The system automatically handles these promoted teams:
- **Levante**
- **Oviedo** 
- **Elche**

These teams are assigned baseline statistics based on the bottom three teams from the 2024/25 season and are positioned at the bottom of the predicted table.

## Data Sources

The prediction system uses historical La Liga data from:
- 2019/20 season
- 2020/21 season
- 2021/22 season
- 2022/23 season
- 2023/24 season
- 2024/25 season

## Limitations

- Predictions are based on historical performance patterns
- External factors (transfers, injuries, managerial changes) are not considered
- The model assumes relative team performance remains consistent
- Promoted teams are assigned conservative baseline statistics

## Future Improvements

Potential enhancements could include:
- Transfer market data integration
- Injury and suspension tracking
- Managerial change impact analysis
- Home/away performance differentiation
- Seasonal form trends

## License

This project is open source and available under the MIT License.
