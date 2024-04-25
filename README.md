# implicit-layers

## About
This is the code referering to the paper `Implied Volatility Surface Approximation using Implicit Deep Neural Networks`,
in which we aim to approximate the implied volatility surface, or the volatility smirk, 
using Implicit Deep Neural Networks (IDNN) implemented in Python/PyTorch. 
The method leverages the intuition behind finding fixed points of functions to obtain the coefficients or 
weights necessary for representing the surface.

## Motivation
The implied volatility surface is a crucial concept in quantitative finance, particularly in options pricing. 
It represents the variation in implied volatility across different strike prices and maturities. 
Accurately modeling the implied volatility surface is essential for risk management, 
trading strategy development, and pricing exotic derivatives. 

Traditional methods often use some form of interpolation techniques, or even explicit neural networks. 
By employing Implicit Deep Neural Networks, 
this project seeks to provide a more interpretable and efficient approach to approximating the implied volatility surface.

## How to Install
1. Clone this repository to your local machine:
   ```
   git clone https://github.com/rzimmerdev/implicit-layers.git
   ```
2. Navigate to the project directory:
   ```
   cd implicit-layers
   ```
3. Install the required dependencies using pip:
   ```
   pip install -r requirements.txt
   ```
   
## How to Run
1. Ensure you have Python and PyTorch installed on your system.
2. Prepare your data: The input data should include a set of options with corresponding market prices and parameters,
    as given in the `archive.zip` dataset, taken from the Kaggle Dataset [option_SPY_dataset_combined.csv](https://www.kaggle.com/datasets/shawlu/option-spy-dataset-combinedcsv).
3. Train the Implicit Deep Neural Network:
   ```
   python src/train.py --data dataset/option_SPY_dataset_combined.csv
   ```
   Replace `/path/to/your/data.csv` with the path to your dataset.
4. Once trained, you can use the trained model to approximate the implied volatility surface for new options data.

## References
... Ainda falta fazer
