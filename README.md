# implicit-layers

## About
This is the code for the report on `Transformers para Geração de Superfícies Suaves de Volatilidade Implícita`,
in which we aim to approximate the implied volatility surface, or the volatility smirk, 
using Deep Neural Networks (DNN) implemented in Python/PyTorch. 
The method leverages the intuition behind finding fixed points of functions to obtain the coefficients or 
weights necessary for representing the surface.

## Motivation
The implied volatility surface is a crucial concept in quantitative finance, particularly in options pricing. 
It represents the variation in implied volatility across different strike prices and maturities. 
Accurately modeling the implied volatility surface is essential for risk management, 
trading strategy development, and pricing exotic derivatives. 

Traditional methods often use some form of interpolation techniques, or simple neural networks. 
By employing a Transformer architecture, this project seeks to provide a time-efficient approach for parameter candidate selection,
which will in turn be used for approximating the implied volatility surface using a parametric SABR model.

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
3. There are some tests you can perform to see the code in action:
   ```
   python tests/test_parametric.py
   python tests/test_sst.py
   python tests/test_sabr.py
   ```
4. To train and test the model, call the main script:
   ```
   python main.py
   ```
   

**_NOTE_**:
- **Preprocessing** the original data takes about **20 minutes**.

- **Training** the model takes about **10 minutes** on a GPU, 
and the model can be used to predict the implied volatility surface in real-time.

- **Testing** is much faster, but requires preprocessed data and trained model weights.

- You can use the trained model to approximate the implied volatility surface for new options data.

## References
The paper and all references used wherein can be found under the `paper/` directory.
Access the report PDF directly [here](paper/paper.pdf).