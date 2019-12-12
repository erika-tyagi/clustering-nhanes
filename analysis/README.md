## Analysis 

The full analyses used in this project are contained in this repository. 

This analysis is sequential in that `pca_clustering.R` uses the `NHANES-clean.csv` file generated in the `process-raw-data` folder as an input and outputs `clustered_data.csv`. `clustered_data.csv` is then used by `Supervised-classifiers.ipynb` and `regression.R`. 

Note that `clustered_data.csv` contains the demographic variables for each individual along with their principal components and cluster assignment – but not the original nutritional features. The full dimension reduction and clustering work is contained in `pca_clustering.R`. The full supervised work is contained in `Supervised-classifiers.ipynb`, which leverages helper functions contained in `sk_models.py`. The full regression analysis is contained in `regression.R`. 
