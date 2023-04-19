This repo contains code used to perform experiments in our paper about generating counterfactual explanations in the context of time-series remote sensing multiclass data. 
# Instructions to reproduce paper results
1. **Set up virtual environment**:  
`conda env create -f environment.yml && conda activate cf-env`
2. **Produce CFE4SITS results**:  
`python ExtractCF.py --do-plots`
3. **Produce NG results**:  _Disclaimer: takes several hours to complete._
`python competitor_NG.py pred --use-cam && python competitor_NG.py results --use-cam`
4. **Produce k-NNC results**:
`python competitor_knn.py train && python competitor_knn.py pred && python competitor_NG.py results`
5.**To generate comparative plots**
`python selected_plots.py`

All plots get saved in the folder `img`. A metrics summary is logged both to the standard output and to a text file in `logs`.

# Codebase descritption
The project can be divided into the following functional blocks:
1. **[Model training]** Training a classifier model
2. **[Noiser training]** Training the counterfactual generative model based on an existing classifier
3. **[Result analysis]** Scripts and notebooks registering metrics, plots and tables for the paper
4. **[General utils]** General utility files, be them scripts or data

## Model training
Files concerning this part:
- `main_sits_model.py`: trains a classifier

## Noiser training
Files concerning this part:
- `main_noiser.py`: trains a noiser ("noiser_weights") from a base model ("model_weights_tempCNN")
-  [FUTURE] `main_multi.py`: trains a noiser for a multivariate case (for now the scripts seems to train a classifier as well?). This is halted now and kept for the future.

## Result analysis
Files concerning this part:
- `ExtractCF.py`: generates several results presented in the paper (chord graph, perturbation examples, average perturbation) and some not yet included (PCA)

## General use files
### Reusable python modules
Modules used across several scripts have been assembled in the `cfsits_tools` folder/module.
<!-- - `model.py` contains declaration of different models used by other scripts -->

### Data files
Files concerning this part:
- all `[x|y]_[train|valid|test]_2020.npy` files
- `dates.txt`