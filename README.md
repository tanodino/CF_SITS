This repo contains code used to perform experiments in our paper about generating counterfactual explanations in the context of time-series remote sensing multiclass data.

# Instructions to reproduce ECML_PKDD 2023 paper results
**Please open our [v2 code](https://github.com/tanodino/CF_SITS/releases/tag/v2)** and follow the instructions therein.
<details><summary>Click to read the v2 instructions</summary><quote>

1. **Set up virtual environment**:
`conda env create -f environment.yml && conda activate cf-env`
2. **Produce CFE4SITS results**:
`python ExtractCF.py --do-plots`
3. **Produce NG results**:  _Disclaimer: takes several hours to complete._
`python competitor_NG.py pred --use-cam && python competitor_NG.py results --use-cam`
4. **Produce k-NNC results**:
`python competitor_knn.py train && python competitor_knn.py pred && python competitor_knn.py results`
5.**To generate comparative plots**
`python selected_plots.py`

All plots get saved in the folder `img`. A metrics summary is logged both to the standard output and to a text file in `logs`.
</quote></details>

# Codebase description
The project can be divided into the following functional blocks:
1. **[Model training]** Training a classifier model
2. **[Noiser training]** Training the counterfactual generative model based on an existing classifier
3. **[Result analysis]** Scripts and notebooks registering metrics, plots and tables for publication
4. **[General utils]** General utility files, be them scripts or data
## General instructions
### Running
 Each script can be somewhat parameterized via command line. Please call `python3 SCRIPT_FILENAME.py --help` to get help specific to each script.
### Result logging
All scripts log results to `logs/SCRIPT_FILENAME/RUN_ID_FOLDER`, where `RUN_ID_FOLDER` is an identifier depending on the parameters that were used to run the script. Model weights, hyperparameters, runtime metrics and plots, they all get saved to this folder.
## Model training
Files concerning this part:
- `main_classif.py`: trains a timeseries classifier

## Noiser training
Files concerning this part:
- `main_noiser.py`: trains a noiser model from a base model
-  [FUTURE] `main_multi.py`: trains a noiser for a multivariate case (for now the scripts seems to train a classifier as well?). This is halted now and kept for the future.

## Result analysis
Files concerning this part:
- `ExtractCF.py`: generates several results presented in the paper (chord graph, perturbation examples, average perturbation) and some not yet included (PCA)

## General use files
### Reusable python modules
Modules used across several scripts have been assembled in the `cfsits_tools` folder/module.
<!-- - `model.py` contains declaration of different models used by other scripts -->

### Data files
In the `data` folder there are files corrsponding to the Koumbia dataset:
- all `[x|y]_[train|valid|test]_2020.npy` files
- `dates.txt`

For experiments with the UCR data, a folder `UCRArchive_2018` containing the dataset should be placed here, within `data`.