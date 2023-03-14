This repo contains code used to perform experiments in our paper about generating counterfactual explanations in the context of time-series remote sensing multiclass data. 

The project can be divided into the following functional blocks:
1. **[Model training]** Training a classifier model
2. **[Noiser training]** Training the counterfactual generative model based on an existing classifier
3. **[Noiser application]** Use the noiser model to produce counterfactual examples
4. **[Result analysis]** Scripts and notebooks registering metrics, plots and tables for the paper
5. **[General utils]** General utility files, be them scripts or data

## Model training
Files concerning this part:
- `main_sits_model.py`: trains a classifier

## Noiser training
Files concerning this part:
- `main_regDiego.py`: trains a noiser ("noiser_weights") from a base model ("model_weights_tempCNN")
-  [FUTURE] `main_multi.py`: trains a noiser for a multivariate case (for now the scripts seems to train a classifier as well?). This is halted now and kept for the future.


## Noiser application
Files concerning this part:
- `reject_option.py`: loads model and noiser, creates CF examples, computes accuracy
- [FAILED] `DataAugmentation.py`:  an attempt to improve the base classifier by reuse generated CF samples as data augmentation.

## Result analysis
Files concerning this part:
- `ExtractCF.py`: generates several results presented in the paper (chord graph, perturbation examples, average perturbation) and some not yet included (PCA)
- `IF_evaluation.py`: plausibility study (via Isolation forest)
- `plot_time_line.py` : plots a timeline based on `dates.txt`
- [DEPRECATED?] `generateCF.py` : loads model and noiser, computes predictions and CFs, then prints the class transitions as matrix.

## General use files
### Reusable python modules
Files concerning this part:
- `model.py` contains declaration of different models used by other scripts

### Data files
Files concerning this part:
- all `[x|y]_[train|valid|test]_2020.npy` files
- `dates.txt`