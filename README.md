This repo contains code used to perform experiments in our paper about generating counterfactual explanations in the context of time-series remote sensing multiclass data. 

Experiments are composed of the following blocks:
1. **[Model training]** Training a classifier model
2. **[Noiser training]** Training the counterfactual generative model based on an existing classifier
3. **[Noiser application]** Use the noiser model to produce counterfactual examples
4. **[Result analysis]** Scripts and notebooks registering metrics, plots and tables for the paper

## Model training
- `main_sits_model.py`: trains a classifier

## Noiser training
- `main_regDiego.py`: trains a noiser ("noiser_weights") from a base model ("model_weights_tempCNN")
-  [FUTURE] `main_multi.py`: trains a noiser for a multivariate case (for now the scripts seems to train a classifier as well?). This is halted now and kept for the future.
- [DEPRECAETD] `main_newSample.py`:  also trains a noiser
- [DEPRECAETD] `main.py`:  also trains a noiser

## Noiser application
Files concerning this part:
- `reject_option.py`: loads model and noiser, creates CF examples, computes accuracy
- [FAILED] `DataAugmentation.py`:  an attempt to improve the base classifier by reuse generated CF samples as data agumentation.

## Result analysis
- `ExtractCF.py`: generates several results presented in the paper (chord graph, perturbation examples, average perturbation) and some not yet included (PCA)
- `IF_evaluation.py`: plausability study (via Isolation forest)
- `plot_time_line.py` : plots a timeline based on `dates.txt`
- [DEPRECATED?] `generateCF.py` : loads model and noiser, computes predictions and CFs, then produces the corresponding chord diagram linking OG and CF classes.

## Geral use files
- `model.py` contains declaration of different models used by other scripts

## Data files
- all `[x|y]_[train|valid|test]_2020.npy` files
- `dates.txt`