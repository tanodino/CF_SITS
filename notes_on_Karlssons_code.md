# Two repos
 - the og messy repo `tstransform`
 - the new package repo `wildboard`

## On the old tstransform code
No license is specified.
### KNN explainer
`NearestNeighbourLabelTransformer` imposes no constraints on the base estimator. 
It does support multiclass, but needs a destination class to produce the counterfactual. 
However, it creates internally a k-NN classifier, trained on the input data. This classifier is only used if we call `predict`. For our experiments, we can rely on `fit` and `transform` only.

`fit` implements the following:
1. Fits KMeans on some input tabular data x
2. For each cluster, determines the class distribution (using given labels y)
3. Uses mask arrays to select only centers on which 
    - destination class freq > 1/C and (C being the number of classes)
    - destination class count > 1 + k/C  (k being the number of neighbors to be used in the kNN base estimator)
4. If any cluster centers remained after filtering, Fits a 1-NN model (`nn_`) on them
5. Fits a k-NN model on the input data (x,y) (k being an input parameter)

`transform` implements the following:
1. applies the 1-nn model to given data x, getting the cluster center closest to it
2. returns the retrieved cluster center

### random forest explainer
`GreedyTreeLabelTransform` trains the Shapelet forest internally, and uses it to extract decision paths. When it tries to change decisions, it does while navigating these paths. As it is, this method only makes sense when applied to tree-based classifiers.

`fit` implements the following:
1. train a bagged ensemble of Shapelet trees
2. In a dictionary of lists, store decision paths (root to leaf) for each tree in the forest, categorized by the class they lead to.

`transform` implements:
1. given a sample x_i, 
    - path transformations are computed for all paths leading to the target class
    - each path gets a cost evaluation
    - the transformed path with minimal cost is selected
2. returns minimal cost transformed paths for each input sample (row) in x

## On wildboar
Released under BSD license.

Implements the kNN and random forest CF explainers, restricting them to be applied only to scikit-learn-like estimators of the respective types. That is, estimators need to be an instance of:
- KNeighborsClassifier from sklearn.neighbors
- ShapeletForestClassifier wildboar.ensemble.BaseShapeletForestClassifier

### KNN explainer
- in `explain/counterfactual/_nn.py`
- uses an input estimator
    - accesses n_neighbors on it
- n_clusters is specified using num_samples/k (k being the number of neighbors used in the base estimator)


### Shapelet random forest explainer
- in `explain/counterfactual/_sf.py`
- from the given estimator, it accesses:
    - tree_
    - classes_
    - predict

## Experiment strategy
- I can re-implement wild boar class locally via inheritance and overwrite the `_validate_estimator` method. --> cleaner and easier to ensure reproducibility
