import numpy as np
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted

class SHAPImportanceRandomForestClassifier(RandomForestClassifier):

    def fit(self, X, y, sample_weight=None):
        self.X_ = X
        super().fit(X, y, sample_weight=sample_weight)
        return self

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_)
        return np.abs(shap_vals[0]).mean(axis=0)

class XSHAPImportanceRandomForestClassifier(RandomForestClassifier):

    def __init__(
        self,
        X_shap=None,
        n_estimators=100,
        *,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
    ):

        if X_shap is None:
            raise ValueError("X_shap should be always be passed. It should be of the same style as X used in fit.")
        else:
            self.X_shap = X_shap

        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
        )

    @property
    def feature_importances_(self):
        check_is_fitted(self)
        explainer = shap.TreeExplainer(self)
        shap_vals = explainer.shap_values(self.X_shap)
        return np.abs(shap_vals[0]).mean(axis=0)