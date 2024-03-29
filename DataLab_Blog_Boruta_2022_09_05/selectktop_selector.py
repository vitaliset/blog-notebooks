from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np

# somewhat equivalent to 
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html
class SelectKTop(SelectorMixin, BaseEstimator):

    def __init__(self, K=5, base_estimator=None, random_state=None):
        self.K = K
        self.base_estimator = base_estimator
        self.random_state = random_state

    def _validate_estimator(self, default):
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default

    def fit(self, X, y):

        X = self._validate_data(
            X,
            dtype=np.float64,
        )

        self._validate_estimator(RandomForestClassifier(random_state=self.random_state))

        self.base_estimator_.fit(X, y)

        self._selected_index_list_ = \
        (pd.DataFrame(list(zip(range(self.n_features_in_), self.base_estimator_.feature_importances_)),
                      columns=['feature_name', 'feature_importance'])
         .sort_values(by='feature_importance', ascending=False)
         .head(self.K)
         .index
         .to_list()
        )

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return np.array([feat in self._selected_index_list_ for feat in range(self.n_features_in_)])
