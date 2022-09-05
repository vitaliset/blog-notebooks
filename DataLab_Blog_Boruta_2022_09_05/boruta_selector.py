from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier

from boruta import BorutaPy

import numpy as np

class Boruta(SelectorMixin, BaseEstimator):

    def __init__(self, n_estimators=1000, perc=100, alpha=0.05,
    max_iter=100, two_step=True, include_support_weak=False,
    base_estimator=None, random_state=None):
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.max_iter = max_iter
        self.two_step = two_step
        self.include_support_weak = include_support_weak
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

        self._validate_estimator(
            RandomForestClassifier(max_depth=5, random_state=self.random_state))

        boruta = \
        (BorutaPy(
            estimator=self.base_estimator_,
            n_estimators = self.n_estimators,
            perc = self.perc,
            alpha = self.alpha,
            two_step = self.two_step,
            max_iter = self.max_iter,
            random_state=self.random_state)
            .fit(np.array(X), np.array(y))
        )

        if self.include_support_weak:
            self._selected_index_list_ = boruta.support_ + boruta.support_weak_
        else:
            self._selected_index_list_ = boruta.support_

        return self

    def _get_support_mask(self):
        check_is_fitted(self)
        return self._selected_index_list_