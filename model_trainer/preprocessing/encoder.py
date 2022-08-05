from sklearn.base import TransformerMixin
import pandas as pd

class Encoder(TransformerMixin):
    def __init__(self,categorical_columns):
        self.categorical_columns=categorical_columns
    
    def fit(self,X):
        X_new = self._transform(X)
        self.columns=X_new.columns
        return self
    
    def transform(self,X):
        X_new = self._transform(X)
        X_new = X_new.reindex(columns=self.columns, fill_value=0)
        return X_new

    def _transform(self,X):
        for column in self.categorical_columns:
            tempdf = pd.get_dummies(X[column], prefix=column,drop_first=True)
            X = pd.merge(
                left=X,
                right=tempdf,
                left_index=True,
                right_index=True,
            )
            X = X.drop(columns=column)
        tempdf = pd.get_dummies(X['Weather_Condition_Arr'].explode()).groupby(level=0).sum()
        X = pd.merge(
            left=X,
            right=tempdf,
            left_index=True,
            right_index=True
        )
        X = X.drop(columns="Weather_Condition_Arr")
        return X

    