from sklearn.base import BaseEstimator,TransformerMixin

class WeatherConditionTransformator(BaseEstimator, TransformerMixin):
    def fit(self, X):
        counts = X['Weather_Condition_Arr'].explode().value_counts()
        self.top_weathers = counts[:10]
        return self
    
    def _weather_condition_mapper(self,weather_condition_arr):
        weathers = set()
        for w in weather_condition_arr:
            if w in self.top_weathers:
                w_to_add = w
            else:
                w_to_add='Weather_Other'
            weathers.add(w_to_add)
        return list(weathers)
            
    def transform(self, X, y=None):
        X['Weather_Condition_Arr'] = X['Weather_Condition_Arr'].map(self._weather_condition_mapper,na_action='ignore')
        return X


class StateTranformator(BaseEstimator,TransformerMixin):
    def fit(self,X):
        self.top_states = X['State'].value_counts()[:10].index
        return self
        
    def _map_state(self,state):
        if state in self.top_states:
            return state
        else:
            return 'Other'
    
    def transform(self,X):
        X['State']=X['State'].map(self._map_state,na_action='ignore')
        return X


class FeatureSelector(TransformerMixin):
    def __init__(self):
        self.wct = WeatherConditionTransformator()
        self.st = StateTranformator()
        
    def fit(self, X):
        self.wct.fit(X)
        self.st.fit(X)
        return self
        
    def transform(self,X):
        X = self.wct.transform(X)
        X = self.st.transform(X)
        return X
