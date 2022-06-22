from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from model_trainer.preprocessing.feature_selection import FeatureSelector
from model_trainer.preprocessing.encoder import Encoder

class BaseTrainer(ABC):
    def __init__(self,X_train_val,Y_train_val,cat_vars=[],reg_vars=[],vtype="k-fold", k=3,split=0.8):
        self.vtype=vtype
        self.cat_vars=cat_vars
        if(vtype=="k-fold"):
            self._init_kfold(X_train_val,Y_train_val,cat_vars,reg_vars,k)
            return
        elif(vtype=="hold-out"):
            self._init_hold_out(X_train_val,Y_train_val,cat_vars,reg_vars,split)
        else:
            raise ValueError(f"Invalid type {type} supported types: kfold and hold-out")
            
    def _init_kfold(self,X_train_val,Y_train_val,cat_vars,reg_vars,k):
        self.X_train_val=X_train_val
        self.Y_train_val=Y_train_val
        self.kf = StratifiedKFold(n_splits=k)
        
    def _init_hold_out(self,X_train_val,Y_train_val,cat_vars,reg_vars,split):
        X_train,X_val,Y_train,Y_val = train_test_split(X_train_val,Y_train_val,train_size=split,stratify=Y_train_val)
        self.X_train,self.X_valid = self._select_and_encode(X_train, X_val)
        self.Y_train,self.Y_valid = self._encode_target(Y_train,Y_val)
        
        
    def _split_using_index(self,train_index,valid_index):
        X_train = self.X_train_val.iloc[train_index].copy()
        X_valid = self.X_train_val.iloc[valid_index].copy()
        Y_train = self.Y_train_val.iloc[train_index].copy()
        Y_valid = self.Y_train_val.iloc[valid_index].copy()
        return X_train, X_valid, Y_train, Y_valid

    def train_and_validate(self,params):
        if(self.vtype=="k-fold"):
            valid_score=self._train_and_validate_kfold(params)
        if(self.vtype=="hold-out"):
            valid_score=self._train_and_validate_hold_out(params)
        return valid_score
            
    def _train_and_validate_kfold(self,params):
        valid_scores = []
        for train_index, valid_index in self.kf.split(self.X_train_val,self.Y_train_val):
            X_train, X_valid, Y_train, Y_valid = self._split_using_index(train_index,valid_index)
            Y_train, Y_valid = self._encode_target(Y_train,Y_valid)
            X_train, X_valid = self._select_and_encode(X_train,X_valid)
            trained_model = self._train_model(params,X_train,Y_train,X_valid,Y_valid)
            valid_score = self._validate_model(trained_model,X_valid,Y_valid)
            valid_scores.append(valid_score)
        return np.mean(valid_scores)
    
    def _train_and_validate_hold_out(self,params):
        trained_model = self._train_model(params,self.X_train,self.Y_train,self.X_valid,self.Y_valid)
        valid_score = self._validate_model(trained_model,self.X_valid,self.Y_valid)
        return valid_score
    
    @abstractmethod
    def _train_model(self,params,X_train,y_train,X_val=None,Y_val=None):
        raise NotImplementedError
    
    @abstractmethod
    def _validate_model(self,model):
        raise NotImplementedError
        
    def _select_and_encode(self,X_train, X_valid):
        fs = FeatureSelector()
        X_train = fs.fit_transform(X_train)
        X_valid = fs.transform(X_valid)
        encoder = Encoder(self.cat_vars)
        X_train = encoder.fit_transform(X_train)
        X_valid = encoder.fit_transform(X_valid)
        return X_train,X_valid
    
    def _encode_target(self,Y_train, Y_valid):
        return Y_train-1, Y_valid-1