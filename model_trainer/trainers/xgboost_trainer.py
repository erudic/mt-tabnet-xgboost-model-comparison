from sklearn.metrics import matthews_corrcoef
from sklearn.utils import class_weight
from model_trainer.trainers.base_trainer import BaseTrainer
import xgboost

class XGBoostTrainer(BaseTrainer):
    def _train_model(self,params,weights,X_train,y_train,X_val,Y_val):
        model = xgboost.XGBClassifier(**params,verbosity=2)
        
        classes_weights = class_weight.compute_sample_weight(
            class_weight='balanced',
            y=y_train
        )

        model.fit(X_train, y_train, sample_weight=classes_weights)
        return model
    
    def _validate_model(self,model,X_val=None,Y_val=None):
        return matthews_corrcoef(Y_val,model.predict(X_val))