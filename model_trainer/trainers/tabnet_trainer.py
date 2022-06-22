from fastai.tabular.all import *
from fast_tabnet.core import *
from sklearn.utils.class_weight import compute_class_weight
from model_trainer.trainers.base_trainer import BaseTrainer

class TabNetTrainer(BaseTrainer):
    def __init__(self,X_train_val,Y_train_val,cat_vars=[],reg_vars=[],vtype="k-fold", k=3,split=0.8,epochs=50):
        self.epochs=epochs
        super().__init__(X_train_val,Y_train_val,cat_vars,reg_vars,vtype, k,split)


    def _train_model(self,params,X_train,y_train,X_val,Y_val):
        bs= params.pop('batch_size')
        lr= params.pop('lr')
        to = self._fastaify_data(X_train,y_train,X_val,Y_val)
        dls = to.dataloaders(bs)
        
        optimizer = params.pop('optimizer')
    
        model = TabNetModel(get_emb_sz(to), len(to.cont_names), dls.c,**params)
        class_weights = self._get_weights(y_train)
        learn = Learner(dls, model,CrossEntropyLossFlat(weight=class_weights), opt_func=optimizer, lr=lr, metrics=[MatthewsCorrCoef()])
        learn.fit_one_cycle(self.epochs)
        return learn
    
    def _validate_model(self,model,X_val=None,Y_val=None):
        return float(model.validate()[1])
    
    def _get_weights(self,Y_train):
        class_weights=compute_class_weight('balanced',classes=[0,1,2,3],y=Y_train)
        class_weights=torch.tensor(class_weights,dtype=torch.float)
        return class_weights
        
    def _fastaify_data(self,X_train,Y_train,X_val,Y_val):
        train = pd.merge(
            left=X_train,
            right=Y_train,
            left_index=True,
            right_index=True,
        )

        val = pd.merge(
            left=X_val,
            right=Y_val,
            left_index=True,
            right_index=True,
        )

        train_len = len(train)
        val_len = len(val)
        splits = [list(range(0,train_len)),list(range(train_len,train_len+val_len))]

        train_val = pd.concat([train,val])
        train_val.reset_index(drop=True)
        
        cont_names = ['Start_Lat','Start_Lng','End_Lat','End_Lng','Distance(mi)',
            'Temperature(F)','Wind_Chill(F)','Humidity(%)','Pressure(in)',
            'Visibility(mi)','Wind_Speed(mph)','Precipitation(in)','Wind_SN',
            'Wind_EW']

        cat_names = [col for col in train_val.columns]
        _=[cat_names.remove(cont_name) for cont_name in cont_names+['Severity']]
        
        to = TabularPandas(
            train_val, 
            [Categorify,FillMissing], 
            cat_names, cont_names, 
            y_names='Severity', 
            y_block = CategoryBlock(), 
            splits=splits
        )
        
        return to