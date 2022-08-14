from hyperopt import hp
from hyperopt.pyll import scope
import numpy as np

tabnet_shared_space={
    "optimizer": hp.choice('optimizer',[
        {
            "opttype":"Adam",
            "wd":hp.loguniform('wdadam', np.log(0.0001), np.log(0.3)),
            "lr":hp.loguniform("lr_adam",np.log(0.005),np.log(0.025)), 
            "lookahead": hp.choice("lookahead_adam",[False,True])
        },
        {
            "opttype":"SGD",
            "wd":hp.loguniform('wdsgd', np.log(0.0001), np.log(0.3)),
            "lr":hp.loguniform("lr_sgd",np.log(0.005),np.log(0.025)),
            "lookahead": hp.choice("lookahead_sgd",[False,True])
        },
        {
            "opttype":"RAdam",
            "wd":hp.loguniform('wdradam', np.log(0.0001), np.log(0.3)),
            "lr":hp.loguniform("lr_radam",np.log(0.005),np.log(0.025)), 
            "lookahead": hp.choice("lookahead_radam",[False,True])
        }
    ]),
    "n":scope.int(hp.choice("n",[8,32,64])),
    "n_steps":scope.int(hp.quniform("n_steps",3,10,1)),
    "gamma":hp.uniform("gamma",1,2),
    "momentum":hp.uniform("momentum",0,1),
    "cw_modifier": hp.uniform("cw_modifier",0.5,1.5)
    }

spaces = {
    "tabnet": {
        "large":{
            **tabnet_shared_space,
            "batch_size":hp.quniform("batch_size",12,15,1),
            "virtual_batch_size":hp.quniform("virtual_batch_size",8,11,1)
        }
    }
}

base_class_weights_large = [0.3, 4.6, 27, 5]