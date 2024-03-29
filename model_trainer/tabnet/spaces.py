from hyperopt import STATUS_OK,hp,tpe,Trials,fmin
from hyperopt.pyll import scope
import numpy as np

tabnet_large_space={
    "lookahead": hp.choice("lookahead",[False,True]),
    "optimizer": hp.choice('optimizer',[
        {
            "opttype":"Adam",
             "wd":hp.loguniform('wdadam', np.log(0.0001), np.log(0.3))
        },
        {
            "opttype":"SGD",
            "wd":hp.loguniform('wdsgd', np.log(0.0001), np.log(0.3))
        },
        {
            "opttype":"RAdam",
            "wd":hp.loguniform('wdradam', np.log(0.0001), np.log(0.3))
        }
    ]),
    "n":scope.int(hp.choice("n",[8,32,64])),
    "n_steps":scope.int(hp.quniform("n_steps",3,10,1)),
    "gamma":hp.uniform("gamma",1,2),
    "momentum":hp.uniform("momentum",0,1),
    "lr":hp.choice("lr",[0.005,0.01,0.02,0.025]),
    "batch_size":hp.quniform("batch_size",12,15,1),
    "virtual_batch_size":hp.quniform("virtual_batch_size",8,11,1)
    }