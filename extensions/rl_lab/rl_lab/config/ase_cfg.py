from . import *

class Go2ASE():
    def __init__(self,exp_name, max_iterations):
        self.env_cfg = BaseEnvCfg()
        self.train_cfg = BaseTrainCfg(exp_name, max_iterations)
        self.train_cfg.algorithm.class_name = "ASEPPOV1"
        self.train_cfg.policy.class_name = "ASEV1"
        
        
    def get_env_cfg(self):
        return self.env_cfg.get_cfg()
    def get_train_cfg(self):
        return self.train_cfg.to_dict()
    
    