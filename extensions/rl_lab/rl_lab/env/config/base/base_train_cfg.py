class BaseTrainCfg:
    def __init__(self, exp_name, max_iterations):
        self.algorithm = {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        }
        self.init_member_classes = {}
        self.policy = {
            "class_name": "ActorCritic",
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
        }
        self.checkpoint = -1
        self.experiment_name = exp_name
        self.load_run = -1
        self.log_interval = 1
        self.max_iterations = max_iterations
        self.num_steps_per_env = 24
        self.record_interval = -1
        self.resume = False
        self.resume_path = None
        self.run_name = ""
        self.runner_class_name = "OnPolicyRunner"
        self.save_interval = 100
        self.empirical_normalization = False
        self.seed = 1

def get_cfg(self):
    return self.__dict__