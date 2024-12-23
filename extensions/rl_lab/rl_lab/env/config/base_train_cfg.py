class AlgorithmConfig:
    def __init__(self):
        self.class_name = "PPO"
        self.clip_param = 0.2
        self.desired_kl = 0.01
        self.entropy_coef = 0.01
        self.gamma = 0.99
        self.lam = 0.95
        self.learning_rate = 0.001
        self.max_grad_norm = 1.0
        self.num_learning_epochs = 5
        self.num_mini_batches = 4
        self.schedule = "adaptive"
        self.use_clipped_value_loss = True
        self.value_loss_coef = 1.0

class PolicyConfig:
    def __init__(self):
        self.class_name = "ActorCritic"
        self.activation = "elu"
        self.actor_hidden_dims = [512, 256, 128]
        self.critic_hidden_dims = [512, 256, 128]
        self.init_noise_std = 1.0
class BaseTrainCfg:
    def __init__(self, exp_name, max_iterations):
        self.algorithm = AlgorithmConfig()
        self.init_member_classes = {}
        self.policy = PolicyConfig()
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

    def to_dict(self):
        cfg_dict = self.__dict__.copy()
        cfg_dict['algorithm'] = self.algorithm.__dict__.copy()
        cfg_dict['policy'] = self.policy.__dict__.copy()
        return cfg_dict