import torch
import numpy as np

class ModelMixing:
    def __init__(self, base_model, main_model, target_model, is_good, seed = 80085):
        self.base_model = base_model
        self.main_model = main_model
        self.target_model = target_model
        self.is_good = is_good
        self.random_state = np.random.RandomState(seed)
        
    def init_masks(self):
        self.masks = []
        for p in self.main_model.parameters():
            self.masks.append(torch.Tensor(self.random_state.uniform(low=0.01, high=0.1, size=p.shape)))
            
    def start_mixing(self):
        self.init_masks()
        while not self.is_good(self.target_model):
            for p_base, p_main, p_target, p_mask in zip(self.base_model.parameters(), self.main_model.parameters(), self.target_model.parameters(), self.masks):
                p_target.data = torch.lerp(p_target.data, p_main.data, p_mask)