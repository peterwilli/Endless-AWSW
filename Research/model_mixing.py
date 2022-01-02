import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class ModelMixing:
    def __init__(self, tokenizer, base_model, main_model, target_model, cold_zone_loss, seed = 80085):
        self.base_model = base_model
        self.main_model = main_model
        self.target_model = target_model
        self.random_state = np.random.RandomState(seed)
        self.tokenizer = tokenizer
        self.cold_zones = None
        self.cold_zone_loss = cold_zone_loss
        
    def diffuse_cold_zone(self, cold_zone, diffusion, lower_bound):
        result = torch.clone(cold_zone).numpy()
        result[result <= lower_bound] = diffusion
        result[result > 1] = 1
        
        return torch.Tensor(result)
    
    def calculate_cold_zones(self):
        restart_tries = 0
        successful_tries = 0
        cold_zone_tmp = []
        self.cold_zones = []
        for p in self.main_model.parameters():
            self.cold_zones.append(torch.zeros(p.shape))
            cold_zone_tmp.append(torch.zeros(p.shape))
            
        amplification = 0.90
        while True:
            for i in range(len(cold_zone_tmp)):
                r = torch.rand(*cold_zone_tmp[i].shape) - amplification
                r[r < 0] = 0
                cold_zone_tmp[i] += r
                cold_zone_tmp[i][cold_zone_tmp[i] > 1] = 1
            for p_base, p_main, p_target, p_mask in zip(self.base_model.parameters(), self.main_model.parameters(), self.target_model.parameters(), cold_zone_tmp):
                p_target.data = torch.lerp(p_main.data, p_base.data, p_mask)
            did_successful_cold_zone = self.cold_zone_loss()
            if did_successful_cold_zone:
                # Copy a successful new cold zone
                successful_tries += 1
                print(f"Successful #{successful_tries} cold-zone calculated (amp: {amplification:.2f} restart_tries: {restart_tries})")
                for i in range(len(cold_zone_tmp)):
                    self.cold_zones[i][:] = cold_zone_tmp[i]
                restart_tries = max(restart_tries - 1, 0)
                amplification = max(amplification - 0.01, 0.1)
            else:
                # Restore cold_zone_tmp to last "known to work" position
                print(f"(restart_tries: {restart_tries} amp: {amplification:.2f})")
                for i in range(len(cold_zone_tmp)):
                    cold_zone_tmp[i][:] = self.cold_zones[i]
                restart_tries += 1
                amplification = min(amplification + 0.01, 0.99)
            
            if restart_tries == 25:
                return None
            
    def mix(self, amount, cold_zone_diffusion = None, lower_bound = 0.2, normalize_cold_zone = False):
        for i, (p_base, p_main, p_target) in enumerate(zip(self.base_model.parameters(), self.main_model.parameters(), self.target_model.parameters())):
            if cold_zone_diffusion is None or self.cold_zones is None:
                mask = amount
            else:
                cold_zone = torch.clone(self.cold_zones[i])
                if normalize_cold_zone:            
                    cold_zone -= torch.min(cold_zone)
                    cold_zone /= torch.max(cold_zone)
                mask = self.diffuse_cold_zone(cold_zone, cold_zone_diffusion, lower_bound) * amount
                del cold_zone
            p_target.data = torch.lerp(p_main.data, p_base.data, mask)