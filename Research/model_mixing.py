import torch
import numpy as np
from scipy.ndimage.filters import gaussian_filter

class ModelMixing:
    def __init__(self, tokenizer, base_model, main_model, target_model, seed = 80085):
        self.base_model = base_model
        self.main_model = main_model
        self.target_model = target_model
        self.random_state = np.random.RandomState(seed)
        self.tokenizer = tokenizer
        
    def diffuse_cold_zone(self, cold_zone, diffusion_steps):
        new_cold_zone = cold_zone
        for i in range(diffusion_steps):
            new_cold_zone = gaussian_filter(new_cold_zone, sigma=1)
            # Make sure the original mask keeps existing
            new_cold_zone = np.where(cold_zone > 0, cold_zone, new_cold_zone)
        return new_cold_zone
        
    def calculate_cold_zones(self):
        def generate_responses(model):
            def generate_response(prompt):
                generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
                return self.tokenizer.decode(model.generate(
                    generated,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )[0])
            return [
                generate_response(word) for word in ["the", "hi", "over"]
            ]
            
        restart_tries = 0
        original_responses = generate_responses(self.main_model)
        cold_zone_tmp = []
        self.cold_zones = []
        for p in self.main_model.parameters():
            self.cold_zones.append(torch.zeros(p.shape))
            cold_zone_tmp.append(torch.zeros(p.shape))
            
        while True:
            for i in range(len(cold_zone_tmp)):
                r = torch.rand(*cold_zone_tmp[i].shape) - 0.95
                r[r < 0] = 0
                cold_zone_tmp[i] += r
                cold_zone_tmp[i][cold_zone_tmp[i] > 1] = 1
            for p_base, p_main, p_target, p_mask in zip(self.base_model.parameters(), self.main_model.parameters(), self.target_model.parameters(), cold_zone_tmp):
                p_target.data = torch.lerp(p_main.data, p_base.data, p_mask)
            test_responses = generate_responses(self.target_model)
            did_successful_cold_zone = True
            for r1, r2 in zip(original_responses, test_responses):
                if r1 != r2:
                    print(f"r1: ({r1}) != r2({r2})")
                    restart_tries += 1
                    did_successful_cold_zone = False
                    if restart_tries > 5:
                        return None
            if did_successful_cold_zone:
                # Copy a successful new cold zone
                for i in range(len(cold_zone_tmp)):
                    self.cold_zones[i][:] = cold_zone_tmp[i]
                restart_tries = 0
            
    def mix(self, amount, cold_zone_diffusion_steps = 0):
        for i, (p_base, p_main, p_target) in enumerate(zip(self.base_model.parameters(), self.main_model.parameters(), self.target_model.parameters())):
            if self.cold_zones:
                mask = self.diffuse_cold_zone(self.cold_zones[i], cold_zone_diffusion_steps)
                # We flip the new_cold_zone here becuase when calculating it
                # we the main model against the base model, here it's the other way around...
                mask = torch.Tensor((1 - mask) * amount)
            else:
                mask = amount
            p_target.data = torch.lerp(p_base.data, p_main.data, mask)