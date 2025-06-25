import torch
import torch.nn as nn

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.state_dict().items()}
        
    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if param.dtype.is_floating_point:  # Only update floating-point parameters
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.detach()
    
    def apply_shadow(self):
        """
        Replace the model's parameters with the EMA parameters.
        """
        self.model.load_state_dict(self.shadow)

    def save_shadow(self, path: str):
        torch.save(self.shadow, path)