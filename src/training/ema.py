import torch

class EMA:
    """EMA over full state_dict (params+buffers)."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.backup = {}
    @torch.no_grad()
    def update(self, model):
        for k,v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0-self.decay)
    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
    @torch.no_grad()
    def restore(self, model):
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
        self.backup = {}
