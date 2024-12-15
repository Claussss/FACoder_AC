def get_total_params(model):
  total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total_params
