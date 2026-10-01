from __future__ import annotations


def integrated_gradients(model, inputs, *, baseline=None, steps: int = 64):
    """Integrated Gradients for [batch, lead, sample] tensors."""
    import torch

    if steps < 2:
        raise ValueError("steps must be at least 2")
    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    if baseline.shape != inputs.shape:
        raise ValueError("baseline and inputs must have the same shape")
    alphas = torch.linspace(0, 1, steps, device=inputs.device)
    gradients = []
    was_training = model.training
    model.eval()
    for alpha in alphas:
        point = (baseline + alpha * (inputs - baseline)).detach().requires_grad_(True)
        output = model(point).sum()
        gradient = torch.autograd.grad(output, point)[0]
        gradients.append(gradient.detach())
    if was_training:
        model.train()
    average_gradient = torch.stack(gradients).mean(dim=0)
    return (inputs - baseline) * average_gradient
