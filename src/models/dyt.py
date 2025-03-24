import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicTanh(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.elementwise_affine:
            return self.weight * torch.tanh(self.alpha * x) + self.bias
        else:
            return torch.tanh(self.alpha * x)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"alpha_init_value={self.alpha_init_value}"
        )


class DynamicAsymmetricTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, smoothing=10.0, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.smoothing = smoothing
        self.elementwise_affine = elementwise_affine

        self.alpha_pos = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)
        self.alpha_neg = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        sig = torch.sigmoid(self.smoothing * x)
        alpha = self.alpha_neg + (self.alpha_pos - self.alpha_neg) * sig  # shape broadcasting supported

        if self.elementwise_affine:
            return self.weight * torch.tanh(alpha * x) + self.bias
        else:
            return torch.tanh(alpha * x)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"elementwise_affine={self.elementwise_affine}, "
        )


class DynamicSoftsign(nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, alpha_init_value=0.5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.alpha_init_value = alpha_init_value
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if self.elementwise_affine:
            return self.weight * F.softsign(self.alpha * x) + self.bias
        else:
            return F.softsign(self.alpha * x)

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"alpha_init_value={self.alpha_init_value}"
        )


class DynamicAsymmetricSoftsign(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, smoothing=10.0, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.smoothing = smoothing
        self.elementwise_affine = elementwise_affine

        self.alpha_pos = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)
        self.alpha_neg = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        # smooth interpolation between alpha_neg and alpha_pos
        sig = torch.sigmoid(self.smoothing * x)
        alpha = self.alpha_neg + (self.alpha_pos - self.alpha_neg) * sig  # shape broadcasting supported

        out = x / (1 + alpha * x.abs())

        if self.elementwise_affine:
            out = out * self.weight + self.bias

        return out
    

class HardSigmoidWithSmoothGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha_pos, alpha_neg, smoothing):
        ctx.save_for_backward(x, alpha_pos, alpha_neg)
        ctx.smoothing = smoothing

        # Fast inference: hard step (non-smooth) but fast
        mask = (x >= 0).float()
        alpha = alpha_neg * (1 - mask) + alpha_pos * mask
        return alpha * x

    @staticmethod
    def backward(ctx, grad_output):
        x, alpha_pos, alpha_neg = ctx.saved_tensors
        smoothing = ctx.smoothing

        # Smooth interpolation for gradient
        sig = torch.sigmoid(smoothing * x)
        alpha = alpha_neg + (alpha_pos - alpha_neg) * sig
        grad_input = grad_output * alpha  # Approximate derivative
        return grad_input, None, None, None


class FastDynamicAsymmetricTanh(nn.Module):
    def __init__(self, normalized_shape, alpha_init_value=0.5, smoothing=10.0, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.smoothing = smoothing
        self.elementwise_affine = elementwise_affine

        self.alpha_pos = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)
        self.alpha_neg = nn.Parameter(torch.ones(normalized_shape) * alpha_init_value)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        alpha_x = HardSigmoidWithSmoothGrad.apply(x, self.alpha_pos, self.alpha_neg, self.smoothing)
        out = torch.tanh(alpha_x)
        if self.elementwise_affine:
            out = out * self.weight + self.bias
        return out

    def extra_repr(self):
        return (
            f"normalized_shape={self.normalized_shape}, "
            f"elementwise_affine={self.elementwise_affine}, "
            f"smoothing={self.smoothing}"
        )


def convert_ln_to_dyt(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicTanh(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyt(child))
    del module
    return module_output


def convert_ln_to_dyat(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicAsymmetricTanh(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyat(child))
    del module
    return module_output


def convert_ln_to_fdyat(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = FastDynamicAsymmetricTanh(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_fdyat(child))
    del module
    return module_output


def convert_ln_to_dys(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicSoftsign(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dys(child))
    del module
    return module_output


def convert_ln_to_dyas(module):
    module_output = module
    if isinstance(module, nn.LayerNorm):
        module_output = DynamicAsymmetricSoftsign(module.normalized_shape, module.elementwise_affine)
    for name, child in module.named_children():
        module_output.add_module(name, convert_ln_to_dyas(child))
    del module
    return module_output
