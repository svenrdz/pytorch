"""
Weight Normalization from https://arxiv.org/abs/1602.07868
"""
import torch.utils.hooks as hooks
from torch.nn.parameter import Parameter


class WeightNorm(object):
    def __init__(self, name, dim):
        self.name = name
        self.dim = dim

    def compute_weight(self, module):
        g = getattr(module, self.name + '_g')
        v = getattr(module, self.name + '_v')
        return v * (g / self.norm(v))

    def norm(self, p):
        """Computes the norm over all dimensions except dim"""
        if self.dim is None:
            return p.norm()
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        output_size = (p.size(0),) + (1,) * (p.dim() - 1)
        p = p.contiguous().view(p.size(0), -1).norm(dim=1).view(*output_size)
        if self.dim != 0:
            p = p.transpose(0, self.dim)
        return p

    @staticmethod
    def apply(module, name, dim):
        fn = WeightNorm(name, dim)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(fn.norm(weight).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        handle = hooks.RemovableHandle(module._forward_pre_hooks)
        module._forward_pre_hooks[handle.id] = fn
        fn.handle = handle

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)

        self.handle.remove()
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def weight_norm(module, name='weight', dim=0):
    """Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by `name` (e.g. "weight") with two parameters: one specifying the magnitude
    (e.g. "weight_g") and one specifying the direction (e.g. "weight_v").
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with `dim=0`, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    `dim=None`.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        Linear (20 -> 40)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module, name='weight'):
    """Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for hook in module._forward_pre_hooks.values():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
