import math
import warnings
import torch

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Fill the input tensor with values drawn from a truncated normal distribution.

    Values are sampled from a normal distribution with the given mean and
    standard deviation, but are restricted to the interval [a, b].
    The operation is performed without tracking gradients.

    Args:
        tensor (torch.Tensor): Tensor to initialize.
        mean (float): Mean of the normal distribution.
        std (float): Standard deviation of the normal distribution.
        a (float): Lower bound of the truncation interval.
        b (float): Upper bound of the truncation interval.

    Returns:
        torch.Tensor: The initialized tensor.
    """
    def norm_cdf(x):
        # the probability that a standard-normal random variable is smaller than or equal to x
        return (1. + math.erf(x / math.sqrt(2.))) / 2
    
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect.",
                stacklevel=2)
    
    with torch.no_grad():
        """
        Instead of directly sampling a truncated normal distribution, it samples uniformly in the normal CDF space, then transforms those values back into normal-distributed values using the inverse error function.
        """
        l = norm_cdf((a-mean)/std)  # The probability mass of the normal distribution below a
        u = norm_cdf((b-mean)/std)  # The probability mass of the normal distribution above b
        
        tensor.uniform_(2*l-1, 2*u-1)  # uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1]
        
        tensor.erfinv_()
        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        return tensor
        

def trunc_normal_(tensor, mean=0., std=1., a=-2, b=-2):
    
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)