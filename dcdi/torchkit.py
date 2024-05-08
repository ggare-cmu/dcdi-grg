"""
Copyright Chin-Wei Huang
"""
import numpy as np
import torch

from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F

delta = 1e-6
c = - 0.5 * np.log(2 * np.pi)


def log(x):
    return torch.log(x * 1e2) - np.log(1e2)

'''
Note-GRG: log_normal function is the log likelihood of the normal distribution
            normal distribution is given by:
            p(x;μ,σ)=1σ2π​exp(−(x−μ)2/(2σ2))

            log_normal(x, mean, log_var, eps=0.00001) = - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c
            where c = - 0.5 * np.log(2 * np.pi)
            The log likelihood of the normal distribution is given by:
            log p(x;μ,σ) = − (x−μ)2/(2σ2)− log(σ)− 1/2 log(2π)
            p(x;μ,σ)=1σ2π​exp(−2σ2(log(x)−μ)2)
            log p(x;μ,σ)=−(x−μ)2/(2σ2)−log(σ)−1/2log(2π)

            Normal distribution is a continuous probability distribution for a real-valued random variable.
            The normal distribution is a probability function that describes how the values of a variable are distributed.
            It is the most important probability distribution in statistics because of its simplicity in application and because it describes many natural phenomena.
            The normal distribution is completely determined by two parameters: the mean and the standard deviation.
            The mean determines the center of the graph of the normal distribution, and the standard deviation determines the height and width of the graph.
            The normal distribution is defined by the following probability density function, where μ is the mean and σ is the standard deviation.
            pdf of normal distribution = f(x;μ,σ)=1/{σ (2π)^0.5} * ​exp(−0.5*(x−μ)^2/σ^2))


            Here:
            c = log(1/√(2π)) = - 0.5 * np.log(2 * np.pi)
            - log_var/2 = -log(σ^2)/2 = -log(σ)
            torch.exp(log_var) = σ^2
'''
def log_normal(x, mean, log_var, eps=0.00001):
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + c


def logsigmoid(x):
    return -softplus(-x)


def log_sum_exp(A, axis=-1, sum_op=torch.sum):
    def maximum(x):
        return x.max(axis)[0]

    A_max = oper(A, maximum, axis, True)

    def summation(x):
        return sum_op(torch.exp(x - A_max), axis)

    B = torch.log(oper(A, summation, axis, True)) + A_max
    return B


def oper(array, oper, axis=-1, keepdims=False):
    a_oper = oper(array)
    if keepdims:
        shape = []
        for s in array.size():
            shape.append(s)
        shape[axis] = -1
        a_oper = a_oper.view(*shape)
    return a_oper


def softplus(x):
    return F.softplus(x) + delta


def softmax(x, dim=-1):
    e_x = torch.exp(x - x.max(dim=dim, keepdim=True)[0])
    out = e_x / e_x.sum(dim=dim, keepdim=True)
    return out


class BaseFlow(torch.nn.Module):

    def sample(self, n=1, context=None, **kwargs):
        dim = self.dim
        if isinstance(self.dim, int):
            dim = [dim, ]

        spl = Variable(torch.FloatTensor(n, *dim).normal_())
        lgd = Variable(torch.from_numpy(
            np.zeros(n).astype('float32')))
        if context is None:
            context = Variable(torch.from_numpy(
                np.ones((n, self.context_dim)).astype('float32')))

        if hasattr(self, 'gpu'):
            if self.gpu:
                spl = spl.cuda()
                lgd = lgd.cuda()
                context = context.gpu()

        return self.forward((spl, lgd, context))

    def cuda(self):
        self.gpu = True
        return super(BaseFlow, self).cuda()


#Note-GRG: Original source code - https://github.com/CW-Huang/torchkit/blob/master/torchkit/flows.py
class SigmoidFlow(BaseFlow):
    """
    Layer used to build Deep sigmoidal flows

    Parameters:
    -----------
    num_ds_dim: uint
        The number of hidden units

    """

    def __init__(self, num_ds_dim=4):
        super(SigmoidFlow, self).__init__()
        self.num_ds_dim = num_ds_dim

    #Note-GRG: act_a implies the activation function for 'a'
    def act_a(self, x):
        return softplus(x)

    #Note-GRG: act_b implies the activation function for 'b'
    def act_b(self, x):
        return x

    #Note-GRG: act_w implies the activation function for 'w'
    def act_w(self, x):
        return softmax(x, dim=2)

    '''
    #Note-GRG: forward function is the forward pass of the flow model
        x is the input, logdet is the log determinant of the Jacobian, dsparams are the parameters of the flow model 
        and mollify is the mollification parameter, delta is the delta parameter
        output y_t = σ^−1( wT · σ( a · x_t + b )), where σ is the sigmoid function and w, a, b are the parameters of the flow model 
                            and σ^-1 is the inverse of sigmoid function i.e. logit function
    '''
    def forward(self, x, logdet, dsparams, mollify=0.0, delta=delta):
        ndim = self.num_ds_dim
        # Apply activation functions to the parameters produced by the hypernetwork
        a_ = self.act_a(dsparams[:, :, 0: 1 * ndim])
        b_ = self.act_b(dsparams[:, :, 1 * ndim: 2 * ndim])
        w = self.act_w(dsparams[:, :, 2 * ndim: 3 * ndim])

        a = a_ * (1 - mollify) + 1.0 * mollify
        b = b_ * (1 - mollify) + 0.0 * mollify

        pre_sigm = a * x[:, :, None] + b  # C #Note-GRG: C & D refers to the supplemental pg12 of Neural Autoregressive Flows paper https://arxiv.org/abs/1804.00779 
        sigm = torch.sigmoid(pre_sigm)
        x_pre = torch.sum(w * sigm, dim=2)  # D
        x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
        x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)  # Logit function (so H) #Note-GRG: H = logit(x) = log(x/(1-x)); logit func is the inverse of sigmoid func 
        xnew = x_ #Note-GRG: xnew = σ^−1( wT · σ( a · x + b ))

        logj = F.log_softmax(dsparams[:, :, 2 * ndim: 3 * ndim], dim=2) + \
               logsigmoid(pre_sigm) + \
               logsigmoid(-pre_sigm) + log(a)

        logj = log_sum_exp(logj, 2).sum(2)

        logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
        logdet += logdet_

        return xnew, logdet
