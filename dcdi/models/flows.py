import torch
from torch.autograd import Variable
from ..torchkit import log_normal, SigmoidFlow
from .base_model import BaseModel


class FlowModel(BaseModel):
    """
    Abstract class for normalizing flow model
    """
    def __init__(self, num_vars, num_layers, hid_dim, num_params, nonlin="leaky-relu",
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        super().__init__(num_vars, num_layers, hid_dim, num_params, nonlin=nonlin,
                         intervention=intervention,
                         intervention_type=intervention_type,
                         intervention_knowledge=intervention_knowledge,
                         num_regimes=num_regimes)
        self.reset_params()

    def compute_log_likelihood(self, x, weights, biases, extra_params,
                               detach=False, mask=None, regime=None):
        """
        Return log-likelihood of the model for each example.
        WARNING: This is really a joint distribution only if the DAGness constraint on the mask is satisfied.
                 Otherwise the joint does not integrate to one.
        :param x: (batch_size, num_vars)
        :param weights: list of tensor that are coherent with self.weights
        :param biases: list of tensor that are coherent with self.biases
        :param mask: tensor, shape=(batch_size, num_vars)
        :param regime: np.ndarray, shape=(batch_size,)
        :return: (batch_size, num_vars) log-likelihoods
        """
        #GRG
        density_params = self.forward_given_params(x, weights, biases, mask, regime)
        return self._log_likelihood(x, density_params)

    def reset_params(self):
        super().reset_params()
        if "flow" in self.__dict__ and hasattr(self.flow, "reset_parameters"):
            self.flow.reset_parameters()


class DeepSigmoidalFlowModel(FlowModel):
    def __init__(self, num_vars, cond_n_layers, cond_hid_dim, cond_nonlin, flow_n_layers, flow_hid_dim,
                 intervention=False, intervention_type="perfect",
                 intervention_knowledge="known", num_regimes=1):
        """
        Deep Sigmoidal Flow model

        :param int num_vars: number of variables
        :param int cond_n_layers: number of layers in the conditioner
        :param int cond_hid_dim: number of hidden units in the layers of the conditioner
        :param str cond_nonlin: type of non-linearity used in the conditioner
        :param int flow_n_layers: number of DSF layers
        :param int flow_hid_dim: number of hidden units in the DSF layers
        :param boolean intervention: True if use interventional version (DCDI)
        :param str intervention_type: Either perfect or imperfect
        :param str intervention_knowledge: Either known or unkown
        :param int num_regimes: total number of regimes in the data
        """
        flow_n_conditioned = flow_hid_dim

        # Conditioner model initialization
        n_conditioned_params = flow_n_conditioned * 3 * flow_n_layers  # Number of conditional params for each variable
        super().__init__(num_vars, cond_n_layers, cond_hid_dim, num_params=n_conditioned_params, nonlin=cond_nonlin,
                         intervention=intervention,
                         intervention_type=intervention_type,
                         intervention_knowledge=intervention_knowledge,
                         num_regimes=num_regimes)
        self.cond_n_layers = cond_n_layers
        self.cond_hid_dim = cond_hid_dim
        self.cond_nonlin = cond_nonlin

        # Flow model initialization
        self.flow_n_layers = flow_n_layers
        self.flow_hid_dim = flow_hid_dim
        self.flow_n_params_per_var = flow_hid_dim * 3 * flow_n_layers  # total number of params #Note-GRG: same as n_conditioned_params
        self.flow_n_cond_params_per_var = n_conditioned_params  # number of conditional params
        self.flow_n_params_per_layer = flow_hid_dim * 3  # number of params in each flow layer
        self.flow = SigmoidFlow(flow_hid_dim)

        # Shared density parameters (i.e, those that are not produced by the conditioner)
        #Note-GRG: always self.flow_n_params_per_var == self.flow_n_cond_params_per_var == self.flow_hid_dim * 3 * self.flow_n_layers
        self.shared_density_params = torch.nn.Parameter(torch.zeros(self.flow_n_params_per_var -
                                                                    self.flow_n_cond_params_per_var))

    def reset_params(self):
        super().reset_params()
        if "flow" in self.__dict__:
            self.flow.reset_parameters()
        if "shared_density_params" in self.__dict__:
            self.shared_density_params.data.uniform_(-0.001, 0.001)

    def _log_likelihood(self, x, density_params):
        """
        Compute the log likelihood of x given some density specification.

        :param x: torch.Tensor, shape=(batch_size, num_vars), the input for which to compute the likelihood.
        :param density_params: tuple of torch.Tensor, len=n_vars, shape of elements=(batch_size, n_flow_params_per_var)
            The parameters of the DSF model that were produced by the conditioner.
        :return: pseudo joint log-likelihood
        """
        # Convert the shape to (batch_size, n_vars, n_flow_params_per_var)
        density_params = torch.cat([x[None, :, :] for x in density_params], dim=0).transpose(0, 1)
        assert len(density_params.shape) == 3
        assert density_params.shape[0] == x.shape[0]
        assert density_params.shape[1] == self.num_vars
        assert density_params.shape[2] == self.flow_n_cond_params_per_var

        #Note-GRG: The entire inject code using condtional and shared is nonsense as shared_density_params is always empty!
        # Inject shared parameters here
        # Add the shared density parameters in each layer's parameter vectors
        # The shared parameters are different for each layer
        # All batch elements receive the same shared parameters
        conditional = density_params.view(density_params.shape[0], density_params.shape[1], self.flow_n_layers, 3, -1)
        
        #Note-GRG: shared_density_params is always empty (None)
        shared = \
            self.shared_density_params.view(self.flow_n_layers, 3, -1)[None, None, :, :, :].repeat(conditional.shape[0],
                                                                                                   conditional.shape[1],
                                                                                                   1, 1, 1)
        density_params = torch.cat((conditional, shared), -1).view(conditional.shape[0], conditional.shape[1], -1)
        assert density_params.shape[2] == self.flow_n_params_per_var

        #Note-GRG deva's: logdet could represent L2 in the latent space
        logdet = Variable(torch.zeros((x.shape[0], self.num_vars)))
        h = x.view(x.size(0), -1)
        for i in range(self.flow_n_layers):
            # Extract params of the current flow layer. Shape is (batch_size, n_vars, self.flow_n_params_per_layer)
            params = density_params[:, :, i * self.flow_n_params_per_layer: (i + 1) * self.flow_n_params_per_layer]
            h, logdet = self.flow(h, logdet, params)

        assert x.shape[0] == h.shape[0]
        assert x.shape[1] == h.shape[1]
        zeros = Variable(torch.zeros(x.shape[0], self.num_vars))
        '''
        Note-GRG:
            In the expression: pseudo_joint_nll = - log_normal(h, zeros, zeros + 1.0) - logdet
            'h' is the latent space representation of the data i.e. the encoded data using the flow model
            so log_normal(h, zeros, zeros + 1.0) is the log-likelihood of the data in the simple gaussian latent space, with mean = 0 and var = 1 (or log_var = 1? [potential bug])
                - so this gives us the (log-)likelihood of the data in the latent space
            logdet is the log determinant of the Jacobian of the transformation from the input space to the latent space 
                - this is the multiplier/scaling factor for the likelihood from the latent space to the input space
                Thus multiplying the likelihood in the latent space with the scaling factor gives us the likelihood in the input space (pseudo_joint_nll)
                as - log(normal_likelihood) - log(det) = -log(det*normal_likelihood) = -log(input_space_likelihood) [i.e. NLL in the input space]
            pseudo_joint_nll is the negative log-likelihood of the data in the latent space
        '''
        # Not the joint NLL until we have a DAG #Note-GRG: need to understand this - why not hoint NLL until we have a DAG?
        pseudo_joint_nll = - log_normal(h, zeros, zeros + 1.0) - logdet #Note-GRG: here mean is zero and log-variance is one #Bug-Fix GRG: should the variance be 1.0 or log-var be 1.0?

        # We return the log product (averaged) of conditionals instead of the logp for each conditional.
        #      Shape is (batch x 1) instead of (batch x n_vars).
        return - pseudo_joint_nll
