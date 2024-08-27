from absl import flags
from typing import Optional, Sequence, Union
import enum
from acme.tf import networks
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import assert_util

from acme.tf.networks import distributions as ad
from acme.tf.networks import DiscreteValuedHead, CriticMultiplexer, LayerNormMLP
from tensorflow_probability.python.bijectors import generalized_pareto as generalized_pareto_bijector
from tensorflow_probability.python.bijectors import softplus as softplus_bijector
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import dtype_util
import numpy as np
from scipy.stats import norm

tfd = tfp.distributions


uniform_initializer = tf.initializers.VarianceScaling(
    distribution='uniform', mode='fan_out', scale=0.333)


FLAGS = flags.FLAGS


class RiskDiscreteValuedDistribution(ad.DiscreteValuedDistribution):
    def __init__(self,
                 values: tf.Tensor,
                 logits: Optional[tf.Tensor] = None,
                 probs: Optional[tf.Tensor] = None,
                 name: str = 'RiskDiscreteValuedDistribution'):
        super().__init__(values, logits, probs, name)

    def _normal_dist_volc(self, quantile):
        prob_density = round(norm.ppf(quantile), 4)
        return prob_density

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        """Implements mean-volc*std for VaR estimation"""
        volc = self._normal_dist_volc(th)
        return self.mean() - volc*self.stddev()

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_logits = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        clogits = tf.where(exclude_logits, zero, self.probs_parameter())
        return tf.reduce_sum(clogits * self.values, axis=-1)

    def gain_loss_tradeoff(self) -> tf.Tensor:
        """Implements gain_loss tradeoff objective function"""
        zero = tf.constant(0, dtype=tf.float32)
        return tf.reduce_mean(FLAGS.k1 * tf.pow(tf.maximum(zero, self._values), FLAGS.alpha) - FLAGS.k2 * tf.pow(tf.maximum(zero, -self._values), FLAGS.alpha), axis=-1)


class RiskDiscreteValuedHead(DiscreteValuedHead):
    def __init__(self,
                 vmin: Union[float, np.ndarray, tf.Tensor],
                 vmax: Union[float, np.ndarray, tf.Tensor],
                 num_atoms: int,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(vmin, vmax, num_atoms, w_init, b_init)

    def __call__(self, inputs: tf.Tensor) -> RiskDiscreteValuedDistribution:
        logits = self._distributional_layer(inputs)
        logits = tf.reshape(logits,
                            tf.concat([tf.shape(logits)[:1],  # batch size
                                       tf.shape(self._values)],
                                      axis=0))
        values = tf.cast(self._values, logits.dtype)

        return RiskDiscreteValuedDistribution(values=values, logits=logits)


def quantile_project(  # pylint: disable=invalid-name
    q: tf.Tensor,
    v: tf.Tensor,
    q_grid: tf.Tensor,
) -> tf.Tensor:
    """Project quantile distribution (quantile_grid, values) onto quantile under the L2-metric over CDFs.

    This projection works for any support q.
    Let Kq be len(q_grid)

    Args:
    q: () quantile
    v: (batch_size, Kq) values to project onto
    q_grid:  (Kq,) Quantiles for P(Zp[i])

    Returns:
    Quantile projection of (q_grid, v) onto q.
    """

    # Asserts that Zq has no leading dimension of size 1.
    if q_grid.get_shape().ndims > 1:
        q_grid = tf.squeeze(q_grid, axis=0)
    q = q[None]
    # Extracts vmin and vmax and construct helper tensors from Zq.
    vmin, vmax = q_grid[0], q_grid[-1]
    d_pos = tf.concat([q_grid, vmin[None]], 0)[1:]
    d_neg = tf.concat([vmax[None], q_grid], 0)[:-1]

    # Clips Zp to be in new support range (vmin, vmax).
    clipped_q = tf.clip_by_value(q, vmin, vmax)  # (1,)
    eq_mask = tf.cast(tf.equal(q_grid, q), q_grid.dtype)
    if tf.equal(tf.reduce_sum(eq_mask), 1.0):
        # (batch_size, )
        return tf.squeeze(tf.boolean_mask(v, eq_mask, axis=1), axis=-1)

    # need interpolation
    pos_neg_mask = tf.cast(tf.roll(q_grid <= q, 1, axis=0), q_grid.dtype) \
        * tf.cast(tf.roll(q_grid >= q, -1, axis=0), q_grid.dtype)
    pos_neg_v = tf.boolean_mask(v, pos_neg_mask, axis=1)    # (batch_size, 2)

    # Gets the distance between atom values in support.
    d_pos = (d_pos - q_grid)[None, :]  # (1, Kq)
    d_neg = (q_grid - d_neg)[None, :]  # (1, Kq)

    clipped_q_grid = q_grid[None, :]  # (1, Kq)
    delta_qp = clipped_q - clipped_q_grid  # (1, Kq)

    d_sign = tf.cast(delta_qp >= 0., dtype=v.dtype)
    delta_hat = (d_sign * delta_qp / d_pos) - \
        ((1. - d_sign) * delta_qp / d_neg)  # (1, Kq)
    # (batch_size, )
    return tf.reduce_sum(tf.clip_by_value(1. - delta_hat, 0., 1.) * v, 1)


@tfp.experimental.register_composite
class QuantileDistribution(tfd.Categorical):
    def __init__(self,
                 values: tf.Tensor,
                 quantiles: tf.Tensor,
                 probs: tf.Tensor,
                 name: str = 'QuantileDistribution'):
        """Quantile Distribution
        values: (batch_size, Kq)
        quantiles: (Kq,) or (batch_size, Kq)
        probs: (Kq,)
        """
        self._quantiles = tf.convert_to_tensor(quantiles)
        self._shape_strings = [f'D{i}' for i, _ in enumerate(quantiles.shape)]
        self._values = tf.convert_to_tensor(values)
        self._probs = tf.convert_to_tensor(probs)

        super().__init__(probs=probs, name=name)
        self._parameters = dict(values=values,
                                quantiles=quantiles,
                                probs=probs,
                                name=name)

    @property
    def quantiles(self) -> tf.Tensor:
        return self._quantiles

    @property
    def values(self) -> tf.Tensor:
        return self._values
    
    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
        return dict(
            values=tfp.util.ParameterProperties(
                event_ndims=lambda self: self.quantiles.shape.rank),
            quantiles=tfp.util.ParameterProperties(
                event_ndims=None),
            probs=tfp.util.ParameterProperties(
                event_ndims=None))

    def _sample_n(self, n, seed=None) -> tf.Tensor:
        indices = super()._sample_n(n, seed=seed)
        return tf.gather(self.values, indices, axis=-1)

    def _mean(self) -> tf.Tensor:
        # assume values are always with equal prob
        return tf.reduce_mean(self.values, axis=-1)

    def _variance(self) -> tf.Tensor:
        dist_squared = tf.square(tf.expand_dims(self.mean(), -1) - self.values)
        return tf.reduce_sum(self.probs_parameter() * dist_squared, axis=-1)

    def _event_shape(self):
        # Omit the atoms axis, to return just the shape of a single (i.e. unbatched)
        # sample value.
        return self._quantiles.shape[:-1]

    def _event_shape_tensor(self):
        return tf.shape(self._quantiles)[:-1]

    def meanstd(self) -> tf.Tensor:
        """Implements mean-volc*std"""
        volc = FLAGS.std_coef
        return self.mean() - volc*self.stddev()

    def var(self, th) -> tf.Tensor:
        quantile = tf.convert_to_tensor(1 - th)
        return quantile_project(quantile, self._values, self.quantiles)

    def cvar(self, th) -> tf.Tensor:
        quantile = 1 - th
        cdf = tf.cumsum(self.probs_parameter(), axis=-1)
        exclude_probs = cdf > quantile
        zero = np.array(0, dtype=dtype_util.as_numpy_dtype(cdf.dtype))
        cprobs = tf.where(exclude_probs, zero, self.probs_parameter())
        return tf.reduce_sum(cprobs * self.values, axis=-1)

    def gain_loss_tradeoff(self) -> tf.Tensor:
        """Implements gain_loss tradeoff objective function"""
        zero = tf.constant(0, dtype=tf.float32)
        return tf.reduce_mean(FLAGS.k1 * tf.pow(tf.maximum(zero, self._values), FLAGS.alpha) - FLAGS.k2 * tf.pow(tf.maximum(zero, -self._values), FLAGS.alpha), axis=-1)


class QuantileDistProbType(enum.Enum):
    LEFT = 1
    MID = 2
    RIGHT = 3


class QuantileDiscreteValuedHead(snt.Module):
    def __init__(self,
                 quantiles: np.ndarray,
                 prob_type: QuantileDistProbType = QuantileDistProbType.MID,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
                 
        super().__init__(name='QuantileDiscreteValuedHead')
        self._quantiles = tf.convert_to_tensor(quantiles)
        assert quantiles[0] > 0
        assert quantiles[-1] < 1.0
        left_probs = quantiles - np.insert(quantiles[:-1], 0, 0.0)
        right_probs = np.insert(
            quantiles[1:], len(quantiles)-1, 1.0) - quantiles
        if prob_type == QuantileDistProbType.LEFT:
            probs = left_probs
        elif prob_type == QuantileDistProbType.MID:
            probs = (left_probs + right_probs) / 2
        elif prob_type == QuantileDistProbType.RIGHT:
            probs = right_probs
        self._probs = tf.convert_to_tensor(probs)
        self._distributional_layer = snt.Linear(tf.size(self._quantiles),
                                                w_init=w_init,
                                                b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        quantile_values = self._distributional_layer(inputs)
        quantile_values = tf.reshape(quantile_values,
                                     tf.concat([tf.shape(quantile_values)[:1],
                                                tf.shape(self._quantiles)],
                                               axis=0))
        quantiles = tf.cast(self._quantiles, quantile_values.dtype)
        probs = tf.cast(self._probs, quantile_values.dtype)
        return QuantileDistribution(values=quantile_values, quantiles=quantiles,
                                    probs=probs)


def huber(x: tf.Tensor, k=1.0):
    return tf.where(tf.abs(x) < k, 0.5 * tf.pow(x, 2), k * (tf.abs(x) - 0.5 * k))


def quantile_regression(q_tm1: QuantileDistribution, r_t: tf.Tensor,
                        d_t: tf.Tensor,
                        q_t: QuantileDistribution):
    """Implements Quantile Regression Loss
    q_tm1: critic distribution of t-1
    r_t:   reward
    d_t:   discount
    q_t:   target critic distribution of t
    """

    z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * q_t.values
    z_tm1 = q_tm1.values
    diff = tf.expand_dims(tf.transpose(z_t), -1) - \
        z_tm1    # (n_tau_p, n_batch, n_tau)
    k = 1
    loss = huber(diff, k) * tf.abs(q_tm1.quantiles -
                                   tf.cast(diff < 0, diff.dtype)) / k

    return tf.reduce_mean(loss, (0, -1))



#####################  GPD

class GPDDistributionHead(snt.Module):
    """Module that outputs parameters for a Generalized Pareto Distribution."""

    def __init__(self, backbone_layer_sizes: Sequence[int] = (512, 512, 256),
                 init_scale=0.1, min_scale=1e-7, heavy_tail=True,
                 w_init: Optional[snt.initializers.Initializer] = None,
                 b_init: Optional[snt.initializers.Initializer] = None):
        super().__init__(name='GPDDistributionHead')

        self._init_scale = init_scale
        self._min_scale = min_scale

        # Shared backbone network
        self._backbone = snt.Sequential([
            networks.CriticMultiplexer(),
            networks.LayerNormMLP(backbone_layer_sizes, activate_final=True)
        ])

        # Shape head
        shape_network_layers = [
            snt.Linear(1, w_init=w_init, b_init=b_init)
        ]
        if heavy_tail:
            shape_network_layers.append(tf.nn.sigmoid)  # Ensures 0 < shape < 1 for heavy tail
        self._shape_head = snt.Sequential(shape_network_layers)

        # Scale head
        def scale_transformation(inputs):
            scale = tf.nn.softplus(inputs)
            # scale = scale * self._init_scale / tf.nn.softplus(tf.zeros_like(scale))
            # scale += self._min_scale
            return scale

        self._scale_head = snt.Sequential([
            snt.Linear(1, w_init=w_init, b_init=b_init),
            scale_transformation
        ])

    @property
    def backbone(self):
        return self._backbone

    @property
    def scale_network(self):
        # Dynamically combines the shared backbone with the scale head
        return snt.Sequential([self._backbone, self._scale_head])

    @property
    def shape_network(self):
        # Dynamically combines the shared backbone with the shape head
        return snt.Sequential([self._backbone, self._shape_head])


"""Generalized Pareto distribution."""
@tfp.experimental.register_composite
class GPDDistribution(tfd.GeneralizedPareto):
  
    def __init__(self,
               loc,
               scale,
               concentration,
               validate_args=False,
               allow_nan_stats=True,
               name: str = 'GPDDistribution'):
        """Construct a Generalized Pareto distribution.

        Args:
        scale: The scale of the distribution. GeneralizedPareto is a
            location-scale distribution, so doubling the `scale` doubles a sample
            and halves the density. Strictly positive floating point `Tensor`. Must
            broadcast with `loc`, `concentration`.
        concentration: The shape parameter of the distribution. The larger the
            magnitude, the more the distribution concentrates near `loc` (for
            `concentration >= 0`) or near `loc - (scale/concentration)` (for
            `concentration < 0`). Floating point `Tensor`.

        Raises:
        TypeError: if `loc`, `scale`, or `concentration` have different dtypes.
        """
        # parameters = dict(locals())
        self._scale = tf.convert_to_tensor(scale)
        self._concentration = tf.convert_to_tensor(concentration)
        self._loc= tf.convert_to_tensor(loc)

        super(GPDDistribution, self).__init__(
            loc=loc,
            scale=scale,
            concentration=concentration,
            validate_args=validate_args,
            allow_nan_stats=allow_nan_stats,
            name=name)
        
        # self._reparameterization_type = tfd.NOT_REPARAMETERIZED

    @classmethod
    def _parameter_properties(cls, dtype, num_classes=None):
    # pylint: disable=g-long-lambda
        return dict(
        loc=tfp.util.ParameterProperties(),
        scale=tfp.util.ParameterProperties(
            default_constraining_bijector_fn=(
                lambda: softplus_bijector.Softplus(low=dtype_util.eps(dtype)))),
        concentration=tfp.util.ParameterProperties())
    

    @property
    def scale(self):
        return self._scale

    @property
    def concentration(self):
        return self._concentration
    
    @property
    def loc(self):
        return self._loc 

    def _log_prob(self, x):
        scale = tf.convert_to_tensor(self.scale)
        concentration = tf.convert_to_tensor(self.concentration)
        z = self._z(x, scale)
        
        # Check for out-of-support samples
        out_of_support = tf.logical_or(x <= 0, tf.logical_and(concentration < 0, x * concentration >= -scale))
        
        eq_zero = tf.equal(concentration, 0)  # Concentration = 0 ==> Exponential.
        nonzero_conc = tf.where(eq_zero, tf.constant(1, dtype=self.dtype), concentration)
        y = 1 / nonzero_conc + tf.ones_like(z, dtype=self.dtype)
        where_nonzero = tf.where(
            tf.equal(y, 0), y, y * tf.math.log1p(nonzero_conc * z))
        
        log_prob = -tf.math.log(scale) - tf.where(eq_zero, z, where_nonzero)
        
        # Set log probability to -inf for out-of-support samples
        log_prob = tf.where(out_of_support, tf.constant(-float('inf'), dtype=self.dtype), log_prob)
        
        return log_prob
    
    def var(self, th, GPD_threshold):
        """Compute Value at Risk (VaR) for a given confidence level th."""
        alpha = 1 - th        
        return (self.scale / self.concentration) * (
            tf.pow((alpha / (1 - GPD_threshold)), -self.concentration) - 1
        )
    


def GPD_quantile_regression(q_tm1: QuantileDistribution, r_t: tf.Tensor,
                        d_t: tf.Tensor, combined_samples: tf.Tensor):
    """Implements Quantile Regression Loss
    q_tm1: critic distribution of t-1
    r_t:   reward
    d_t:   discount
    combined_samples: combined body and tail samples of the target distribution
    """

    # Use combined_samples as the target for the loss calculation
    z_t = tf.reshape(r_t, (-1, 1)) + tf.reshape(d_t, (-1, 1)) * combined_samples
    z_tm1 = q_tm1.values
    diff = tf.expand_dims(tf.transpose(z_t), -1) - z_tm1  # (n_tau_p, n_batch, n_tau)

    k = 1
    loss = huber(diff, k) * tf.abs(q_tm1.quantiles - tf.cast(diff < 0, diff.dtype)) / k

    return tf.reduce_mean(loss, (0, -1))



def compute_gpd_loss(excess_samples, loc, scale, shape):
    # Assuming loc, scale, and shape are batched tensors and you want to apply operations batch-wise
    batch_indices = tf.range(tf.shape(excess_samples)[0])


    def compute_batch_log_probs(index):
        batch_excess_samples = excess_samples[index, :]
        positive_excess_samples = tf.boolean_mask(batch_excess_samples, batch_excess_samples > 0)

        # Create a GPDDistribution instance for the current batch
        GPD_dis = GPDDistribution(loc=loc[index], scale=scale[index], concentration=shape[index])

        # Initialize a tensor filled with zeros of the same length as batch_excess_samples
        log_probs_batch = tf.zeros_like(batch_excess_samples, dtype=tf.float32)

        # Compute log probabilities for positive excess samples, if any
        log_probs = tf.cond(
            tf.size(positive_excess_samples) > 0,
            lambda: GPD_dis.log_prob(positive_excess_samples),
            lambda: tf.zeros_like(positive_excess_samples, dtype=tf.float32)  # Use zeros if no positive samples
        )
    
    
        indices = tf.where(batch_excess_samples > 0)

        log_probs_batch = tf.tensor_scatter_nd_update(log_probs_batch, indices, log_probs)

        return log_probs_batch


    # Apply the modified function to each index
    log_probs= tf.map_fn(
        compute_batch_log_probs,
        batch_indices,
        fn_output_signature=tf.float32
    )


    # Filter and compute loss as beforeclear
    log_probs_filtered = tf.boolean_mask(log_probs, tf.logical_and(tf.not_equal(log_probs, 0.0), tf.logical_not(tf.math.is_nan(log_probs))))
    has_valid_log_probs = tf.reduce_any(tf.not_equal(log_probs_filtered, 0.0))
    gpd_loss = -tf.reduce_mean(log_probs_filtered)
    return gpd_loss, has_valid_log_probs


## Upsampling to ensure that n_samples exist for CVaR
def GPD_samples(loc, scale, shape, threshold_values, n_samples):
    
    batch_indices = tf.range(tf.shape(loc)[0])
    
    def generate_until_exceedance(index):

        GPD_dis = GPDDistribution(loc=loc[index], scale=scale[index], concentration=shape[index])
        
        def condition(exceeded_samples, num_exceeded):
            # Continue sampling until desired number of samples exceed the threshold
            return num_exceeded < n_samples
        
        def body(exceeded_samples, num_exceeded):
            # Generate more samples and filter
            batch_samples = GPD_dis.sample(n_samples)
            batch_exceeded_samples = tf.boolean_mask(batch_samples, batch_samples > threshold_values[index])
            # Concatenate new exceeded samples with existing ones
            exceeded_samples = tf.concat([exceeded_samples, batch_exceeded_samples], axis=0)
            # Update the count of exceeded samples
            num_exceeded = tf.shape(exceeded_samples)[0]
            return exceeded_samples, num_exceeded
        
        # Initialize exceeded_samples tensor
        exceeded_samples = tf.constant([], dtype=tf.float32)
        num_exceeded = 0
        
        # Loop until desired number of samples exceed the threshold
        exceeded_samples, _ = tf.while_loop(
            condition,
            body,
            [exceeded_samples, num_exceeded],
            shape_invariants=[tf.TensorShape([None]), tf.TensorShape([])]
        )
        
        # Compute the mean of exceeded samples
        mean_exceeded = tf.reduce_mean(exceeded_samples)
        
        return mean_exceeded

    # Apply generate_until_exceedance to each batch
    means_exceedance = tf.map_fn(
        generate_until_exceedance,
        batch_indices,
        fn_output_signature=tf.float32
    )
    
    return means_exceedance



def quantile_numbers(GPD_threshold, direction, quantile_interval=0.005):

    quantiles=np.arange(quantile_interval, 1.0, quantile_interval)

    if direction==1:
        n= np.sum(quantiles>= 1-GPD_threshold)

    elif direction==0:
        n= np.sum(quantiles < 1-GPD_threshold)

    return np.int32(n)
