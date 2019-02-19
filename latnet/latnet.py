__author__ = 'AD'

import tensorflow as tf

import numpy as np
from tensorflow.python.framework.errors import OpError


class Latnet:
    FLOAT = tf.float64

    @staticmethod
    def replace_nan_with_zero(w):
        """
        Replaces NANs with zeros.

        Args:
            w: Input tensor

        Returns:
            replaces NAN elements in input argument `w' with zeros.
        """

        return tf.where(tf.is_nan(w), tf.ones_like(w) * 0.0, w)

    @staticmethod
    def contains_nan(w):
        """
        Given argument `w', the method returns whether there are any NANs in `w'.
        """

        return tf.reduce_all(tf.is_nan(w))

    @staticmethod
    def get_opzimier(objective, trainables, learning_rate, max_global_norm=1.0):
        """
        Calculates the Tensorflow operation for optimizing `objective' function using AdamOptimizer.
        Note that NANs in the gradients will be replaced by zeros.

        Args:
            objective: double. Objective function to be optimized.
            trainables: List of variables which will be optimized.
            learning_rate: Learning rate of AdamOptimizer
            max_global_norm: Used for gradient clipping.
        """
        grads = tf.gradients(objective, trainables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm=max_global_norm)
        grad_var_pairs = zip([Latnet.replace_nan_with_zero(g) for g in grads], trainables)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        return optimizer.apply_gradients(grad_var_pairs), grads, Latnet.contains_nan(grads)

    @staticmethod
    def run_model(t, Y, init_sigma2_g, init_sigma2_n, init_lengthscle, init_variance,
                  lambda_prior, lambda_postetior,
                  var_lr, hyp_lr,
                  n_samples,
                  seed=None,
                  fixed_kernel=False
                  ):
        """
        This method calculates Tensorflow quantities that will be used for training LATNET.

        Args:
            t: array of size N x 1
            Y: array of size N x T
            init_sigma2_g: double. Initial variance of the connection noise.
            init_sigma2_n: double. Initial variance of the observation noise.
            init_lengthscle: double. Initial length scale of the RBF kernel.
            init_variance: double. Initial variance of the RBF kenel.
            lambda_prior: double. lambda of the Concrete distribution used for prior over `A_ij' matrix.
            lambda_postetior: double. lambda of the Concrete distribution used for posterior over `A_ij' matrix.
            var_lr: double. Learning rate for updating variational parameters.
            hyp_lr: Learning rate for updating hyper parameters.
            n_samples: int. Number of Monte-Carlo samples used.
            seed: int. Seed used for random number generators.
            fix_kernel: Boolean. Whether to optimize kenrel parameters during the optimization of hyper-parameters.

        Returns:
            var_opt: Tensorflow operation for optimizing variational parameters for one step.
            hyp_opt: Tensorflow operation for optimizing hyper-parameters for one step.
            elbo: Tensor with shape () containing ELBO, i.e., evidence lower-bound.
            kl_normal: Tensor with shape () containing KL-divergence between prior over W and posterior W. It is
                calculated analytically.
            kl_logstic: Tensor with shape () containing KL-divergence between prior over A and posterior A.
                It is calculated using sampling.
            ell: Tensor with shape () containing approximated expected log-likelihood term.
            log_sigma2_n: Tensor of shape () containing the logarithm of sigma2_n, i.e., variance of observation noise.
            log_sigma2_g: Tensor of shape () containing the logarithm of sigma2_g, i.e., variance of connection noise.
            mu: Tensor of shape N x N containing variational parameter mu which is mean of W.
            log_sigma2: Tensor of shape N x N containing logarithm of sigma_2 which is variance of W.
            log_alpha: Tensor of shape N x N containing the logarithm of parameter alpha for Concrete distribution over A_ij.
            log_lengthscale: Tensor of shape () containing the logarithm of length scale of kernel.
            log_variance: Tensor of shape () containing the logarithm of variance of kernel.
            var_nans: Tensor of shape () representing whether there is any NAN in the gradients of variational parameters.
            hyp_nans: Tensor of shape () representing whether there is any NAN in the gradients of hyper parameters.
        """

        # get variables that will be optimized.
        log_sigma2_n, log_sigma2_g, mu, log_sigma2, log_alpha, \
        log_lengthscale, log_variance = Latnet.get_vairables(t, Y,
                                                             init_sigma2_g, init_sigma2_n, init_lengthscle,
                                                             init_variance)
        # get KL terms of ELL terms.
        kl_normal, kl_logstic, ell = Latnet.get_elbo(t, Y, tf.exp(log_sigma2_n),
                                                     tf.exp(log_sigma2_g), mu, tf.exp(log_sigma2), tf.exp(log_alpha),
                                                     tf.exp(log_lengthscale), tf.exp(log_variance),
                                                     tf.constant(lambda_prior, dtype=Latnet.FLOAT),
                                                     tf.constant(lambda_postetior, dtype=Latnet.FLOAT), n_samples, seed=seed)

        # calculating ELBO
        elbo = tf.negative(kl_normal) + tf.negative(kl_logstic) + ell

        # get the operation for optimizing variational parameters
        var_opt, _, var_nans = Latnet.get_opzimier(tf.negative(elbo), [mu, log_sigma2, log_alpha], var_lr)

        if fixed_kernel:
            # get the operation for optimizing hyper-parameters except kernel parameters.
            hyp_opt, _, hyp_nans = Latnet.get_opzimier(tf.negative(elbo),
                                                       [log_sigma2_n, log_sigma2_g],
                                                       hyp_lr)
        else:
            # get the operation for optimizing hyper-parameters (kernel parameters and noise parameters)
            hyp_opt, _, hyp_nans = Latnet.get_opzimier(tf.negative(elbo),
                                                       [log_sigma2_n, log_sigma2_g, log_lengthscale, log_variance], hyp_lr)
        return var_opt, hyp_opt, \
               elbo, \
               kl_normal, \
               kl_logstic, \
               ell, \
               log_sigma2_n, log_sigma2_g, mu, log_sigma2, log_alpha, \
               log_lengthscale, log_variance, var_nans, hyp_nans

    @staticmethod
    def kernel(X, variance, lengthscale, T):
        """
        RBF - Radial basis kernel (aka Squared Exponential Kernel)

        Args:
            X: Tensor of shape T x 1 containing the location of data-points.
            variance: double. Tensor of shape (), determining the variance of kernel.
            lengthscale: double. Tensor of shape (), determining the lengthscale of the kernel.
            T: int. Number of data-points.


        Returns:
            : Tensor of shape T x T, in which element ij is: variance * exp(-(x_i _ x_j)^2/(2 * lengthscale ** 2))
            Note that a latent noise is added to the kernel for numerical stability.

        """
        dist = tf.reduce_sum(tf.square(X), 1)
        dist = tf.reshape(dist, [-1, 1])
        sq_dists = tf.add(tf.subtract(dist, tf.multiply(tf.cast(2., Latnet.FLOAT), tf.matmul(X, tf.transpose(X)))),
                          tf.transpose(dist))
        return tf.multiply(variance, tf.exp(
            tf.negative(tf.div(tf.abs(sq_dists), tf.multiply(tf.cast(2.0, Latnet.FLOAT), tf.square(lengthscale)))))) + \
               tf.constant(1e-5 * np.identity(T), dtype=Latnet.FLOAT)

    @staticmethod
    def logp_logistic(X, alpha, lambda_):
        """
        Logarithm of Concrete distribution with parameter `alpha' and hyper-parameter `lambda_' at points `X', i.e.,
            Concrete(X;alpha, lambda_)


        Args:
            X: Tensor of shape S x N x N. Locations at which the distribution will be calculated.
            alpha:  Tensor of shape N x N. To be written.
            lambda_: double. Tensor of shape ().


        Returns:
            : A tensor of shape S x N x N. Element ijk is:
                log lambda_  - lambda_ * X_ijk + log alpha_jk - 2 log (1 + exp (-lambda_ * X_ijk + log alpha_jk))

        """

        mu = tf.log(alpha)
        return tf.subtract(tf.add(tf.subtract(tf.log(lambda_), tf.multiply(lambda_, X)), mu),
                      tf.multiply(tf.constant(2.0, dtype=Latnet.FLOAT),
                                  tf.log(tf.add(tf.constant(1.0, dtype=Latnet.FLOAT),
                                                tf.exp(tf.add(tf.negative(tf.multiply(lambda_, X)), mu))))))

    @staticmethod
    def get_KL_logistic(X, posterior_alpha, prior_lambda_, posterior_lambda_, prior_alpha):
        """
        Calculates KL divergence between two Concrete distributions using samples from posterior Concrete distribution.

        KL(Concrete(alpha, posterior_lambda_) || Concrete(prior_alpha, prior_lambda))

        Args:
            X: Tensor of shape S x N x N. These are samples from posterior Concrete distribution.
            posterior_alpha: Tensor of shape N x N. alpha for posterior distributions.
            prior_lambda_: Tensor of shape (). prior_lambda_ of prior distribution.
            posterior_lambda_: Tensor of shape (). posterior_lambda_ for posterior distribution.
            prior_alpha: Tensor of shape N x N. alpha for prior distributions.

        Returns:
            : Tensor of shape () representing KL divergence between the two concrete distributions.

        """
        logdiff = Latnet.logp_logistic(X, posterior_alpha, posterior_lambda_) - Latnet.logp_logistic(X, prior_alpha,
                                                                                                     prior_lambda_)
        logdiff = tf.matrix_set_diag(logdiff, tf.zeros((tf.shape(logdiff)[0], tf.shape(logdiff)[1]),
                                                       dtype=Latnet.FLOAT))  # set diagonal part to zero
        return tf.reduce_sum(tf.reduce_mean(logdiff, [0]))

    @staticmethod
    def get_KL_normal(posterior_mu, posterior_sigma2, prior_mu, prior_sigma2):
        """
        Calculates KL divergence between two Mormal distributions, i.e.,
            KL(Normal(mu, sigma2), Normal(prior_mu, prior_sigma2))

        Args:
            posterior_mu: Tensor of shape N x N.
            posterior_sigma2: Tensor of shape N x N.
            prior_mu: Tensor of shape N x N.
            prior_sigma2: Tensor of shape N x N.

        Returns:
            Tensor of shape (), which contains the KL divergence.
        """

        _half = tf.constant(0.5, dtype=Latnet.FLOAT)
        kl = tf.add(tf.div(tf.add(tf.square(tf.subtract(posterior_mu, prior_mu)), posterior_sigma2),
                           tf.multiply(tf.constant(2., dtype=Latnet.FLOAT), prior_sigma2))
                    , -_half + _half * tf.log(prior_sigma2) - _half * tf.log(posterior_sigma2))

        kl = tf.matrix_set_diag(tf.expand_dims(kl, -3), tf.zeros((1, tf.shape(kl)[0]), dtype=Latnet.FLOAT))
        return tf.reduce_sum(kl[0])

    @staticmethod
    def get_priors(N):
        """
        Return parameters for prior distributions over W and A,
            W ~ Normal(prior_mu, prior_sigma2),
            A ~ Concrete(prior_alpha)

        Args:
            N: int. Number of nodes

        Returns:
            prior_mu: Tensor of size N x N.
            prior_sigma2: Tensor of size N x N.
            prior_alpha: Tensor of size N x N.
        """

        NbyN = (N, N)
        p = 0.5
        prior_mu = tf.zeros(NbyN, dtype=Latnet.FLOAT)
        prior_sigma2 = tf.multiply(tf.cast(1. / (N * p), Latnet.FLOAT), tf.ones(NbyN, dtype=Latnet.FLOAT))
        prior_alpha = tf.multiply(tf.cast(p / (1. - p), Latnet.FLOAT), tf.ones(NbyN, dtype=Latnet.FLOAT))
        return prior_mu, prior_sigma2, prior_alpha

    @staticmethod
    def get_vairables(t, Y, init_sigma2_g, init_sigma2_n, init_lengthscle, init_variance):
        """
        Get tensor variables for the parameters that will be optimized (variational and hyper-parameters).
            W ~ Normal(mu, sigma2)
            A ~ Concrete(exp(log_alpha))

        Args:
            t: Tensor of shape T x 1
            Y: Tensor of shape T x N
            init_sigma2_g: init value of variance of connection noise.
            init_sigma2_n: init value of variance of observation nose.
            init_lengthscle: init value of kernel length-scale.
            init_variance: init value of kernel variance.

        Returns:
            log_sigma2_n: Tensor variable of shape ().
            log_sigma2_g: Tensor variable of shape ().
            mu: Tensor variable of shape N x N.
            log_sigma2: Tensor variable of shape N x N, which contains logarithm of sigma2.
            log_alpha: Tensor variable of shape N x N, which contains logarithm of alpha.
            log_lengthscale: Tensor variable of shape (), which contains logarithm of length-scale.
            log_variance: Tensor variable of shape (), which contains logarithm of variance.
        """

        N = Y.shape[1]
        T = Y.shape[0]

        prior_mu, prior_sigma2, prior_alpha = Latnet.get_priors(N)

        mu = tf.Variable(prior_mu, dtype=Latnet.FLOAT)
        log_sigma2 = tf.Variable(tf.log(prior_sigma2), dtype=Latnet.FLOAT)
        log_alpha = tf.Variable(tf.log(prior_alpha), dtype=Latnet.FLOAT)

        log_sigma2_n = tf.Variable(tf.log(tf.constant(init_sigma2_n, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)
        log_sigma2_g = tf.Variable(tf.log(tf.constant(init_sigma2_g, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)

        log_lengthscale = tf.Variable(tf.log(tf.constant(init_lengthscle, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)
        log_variance = tf.Variable(tf.log(tf.constant(init_variance, dtype=Latnet.FLOAT)), dtype=Latnet.FLOAT)

        return log_sigma2_n, log_sigma2_g, mu, log_sigma2, log_alpha, log_lengthscale, log_variance

    @staticmethod
    def get_elbo(t, Y, sigma2_n, sigma2_g, mu, sigma2, alpha, lengthscale, variance,
                 prior_lambda_, posterior_lambda_, n_samples, seed=None):

        """
        Calculates evidence lower bound (ELBO) for a set of posterior parameters using re-parametrization trick.

        posterior:    W ~ Normal(mu, sigma2), A ~ Normal(alpha)

        Args:
            t: Tensor with shape N x 1, which includes observation times.
            Y: Tensor wit shape T x N, which includes observation from N nodes at T time points.
            sigma2_n: Tensor with shape (). Observation noise variance.
            sigma2_g: Tensor with shape (). Connection noise variance.
            mu: Tensor with shape N x N.
            sigma2: Tensor with shape N x N.
            alpha: Tensor with shape N x N.
            lengthscale: double. Tensor with shape ().
            variance: double. Tensor with shape ().
            prior_lambda_: double. Tensor with shape ().
            posterior_lambda_: double. Tensor with shape ().
            n_samples: int. Tensor with shape (). Number of Monte-Carlo samples.
            seed: int. Seed used for random number generators.

        Returns:
            KL_normal: Tensor of shape () which contains KL-divergence between W_prior and W_posterior.
            KL_logistic:  Tensor of shape () which contains approximated KL-divergence between A_prior and A_posterior.
            ell: Tensor of shape () which contains approximated log-likelihood using Monte-Carlo samples.
        """

        # number of nodes
        N = Y.shape[1]

        # number of observation per node.
        T = Y.shape[0]

        # tensor of shape T x T with kernel values.
        Kt = Latnet.kernel(t, variance, lengthscale, T)

        #number of Monte-Caro samples
        S = n_samples
        eps = tf.constant(1e-20, dtype=Latnet.FLOAT)

        SbyNbyN = ((S, N, N))

        # sampling for W
        if seed is not None:
            z = tf.random_normal(SbyNbyN, dtype=Latnet.FLOAT, seed=seed)
        else:
            z = tf.random_normal(SbyNbyN, dtype=Latnet.FLOAT)

        Y = tf.constant(Y, dtype=Latnet.FLOAT)
        W = tf.multiply(z, tf.sqrt(sigma2)) + mu
        W = tf.matrix_set_diag(W, tf.zeros((S, N), dtype=Latnet.FLOAT))

        # sampling for A

        if seed is not None:
            U = tf.random_uniform(SbyNbyN, minval=0, maxval=1, dtype=Latnet.FLOAT, seed=seed)
        else:
            U = tf.random_uniform(SbyNbyN, minval=0, maxval=1, dtype=Latnet.FLOAT)

        X = tf.div(
            tf.add(tf.log(alpha), tf.subtract(tf.log(U + eps), tf.log(tf.constant(1.0, dtype=Latnet.FLOAT) - U + eps))),
            posterior_lambda_)
        A = tf.sigmoid(X)
        A = tf.matrix_set_diag(A, tf.zeros((S, N), dtype=Latnet.FLOAT))

        prior_mu, prior_sigma2, prior_alpha = Latnet.get_priors(N)

        return \
            Latnet.get_KL_normal(mu, sigma2, prior_mu, prior_sigma2), \
            Latnet.get_KL_logistic(X, alpha, prior_lambda_, posterior_lambda_, prior_alpha), \
            Latnet.batch_ll(Kt, t, Y, sigma2_n, sigma2_g, A, W, N, T, S)

    @staticmethod
    def batch_ll(Kt, t, Y, sigma2_n, sigma2_g, A, W, N, T, S):
        """
        Approximates log-expected likelihood term using Monte-Carlo samples.

        Args:
            Kt: Tensor of shape N x N containing kernel values at observation times `t'.
            t: Tensor of shape N x 1 containing observation times.
            Y: Tensor of shape T x N containing T observations from N nodes.
            sigma2_n: Tensor of shape () containing observation noise variance.
            sigma2_g:  Tensor of shape () containing connection noise variance
            A: Tensor of shape S x N x N containing samples from A for each connection ij.
            W: Tensor of shape S x N x N containing samples from W for each connection ij.
            N: Number of nodes.
            T: Number of observation per nodes.
            S: Number of samples.

        Returns:
            Tesnro of shape () which contains approximated expected log-likelihood.
        """

        # if version_comp("0.12", tf.__version__[0:4]) > 0:
        #     matmul = tf.batch_matmul
        # else:
        matmul = tf.matmul

        I = tf.constant(np.identity(N), dtype=Latnet.FLOAT)
        B = tf.multiply(A, W)
        IB = tf.subtract(I, B)
        L = tf.matrix_inverse(IB)
        Eg = matmul(matmul(L, matmul(B, tf.transpose(B, [0, 2, 1]))),
                    tf.transpose(L, [0, 2, 1]))
        Sn = tf.add(tf.multiply(sigma2_g, Eg), tf.multiply(sigma2_n, I))
        Kf = tf.matrix_inverse(matmul(tf.transpose(IB, [0, 2, 1]), IB))
        lt, Qt = tf.self_adjoint_eig(Kt)
        lt = tf.cast(lt, Latnet.FLOAT)
        Qt = tf.cast(Qt, Latnet.FLOAT)
        ln, Qn = tf.self_adjoint_eig(Sn)
        Ln_inv = tf.matrix_diag(tf.sqrt(tf.div(tf.constant(1., dtype=Latnet.FLOAT), ln)))
        Ln_inv_Qn = matmul(Ln_inv, tf.transpose(Qn, [0, 2, 1]))
        Ktilda_f = matmul(matmul(Ln_inv_Qn, Kf), tf.transpose(Ln_inv_Qn, [0, 2, 1]))
        ltilda_f, Qtilda_f = tf.self_adjoint_eig(Ktilda_f)

        Lt_Lf = tf.add(matmul(tf.transpose(tf.expand_dims(tf.tile(tf.expand_dims(lt, -1), [1, S]), -3)),
                              tf.expand_dims(ltilda_f, -2)),
                       tf.constant(1.0, dtype=Latnet.FLOAT))

        logdet = tf.multiply(tf.cast(T, Latnet.FLOAT), tf.reduce_sum(tf.log(ln))) + \
                 tf.reduce_sum(tf.log(Lt_Lf))

        Ytilda = matmul(matmul(tf.tile(tf.expand_dims(Y, -3), [S, 1, 1]), Qn), Ln_inv)

        Qt_expanded = tf.tile(tf.expand_dims(Qt, -3), [S, 1, 1])
        Ytf = tf.multiply(tf.div(tf.constant(1.0, dtype=Latnet.FLOAT), Lt_Lf),
                          matmul(tf.transpose(Qt_expanded, [0, 2, 1]), matmul(Ytilda, Qtilda_f)))
        ySy = tf.reduce_sum(tf.matrix_diag_part(
            matmul(matmul(matmul(tf.transpose(Ytilda, [0, 2, 1]), Qt_expanded), Ytf),
                   tf.transpose(Qtilda_f, [0, 2, 1]))))
        _half = tf.constant(0.5, dtype=Latnet.FLOAT)
        return tf.multiply(-_half, tf.div(logdet, S)) + tf.multiply(-_half, tf.div(ySy, S)) \
               - (N * T) / tf.constant(2.0, Latnet.FLOAT) * tf.log(tf.constant(2.0, dtype=Latnet.FLOAT) * np.pi)

    @staticmethod
    def optimize(t, Y, targets, total_iters, local_iters, logger,
                 init_sigma2_g=0.01, init_sigma2_n=0.1, init_lengthscle=0.1, init_variance=1.0,
                 lambda_prior=1.0, lambda_postetior=0.1,
                 var_lr=0.01, hyp_lr=1e-4, n_samples=200,
                 log_every=10,
                 callback=None,
                 seed=None,
                 fix_kernel=False
                 ):
        """
        Optimizing LATNET model based on a dataset.

        N: number of nodes
        T: number of observation per node

        Args:
            t: array of size N x 1
            Y: array of size N x T
            targets: list of string that determine with set of variables to optimize, e.g., ['var','hyp']
                'var': variational parameters
                'hyp': hyper-parameters (including kernel parameters and noises)
            total_iters: int. number of times that the parameters sets will be optimized (global iterations).
            local_iters: int. number of iterations that each set of parameters will be optimized in each global iteration.
            logger: looger that will be used for logging the results.
            init_sigma2_g: double. Initial variance of the connection noise.
            init_sigma2_n: double. Initial variance of the observation noise.
            init_lengthscle: double. Initial length-scale of the RBF kernel.
            init_variance: double. Initial variance of the RBF kenel.
            lambda_prior: double. lambda of the Concrete distribution used for prior over `A_ij' matrix.
            lambda_postetior: double. lambda of the Concrete distribution used for posterior over `A_ij' matrix.
            var_lr: double. Learning rate for updating variational parameters.
            hyp_lr: Learning rate for updating hyper-parameters.
            n_samples: int. Number of Monte-Carlo samples used.
            log_every: int. The frequency of logging in each iteration.
            callback: Callback function after each iteration.
            seed: int. Seed used for random number generators.
            fix_kernel: Boolean. Whether to optimize kernel parameters during the optimization of hyper-parameters.

        Returns:

            W ~ N(mu, sigma2)
            A ~ Concrete(alpha)

            elbo_: double. Optimized ELBO.
            sigma2_n_: double. Optimized sigma2_n, i.e., observation noise variance.
            sigma2_g_: double. Optimized sigma2_g, i.e., observation connection variance.
            mu_: matrix of size N x N.
            sigma2_: matrix of size N x N.
            alpha_: matrix of size N x N.
            lengthscale: double. Optimized length scale of the kernel.
            variance: double. Optimized variance of variance of RBF kernel.
        """

        with tf.device('/device:GPU:0'):
            var_opt, hyp_opt, \
            elbo, \
            kl_normal, \
            kl_logstic, \
            ell, \
            log_sigma2_n, log_sigma2_g, mu, log_sigma2, log_alpha, \
            log_lengthscale, log_variance, var_nans, hyp_nans = Latnet.run_model(tf.cast(t, Latnet.FLOAT), Y,
                                                                                 init_sigma2_g, init_sigma2_n,
                                                                                 init_lengthscle, init_variance,
                                                                                 lambda_prior, lambda_postetior,
                                                                                 var_lr, hyp_lr,
                                                                                 n_samples,
                                                                                 seed=seed,
                                                                                 fixed_kernel=fix_kernel
                                                                                 )

        # not to use GPUs for optimization because of memory problem.
        config = tf.ConfigProto(log_device_placement=True)

        with tf.Session(config=config) as sess:

            # with tf.Session() as sess:

            init_op = tf.initialize_all_variables()
            sess.run(init_op)

            # current global iteration over optimization steps.
            _iter = 0

            logger.debug("prior lambda={:f}; posterior lambda={:f}; variational learning rate={:f};"
                         " hyper learning rate={:f}".format(lambda_prior, lambda_postetior, var_lr, hyp_lr))

            while total_iters is None or _iter < total_iters:
                logger.debug("iteration %d started" % _iter)

                # optimizing variational parameters
                if 'var' in targets:
                    logger.debug("optimizing variational parameters")
                    for i in range(0, local_iters['var']):
                        try:
                            output = sess.run([elbo, kl_normal, kl_logstic, ell, var_opt, var_nans])
                            if i % log_every == 0:
                                logger.debug(
                                    'elbo at local iter {:d} and glocal iter {:d} is {:f} (KL norm={:f}; KL logistic={:f}, ell={:f}) with {:d} nan in grads (error= {:d}). '
                                    .format(i, _iter, output[0], output[1], output[2], output[3], output[5],
                                            output[5] != 0))
                        except OpError as e:
                            logger.error(e.message)

                # optimizing hyper parameters
                if 'hyp' in targets:
                    logger.debug("optimizing hyper parameters")
                    logger.debug("started optimizing hyper parameters")
                    for i in range(0, local_iters['hyp']):
                        try:
                            output = sess.run([elbo, kl_normal, kl_logstic, ell, hyp_opt, hyp_nans])
                            if i % log_every == 0:
                                logger.debug(
                                    'elbo at local iter {:d} and glocal iter {:d} is {:f} (KL norm={:f}; KL logistic={:f}, ell={:f}) with {:d} nan in grads (error= {:d}). '
                                    .format(i, _iter, output[0], output[1], output[2], output[3], output[5],
                                            output[5] != 0))
                        except OpError as e:
                            logger.error(e.message)
                if callback is not None:
                    sigma2_n_, sigma2_g_, mu_, sigma2_, alpha_, lengthscale_, variance_ = \
                        sess.run((
                                  tf.exp(log_sigma2_n),
                                  tf.exp(log_sigma2_g),
                                  mu,
                                  tf.exp(log_sigma2),
                                  tf.exp(log_alpha),
                                  tf.exp(log_lengthscale),
                                  tf.exp(log_variance)))
                    callback(alpha_, mu_, sigma2_, sigma2_n_, sigma2_g_, lengthscale_, variance_)
                _iter += 1

            elbo_, sigma2_n_, sigma2_g_, mu_, sigma2_, alpha_, lengthscale, variance = \
                sess.run((elbo,
                          tf.exp(log_sigma2_n),
                          tf.exp(log_sigma2_g),
                          mu,
                          tf.exp(log_sigma2),
                          tf.exp(log_alpha),
                          tf.exp(log_lengthscale),
                          tf.exp(log_variance)))

        return elbo_, sigma2_n_, sigma2_g_, mu_, sigma2_, alpha_, lengthscale, variance


        # this method was used for make sure that batch calculatino of ELL is working properly.
        # @staticmethod
        # def test_batch_vs_seq(t, Y, sigma2_n, sigma2_g, mu, sigma2, P):
        #
        #     S = 10
        #     SbyBbyN = ((S, tf.shape(mu)[0], tf.shape(mu)[0]))
        #     z = tf.random_normal(SbyBbyN, dtype=Latnet.FLOAT)
        #
        #     N = Y.shape[1]
        #     T = Y.shape[0]
        #
        #     Y = tf.constant(Y, dtype=Latnet.FLOAT)
        #     W = tf.multiply(z, sigma2) + mu
        #     A = W
        #     Kt= Latnet.get_kernel(t[0])
        #     t = tf.constant(t[0], Latnet.FLOAT)
        #
        #     i = tf.constant(0)
        #     ld = tf.constant(0.0)
        #     quad = tf.constant(0.0)
        #     c = lambda i, ld, quad: tf.less(i, 10)
        #     def b(i, ld, quad):
        #         return tf.add(i, 1), \
        #         tf.add(Latnet.get_ll(Kt, t, Y, sigma2_n, sigma2_g, A[i], W[i], N, T, S)[0], ld), \
        #         tf.add(Latnet.get_ll(Kt, t, Y, sigma2_n, sigma2_g, A[i], W[i], N, T, S)[1], quad)
        #     r = tf.while_loop(c, b, [i, ld, quad])
        #
        #     return Latnet.batch_ll(Kt, t, Y, sigma2_n, sigma2_g, A, W, N, T, S), tf.div(r[1], S),tf.div(r[2], S)
        #
        #
        # this method calculates ELL, but not in batch mode.
        # @staticmethod
        # def get_ll(Kt, t, Y, sigma2_n, sigma2_g, A, W, N, T, S):
        #
        #     I = tf.constant(np.identity(N), dtype=Latnet.FLOAT)
        #     B = tf.multiply(A, W)
        #     B = tf.matrix_set_diag(B, tf.zeros((S, N)))
        #     L = tf.matrix_inverse(tf.subtract(I, B))
        #     Eg = tf.matmul(tf.matmul(L, tf.matmul(B, tf.transpose(B))), tf.transpose(L))
        #     Sn = tf.add(tf.multiply(sigma2_g, Eg), tf.multiply(sigma2_n, I))
        #     Kf = tf.matmul(L, tf.transpose(L))
        #     lt, Qt = tf.self_adjoint_eig(Kt)
        #     lt = tf.cast(lt, Latnet.FLOAT)
        #     Qt = tf.cast(Qt, Latnet.FLOAT)
        #     ln, Qn = tf.self_adjoint_eig(Sn)
        #     Ln_inv = tf.diag(tf.sqrt(tf.div(tf.constant(1., dtype=Latnet.FLOAT), ln)))
        #     Ktilda_f = tf.matmul(tf.matmul(Ln_inv, tf.matmul(tf.matmul(tf.transpose(Qn), Kf), Qn)), Ln_inv)
        #     ltilda_f, Qtilda_f = tf.self_adjoint_eig(Ktilda_f)
        #
        #     Lt_Lf = tf.add(tf.matmul(tf.expand_dims(lt, -1), tf.expand_dims(ltilda_f, -2)),
        #                                      tf.constant(1.0, dtype=Latnet.FLOAT))
        #     logdet = tf.multiply(tf.cast(T, Latnet.FLOAT), tf.reduce_sum(tf.log(ln))) + \
        #          tf.reduce_sum(tf.log(Lt_Lf))
        #
        #     Ytilda = tf.matmul(tf.matmul(Y, Qn), Ln_inv)
        #     Ytf = tf.multiply(tf.div(tf.constant(1.0, dtype=Latnet.FLOAT), Lt_Lf), tf.matmul(tf.transpose(Qt), tf.matmul(Ytilda, Qtilda_f)))
        #     ySy = tf.trace(tf.matmul(tf.matmul(tf.matmul(tf.transpose(Ytilda), Qt), Ytf), tf.transpose(Qtilda_f)))
        #
        #     return logdet, ySy
