"""Multivariate t-distribution.

Author: Gregory Gundersen 2020. Architecture based on SciPy's
`_multivariate.py` module by Joris Vankerschaver 2013.

Modified: Paul Sarbu 2020.
"""

import numpy as np
from scipy._lib._util import check_random_state
from scipy.stats._multivariate import _PSD, multi_rv_generic, multi_rv_frozen
from scipy.stats._multivariate import _squeeze_output
from scipy.stats import multivariate_normal
from scipy.special import gammaln


# -----------------------------------------------------------------------------

class multivariate_t_gen(multi_rv_generic):

    def __init__(self, seed=None):
        """Initialize a multivariate t-distributed random variable.

        Parameters
        ----------
        seed : Random state.
        """
        self._random_state = check_random_state(seed)

    def __call__(self, loc=None, scale=1, df=1, seed=None):
        """Create a frozen multivariate t-distribution. See
        `multivariate_t_frozen` for parameters.
        """
        return multivariate_t_frozen(df, loc=loc, scale=scale, seed=seed)

    def pdf(self, x, df, loc=None, scale=1):
        """Multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        df : int or float
            Degrees of freedom.
        loc : array_like, optional
            Mean of the distribution (default zero).
        scale : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).

        Returns
        -------
        logpdf : Probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        df = self._process_degrees_of_freedom(df)

        if np.inf == df:
            # multivariate normal distribution
            lp = multivariate_normal.logpdf(x, mean=loc, cov=scale)
        else:
            dim, loc, scale = self._process_parameters(loc, scale)
            x = self._process_quantiles(x, dim)
            scale_info = _PSD(scale)

            # generic multivariate t distribution
            lp = self._logpdf(x, df, loc, scale_info.U, scale_info.log_pdet, dim)

        return _squeeze_output(np.exp(lp))

    def logpdf(self, x, df, loc=None, scale=1):
        """Log of the multivariate t-distribution probability density function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        df : int or float
            Degrees of freedom.
        loc : array_like, optional
            Mean of the distribution (default zero).
        scale : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).

        Returns
        -------
        logpdf : Log of the probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        df = self._process_degrees_of_freedom(df)

        if np.inf == df:
            # multivariate normal distribution
            lp = multivariate_normal.logpdf(x, mean=loc, cov=scale)
        else:
            dim, loc, scale = self._process_parameters(loc, scale)
            x = self._process_quantiles(x, dim)
            scale_info = _PSD(scale)

            # generic multivariate t distribution
            lp = self._logpdf(x, df, loc, scale_info.U, scale_info.log_pdet, dim)

        return _squeeze_output(lp)

    def _logpdf(self, x, df, loc, U, log_pdet, dim):
        """Utility method. See `pdf`, `logpdf` for parameters.
        """
        dev = x - loc  # (n,d)
        maha = np.square(np.dot(dev, U)).sum(axis=-1)  # (n,)

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * np.log(df * np.pi)
        D = 0.5 * log_pdet
        E = -t * np.log(1 + (1./df) * maha)  # (n,)

        return A - B - C - D + E  # (n,)

    def rvs(self, df, loc=None, scale=1, size=1, random_state=None):
        """Draw random samples from a multivariate t-distribution.

        Parameters
        ----------
        df : int or float
            Degrees of freedom.
        loc : array_like, optional
            Mean of the distribution (default zero).
        scale : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).
        size : integer, optional
            Number of samples to draw (default one).
        random_state : {None, int, np.random.RandomState, np.random.Generator}, optional
            Used for drawing random variates (default is None).

        Returns
        -------
        samples : Drawn samples from the multivariate skew-normal-distribution.

        Examples
        --------
        FIXME.
        """
        df = self._process_degrees_of_freedom(df)
        dim, loc, scale = self._process_parameters(loc, scale)

        if random_state is not None:
            rng = check_random_state(random_state)

        else:
            rng = self._random_state

        if np.inf == df:
            # multivariate normal distribution
            x = np.ones(size)

        else:
            # generic multivariate t-distribution
            x = rng.chisquare(df, size=size) / df

        z = rng.multivariate_normal(mean=np.zeros(dim), cov=scale,
                                    size=size)  # (n,d)
        # add location and scale
        samples = loc + z / np.sqrt(x)[:, None]

        return _squeeze_output(samples)

    def dpdf(self, x, df, loc=None, scale=1):
        """Derivative of the multivariate t-distribution probability density
        function.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        df : int or float
            Degrees of freedom.
        loc : array_like, optional
            Mean of the distribution (default zero).
        scale : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).

        Returns
        -------
        dpdf : Derivative of the probability density function evaluated at `x`.

        Examples
        --------
        FIXME.
        """
        df = self._process_degrees_of_freedom(df)
        dim, loc, scale = self._process_parameters(loc, scale)
        x = self._process_quantiles(x, dim)
        scale_info = _PSD(scale)

        if np.inf == df:
            # multivariate normal distribution
            out = self._dmnpdf(x, loc, scale, scale_info.pinv)

        else:
            # generic multivariate t-distribution
            dev = x - loc  # (n,d)
            maha = np.square(np.dot(dev, scale_info.U)).sum(axis=-1)  # (n,)

            t = 0.5 * (df + dim)
            A = gammaln(t)
            B = gammaln(0.5 * df)
            C = dim/2. * np.log(df * np.pi)
            D = 0.5 * scale_info.log_pdet
            E = -(t+1) * np.log(1 + (1./df) * maha)  # (n,)

            sm = np.dot(dev, scale_info.pinv)  # (n,d)*(d,d)=(n,d)

            out = -2. * t / df * np.transpose(
                np.transpose(sm) * np.exp(A - B - C - D + E))  # (n,d)

        return _squeeze_output(out)

    def _dmnpdf(self, x, loc, scale, scale_inv):
        """Utility method computing derivative of multivariate normal
        distribution. See `dpdf` for parameters.
        """
        mnpdf = multivariate_normal.pdf(x, mean=loc, cov=scale)  # (n,)
        L = -np.dot(x - loc, scale_inv)  # (n,d)
        return np.transpose(np.transpose(L) * mnpdf)  # (n,d)

    def _process_quantiles(self, x, dim):
        """Adjust quantiles array so that last axis labels the components of
        each data point.
        """
        x = np.asarray(x, dtype=float)
        if x.ndim == 0:
            x = x[np.newaxis]
        elif x.ndim == 1:
            if dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        return x

    def _process_degrees_of_freedom(self, df):
        """Make sure degrees of freedom are valid. Separate treatment to
        avoid duplication of code when df == np.inf
        """
        if df is None:
            df = 1
        elif df <= 0:
            raise ValueError("'df' must be greater than zero.")
        elif np.isnan(df):
            raise ValueError("'df' is 'nan' but must be greater than zero or 'np.inf'.")

        return df

    def _process_parameters(self, loc, scale):
        """Infer dimensionality from loc array and scale matrix, handle
        defaults, and ensure compatible dimensions.
        """
        if loc is None and scale is None:
            loc = np.asarray(0, dtype=float)
            scale = np.asarray(1, dtype=float)
            dim = 1
        elif loc is None:
            scale = np.asarray(scale, dtype=float)
            if scale.ndim < 2:
                dim = 1
            else:
                dim = scale.shape[0]
            loc = np.zeros(dim)
        elif scale is None:
            loc = np.asarray(loc, dtype=float)
            dim = loc.size
            scale = np.eye(dim)
        else:
            scale = np.asarray(scale, dtype=float)
            loc = np.asarray(loc, dtype=float)
            dim = loc.size

        if dim == 1:
            loc.shape = (1,)
            scale.shape = (1, 1)

        if loc.ndim != 1 or loc.shape[0] != dim:
            raise ValueError("Array 'loc' must be a vector of length %d." %
                             dim)
        if scale.ndim == 0:
            scale = scale * np.eye(dim)
        elif scale.ndim == 1:
            scale = np.diag(scale)
        elif scale.ndim == 2 and scale.shape != (dim, dim):
            rows, cols = scale.shape
            if rows != cols:
                msg = ("Array 'cov' must be square if it is two dimensional,"
                       " but cov.shape = %s." % str(scale.shape))
            else:
                msg = ("Dimension mismatch: array 'cov' is of shape %s,"
                       " but 'loc' is a vector of length %d.")
                msg = msg % (str(scale.shape), len(loc))
            raise ValueError(msg)
        elif scale.ndim > 2:
            raise ValueError("Array 'cov' must be at most two-dimensional,"
                             " but cov.ndim = %d" % scale.ndim)

        return dim, loc, scale


class multivariate_t_frozen(multi_rv_frozen):

    def __init__(self, df, loc=None, scale=1, seed=None):
        """
        Create a frozen multivariate normal distribution.

        Parameters
        ----------
        x : array_like
            Points at which to evaluate the log of the probability density
            function.
        df : int or float
            Degrees of freedom.
        loc : array_like, optional
            Mean of the distribution (default zero).
        scale : array_like, optional
            Positive definite shape matrix. This is not the distribution's
            covariance matrix (default one).

        Examples
        --------
        FIXME.
        """
        self._dist = multivariate_t_gen(seed)
        df = self._dist._process_degrees_of_freedom(df)
        dim, loc, scale = self._dist._process_parameters(loc, scale)
        self.dim, self.df, self.loc, self.scale = dim, df, loc, scale
        self.scale_info = _PSD(scale)

    def logpdf(self, x):
        x = self._dist._process_quantiles(x, self.dim)
        U = self.scale_info.U
        log_pdet = self.scale_info.log_pdet
        return self._dist._logpdf(x, self.df, self.loc, U, log_pdet, self.dim)

    def pdf(self, x):
        return np.exp(self.logpdf(x))

    def rvs(self, size=1, random_state=None):
        """
        Draw random samples from a multivariate normal distribution.

        Parameters
        ----------
        size : integer, optional
            Number of samples to draw (default 1).
        random_state : np.random.RandomState instance
            RandomState used for drawing the random variates.

        Returns
        -------
        rvs : ndarray or scalar
            Random variates of size (`size`, `N`), where `N` is the
            dimension of the random variable.
        """
        return self._dist.rvs(self.df,
                              loc=self.loc,
                              scale=self.scale,
                              size=size,
                              random_state=random_state)


# -----------------------------------------------------------------------------
multivariate_t = multivariate_t_gen()