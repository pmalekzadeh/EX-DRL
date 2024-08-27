import unittest
from scipy.stats import norm
import numpy as np
from bsm_pricers import vanilla_price_and_greeks, dip_barrier_option_analytical
from domain.pricer.bsm_pricers import _phi, _norm_pdf

class PricerTestCase(unittest.TestCase):
    
    def test_phi(self):
        self.assertAlmostEqual(_phi(np.array([1.2]))[0], norm.cdf(1.2), delta=1e-6)
        self.assertAlmostEqual(_phi(np.array([0.2]))[0], norm.cdf(0.2), delta=1e-6)

    def test_norm_pdf(self):
        self.assertAlmostEqual(_norm_pdf(np.array([1.2]))[0], norm.pdf(1.2), delta=1e-6)
        self.assertAlmostEqual(_norm_pdf(np.array([0.2]))[0], norm.pdf(0.2), delta=1e-6)


class TestPriceAndGreeks(unittest.TestCase):
    def test_vanilla_price_and_greeks(self):
        ttm = np.array([252.0, 31])
        S = np.array([100.0, 9.353995306])
        iv = np.array([0.2, 0.3])
        K = np.array([100.0, 10.0])
        call = True
        T = np.array([252.0, 252.0])

        price, delta, gamma, vega = vanilla_price_and_greeks(ttm, S, iv, K, call, T)

        # These are the expected values. You might need to adjust them based on your specific implementation.
        expected_price = np.array([7.96558, 0.16205])
        expected_delta = np.array([0.53983, 0.28026])
        expected_gamma = np.array([0.01985, 0.34217])
        expected_vega = np.array([0.39695, 0.0110489])

        np.testing.assert_almost_equal(price, expected_price, decimal=5)
        np.testing.assert_almost_equal(delta, expected_delta, decimal=5)
        np.testing.assert_almost_equal(gamma, expected_gamma, decimal=5)
        np.testing.assert_almost_equal(vega, expected_vega, decimal=5)

    def test_barrier_price(self):
        # These are placeholder inputs. Replace them with the actual inputs for your function.
        S = np.array([100.0])
        K = np.array([100.0])
        T = np.array([1.0])
        vol = np.array([0.2])
        H = np.array([80.0])
        effective_discrete_barrier = H*np.exp(-0.5826*vol*np.sqrt(1/252))
        cross_indicator = np.array([0.0])
        
        price = dip_barrier_option_analytical(S, K, H, T, 0.0, 0.0, vol, cross_indicator)

        # This is a placeholder expected value. Replace it with the actual expected output for your function.
        expected_price = np.array([5.81])

        np.testing.assert_almost_equal(price, expected_price, decimal=2)


if __name__ == '__main__':
    unittest.main()