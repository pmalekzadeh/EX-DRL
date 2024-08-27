import numba
import numpy as np
import math

@numba.jit(nopython=True)
def _phi(x):
    norm_cdf = np.zeros_like(x)
    for i in range(len(x)):
        norm_cdf[i] = 0.5 * (1.0 + math.erf(x[i] / math.sqrt(2.0)))
    return norm_cdf

@numba.jit(nopython=True)
def _norm_pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)

@numba.jit(nopython=True, cache=True)
def vanilla_price_and_greeks(ttm, S, iv, K, call, T):
    """Black Scholes Formula

    Returns:
        np.ndarray: option price in the same shape of stock price S.
        np.ndarray: option delta in the same shape of stock price S.
        np.ndarray: option gamma in the same shape of stock price S.
        np.ndarray: option vega in the same shape of stock price S.
    """
    q = r = 0.0
    
    # active option
    active_option = (ttm > 0).astype('uint0')
    matured_option = (ttm == 0).astype('uint0')

    # active option
    fT = np.maximum(ttm, 1)/T
    d1 = (np.log(S / K) + (r - q + iv * iv / 2)
            * np.abs(fT)) / (iv * np.sqrt(fT))
    d2 = d1 - iv * np.sqrt(fT)
    n_prime = np.exp(-1 * d1 * d1 / 2) / np.sqrt(2 * np.pi)

    active_bs_price = np.where(call, 
        S * np.exp(-q * fT) * _phi(d1) - K * \
            np.exp(-r * fT) * _phi(d2),
        - S * np.exp(-q * fT) * _phi(-d1) + K * \
            np.exp(-r * fT) * _phi(-d2))
    active_bs_delta = np.where(call, 
        np.exp(-q * fT) * _phi(d1), 
        - np.exp(-q * fT) * _phi(-d1))
    active_bs_gamma = (n_prime * np.exp(-q * fT)) / \
        (S * iv * np.sqrt(fT))
    active_bs_vega = (1/100) * S * np.exp(-q * fT) * \
        np.sqrt(fT) * n_prime

    # matured option
    payoff = np.maximum(S - K, 0)

    # consolidate
    price = active_option*active_bs_price + matured_option*payoff
    delta = active_option*active_bs_delta
    gamma = active_option*active_bs_gamma
    vega = active_option*active_bs_vega

    return price, delta, gamma, vega


@numba.jit(nopython=True, cache=True)
def dip_barrier_option_analytical(S, K, H, tau, r, q, sigma, barrier_crossing_indicator):
    # Adjustment to barrier for discrete monitoring
    H = H*np.exp(-0.5826*sigma*np.sqrt(1/252))

    barrier_crossing_indicator = np.where(
        S > H, barrier_crossing_indicator, np.ones_like(barrier_crossing_indicator))
    not_passed = (barrier_crossing_indicator == 0).astype('uint0')
    passed = (barrier_crossing_indicator == 1).astype('uint0')
    
    not_last_day = (tau > 0).astype('uint0')
    last_day = (tau == 0).astype('uint0')
    # set a number for tau=0 to prevent divide by zero, will not affect final value
    tau = np.where(tau == 0, 0.00001, tau)

    """ Use n_obs=-1 for continuous monitoring """
    # Black Scholes call and put prices - checked
    d1 = (np.log(S/K) + (r-q + 0.5 * sigma**2)
            * tau) / (sigma * np.sqrt(tau))
    d2 = (np.log(S/K) + (r-q - 0.5 * sigma**2)
            * tau) / (sigma * np.sqrt(tau))
    P = K * np.exp(-r*tau) * _phi(-d2) - S * \
        np.exp(-q*tau) * _phi(-d1)

    # Parameters - checked
    lambda_ = (r - q + 0.5*sigma**2) / sigma**2
    y = np.log((H**2)/(S*K)) / (sigma*np.sqrt(tau)) + \
        lambda_*sigma*np.sqrt(tau)
    x1 = np.log(S/H) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    y1 = np.log(H/S) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)


    def h_less_k():    
        P_di = not_passed*(-S*np.exp(-q*tau) * _phi(-x1) + K*np.exp(-r*tau) * _phi(-x1 + sigma*np.sqrt(tau))
                            + S*np.exp(-q*tau) * (H/S)**(2*lambda_) *
                            (_phi(y)-_phi(y1))
                            - K*np.exp(-r*tau) * (H/S)**(2*lambda_-2) * (_phi(y-sigma*np.sqrt(tau))-_phi(y1-sigma*np.sqrt(tau))))
        P_di = P_di+passed*P
        P_di = last_day*passed*np.maximum(K-S, 0) + not_last_day * P_di
        return P_di

    def h_greater_k():
        P_di = passed*P
        P_di = P_di+not_passed*P
        P_di = last_day*passed*np.maximum(K-S, 0) + not_last_day * P_di
        return P_di

    return np.where(H < K, h_less_k(), h_greater_k())
        
# Function to price continuously monitored barrier options' delta with closed-form solutions.
@numba.jit(nopython=True, cache=True)
def dip_barrier_delta_analytical(S, K, H, tau, r, q, sigma, barrier_crossing_indicator):
    """ Use n_obs=-1 for continuous monitoring """
    # Black Scholes call and put delta - checked
    d1 = (np.log(S/K) + (r-q + 0.5 * sigma**2)
            * tau) / (sigma * np.sqrt(tau))
    delta_P = -1 * np.exp(-q*tau) * _phi(-d1)

    # Adjustment to barrier for discrete monitoring
    H = H*np.exp(-0.5826*sigma*np.sqrt(1/252))

    # print(S, K, H, T, r, q, sigma, n_obs)
    # Parameters
    lambda_ = (r - q + 0.5*sigma**2) / sigma**2
    y = np.log((H**2)/(S*K)) / (sigma*np.sqrt(tau)) + \
        lambda_*sigma*np.sqrt(tau)
    x1 = np.log(S/H) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    y1 = np.log(H/S) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    dy_ds = -1/(S*sigma*np.sqrt(tau))
    dx1_ds = 1/(S*sigma*np.sqrt(tau))
    dy1_ds = -1/(S*sigma*np.sqrt(tau))
    barrier_crossing_indicator = np.where(
        S > H, barrier_crossing_indicator, np.ones_like(barrier_crossing_indicator))
    not_passed = (barrier_crossing_indicator == 0).astype('uint0')
    passed = (barrier_crossing_indicator == 1).astype('uint0')
    not_passed = (barrier_crossing_indicator == 0).astype('uint0')
    passed = (barrier_crossing_indicator == 1).astype('uint0')

    def h_less_k():
        delta_Pdi = not_passed*(-np.exp(-q*tau)*(_phi(-x1)-S*_norm_pdf(-x1)*dx1_ds)
                                - K*np.exp(-r*tau) * _norm_pdf(-x1+sigma*np.sqrt(tau))*dx1_ds+np.exp(-q*tau)*(H**(2*lambda_))*(
                                    (-2*lambda_+1)*(S**(-2*lambda_))*_phi(y)+(S**(-2*lambda_+1))*_norm_pdf(y)*dy_ds)
                                - np.exp(-q*tau)*(H**(2*lambda_))*((-2*lambda_+1)*(
                                    S**(-2*lambda_))*_phi(y1)+(S**(-2*lambda_+1))*_norm_pdf(y1)*dy1_ds)
                                - K*np.exp(-r*tau)*(H**(2*lambda_-2))*((-2*lambda_+2)*(S**(-2*lambda_+1))*_phi(
                                    y-sigma*np.sqrt(tau))+(S**(-2*lambda_+2))*_norm_pdf(y-sigma*np.sqrt(tau))*dy_ds)
                                + K*np.exp(-r*tau)*(H**(2*lambda_-2))*((-2*lambda_+2)*(S**(-2*lambda_+1))*_phi(y1-sigma*np.sqrt(tau))+(S**(-2*lambda_+2))*_norm_pdf(y1-sigma*np.sqrt(tau))*dy1_ds))
        delta_Pdi = delta_Pdi+passed*delta_P
        return delta_Pdi
    
    def h_greater_k():
        delta_Pdi = passed*delta_P
        delta_Pdi = delta_Pdi+not_passed*delta_P
        return delta_Pdi

    return np.where(H < K, h_less_k(), h_greater_k())

# Function to price continuously monitored barrier options' vega with closed-form solutions.
# NOTE: (EXTREMELY IMPORTANT) This implementation is not all correct. I have checked a few cases with DerivaGem but haven't done all of them.
# So please only use the Delta and Gamma greeks' analytical solution. The vega one is not checked. Delta and Gamma and the option price ones are checked multiple-times and are        #### error-free.
@numba.jit(nopython=True, cache=True)
def dip_barrier_vega_analytical(S, K, H, tau, r, q, sigma, barrier_crossing_indicator):
    # Black Scholes call and put vega
    d1 = (np.log(S/K) + (r-q + 0.5 * sigma**2)
            * tau) / (sigma * np.sqrt(tau))
    vega_P = (1/100)*S*np.exp(-q*tau)*_norm_pdf(d1)*np.sqrt(tau)

    # Adjustment to barrier for discrete monitoring
    H = H*np.exp(-0.5826*sigma*np.sqrt(1/252))

    # Parameters
    lambda_ = (r - q + 0.5*sigma**2) / sigma**2
    y = np.log((H**2)/(S*K)) / (sigma*np.sqrt(tau)) + \
        lambda_*sigma*np.sqrt(tau)
    x1 = np.log(S/H) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    y1 = np.log(H/S) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    dlambda_dsigma = 1/sigma - 1/(sigma**3)*(2*r-2*q+sigma**2)
    dy_dsigma = -(np.log((H**2)/(S*K))/((sigma**2) *
                    np.sqrt(tau))) + lambda_*np.sqrt(tau)
    dx1_dsigma = -(np.log(S/H)/((sigma**2)*np.sqrt(tau))) + \
        lambda_*np.sqrt(tau)
    dy1_dsigma = -(np.log(H/S)/((sigma**2)*np.sqrt(tau))) + \
        lambda_*np.sqrt(tau)

    not_passed = (barrier_crossing_indicator == 0).astype('uint0')
    passed = (barrier_crossing_indicator == 1).astype('uint0')

    def h_less_k():
        vega_Pdi = not_passed*((1/100)*(S*np.exp(-q*tau)*_norm_pdf(-x1)*dx1_dsigma + K*np.exp(-r*tau)*(_norm_pdf(-x1+sigma*np.sqrt(tau))*(-dx1_dsigma+np.sqrt(tau)))
                                        + S * np.exp(-q*tau) * (np.log(H/S)*(H/S)**(2*lambda_)*2*dlambda_dsigma*_phi(
                                            y)+(H/S)**(2*lambda_)*_norm_pdf(y)*(dy_dsigma))
                                        - S * np.exp(-q*tau) * (np.log(H/S)*(H/S)**(2*lambda_)*2*dlambda_dsigma*_phi(
                                            y1)+(H/S)**(2*lambda_)*_norm_pdf(y1)*(dy1_dsigma))
                                        - K*np.exp(-r*tau)*(np.log(H/S)*(H/S)**(2*lambda_-2)*2*dlambda_dsigma*_phi(y-sigma*np.sqrt(
                                            tau))+(H/S)**(2*lambda_-2)*_norm_pdf(y-sigma*np.sqrt(tau))*(dy_dsigma-np.sqrt(tau)))
                                        + K*np.exp(-r*tau)*(np.log(H/S)*(H/S)**(2*lambda_-2)*2*dlambda_dsigma*_phi(y1-sigma*np.sqrt(tau))+(H/S)**(2*lambda_-2)*_norm_pdf(y1-sigma*np.sqrt(tau))*(dy1_dsigma-np.sqrt(tau)))))
        vega_Pdi = vega_Pdi+passed*vega_P
        return vega_Pdi
    
    def h_greater_k():
        vega_Pdi = passed*vega_P
        vega_Pdi = not_passed*vega_P
        return vega_Pdi

    return np.where(H < K, h_less_k(), h_greater_k())

# Function to price continuously monitored barrier options' gamma with closed-form solutions.
@numba.jit(nopython=True, cache=True)
def dip_barrier_gamma_analytical(S, K, H, tau, r, q, sigma, barrier_crossing_indicator):
    # Black Scholes call and put gamma
    d1 = (np.log(S/K) + (r-q + 0.5 * sigma**2)
            * tau) / (sigma * np.sqrt(tau))
    gamma_P = np.exp(-q*tau) * _norm_pdf(d1)/(S*sigma*np.sqrt(tau))

    # Adjustment to barrier for discrete monitoring
    H = H*np.exp(-0.5826*sigma*np.sqrt(1/252))

    # Parameters
    lambda_ = (r - q + 0.5*sigma**2) / sigma**2
    y = np.log((H**2)/(S*K)) / (sigma*np.sqrt(tau)) + \
        lambda_*sigma*np.sqrt(tau)
    x1 = np.log(S/H) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    y1 = np.log(H/S) / (sigma*np.sqrt(tau)) + lambda_*sigma*np.sqrt(tau)
    dy_ds = -1/(S*sigma*np.sqrt(tau))
    dx1_ds = 1/(S*sigma*np.sqrt(tau))
    dy1_ds = -1/(S*sigma*np.sqrt(tau))
    phi_y_prime = -1/(np.sqrt(2*np.pi))*y*np.exp(-1/2*(y**2))*dy_ds
    phi_x1_prime = -1/(np.sqrt(2*np.pi))*x1*np.exp(-1/2*(x1**2))*dx1_ds
    phi_y1_prime = -1/(np.sqrt(2*np.pi))*y1*np.exp(-1/2*(y1**2))*dy1_ds
    phi_neg_x1_prime = phi_x1_prime
    phi_y_minus_sigma_sqrt_T_prime = -1 / \
        (np.sqrt(2*np.pi))*(y-sigma*np.sqrt(tau)) * \
        np.exp(-1/2*((y-sigma*np.sqrt(tau))**2))*dy_ds
    phi_x1_minus_sigma_sqrt_T_prime = -1 / \
        (np.sqrt(2*np.pi))*(x1-sigma*np.sqrt(tau)) * \
        np.exp(-1/2*((x1-sigma*np.sqrt(tau))**2))*dx1_ds
    phi_y1_minus_sigma_sqrt_T_prime = -1 / \
        (np.sqrt(2*np.pi))*(y1-sigma*np.sqrt(tau)) * \
        np.exp(-1/2*((y1-sigma*np.sqrt(tau))**2))*dy1_ds
    phi_neg_x1_minus_sigma_sqrt_T_prime = phi_x1_minus_sigma_sqrt_T_prime
    dy2_ds2 = 1/(S**2*sigma*np.sqrt(tau))
    dx12_ds2 = -1/(S**2*sigma*np.sqrt(tau))
    dy12_ds2 = 1/(S**2*sigma*np.sqrt(tau))

    not_passed = (barrier_crossing_indicator == 0).astype('uint0')
    passed = (barrier_crossing_indicator == 1).astype('uint0')
    
    def h_less_k():
        gamma_Pdi = not_passed*(np.exp(-q*tau)*_norm_pdf(-x1)*dx1_ds + np.exp(-q*tau) * (_norm_pdf(-x1)*dx1_ds + S*phi_neg_x1_prime*dx1_ds + S*_norm_pdf(-x1)*dx12_ds2)
                                - K*np.exp(-r*tau)*(phi_neg_x1_minus_sigma_sqrt_T_prime *
                                                    dx1_ds + _norm_pdf(-x1+sigma*np.sqrt(tau))*dx12_ds2)
                                + (H**(2*lambda_))*np.exp(-q*tau)*(-2*lambda_+1)*((-2*lambda_)*(
                                    S**(-2*lambda_-1))*_phi(y) + (S**(-2*lambda_))*_norm_pdf(y)*dy_ds)
                                + (H**(2*lambda_)) * np.exp(-q*tau) * ((-2*lambda_+1) * (S**(-2*lambda_))*_norm_pdf(
                                    y)*dy_ds + (S**(-2*lambda_+1))*phi_y_prime*dy_ds + (S**(-2*lambda_+1))*_norm_pdf(y)*dy2_ds2)
                                - (H**(2*lambda_))*np.exp(-q*tau)*(-2*lambda_+1)*((-2*lambda_)*(
                                    S**(-2*lambda_-1))*_phi(y1) + (S**(-2*lambda_))*_norm_pdf(y1)*dy1_ds)
                                - (H**(2*lambda_)) * np.exp(-q*tau) * ((-2*lambda_+1) * (S**(-2*lambda_))*_norm_pdf(y1)*dy1_ds + (
                                    S**(-2*lambda_+1))*phi_y1_prime*dy1_ds + (S**(-2*lambda_+1))*_norm_pdf(y1)*dy12_ds2)
                                - K*np.exp(-r*tau)*(H**(2*lambda_-2))*(-2*lambda_+2)*((-2*lambda_+1)*(S**(-2*lambda_)) *
                                                                                        _phi(y-sigma*np.sqrt(tau)) + (S**(-2*lambda_+1))*_norm_pdf(y-sigma*np.sqrt(tau))*dy_ds)
                                - K*np.exp(-r*tau)*(H**(2*lambda_-2))*(phi_y_minus_sigma_sqrt_T_prime*(S**(-2*lambda_+2))*dy_ds + (S**(-2*lambda_+2))
                                                                        * _norm_pdf(y-sigma*np.sqrt(tau))*dy2_ds2 + _norm_pdf(y-sigma*np.sqrt(tau))*dy_ds*(-2*lambda_+2)*S**(-2*lambda_+1))
                                + K*np.exp(-r*tau)*(H**(2*lambda_-2))*(-2*lambda_+2)*((-2*lambda_+1)*(S**(-2*lambda_))*_phi(
                                    y1-sigma*np.sqrt(tau)) + (S**(-2*lambda_+1))*_norm_pdf(y1-sigma*np.sqrt(tau))*dy1_ds)
                                + K*np.exp(-r*tau)*(H**(2*lambda_-2))*(phi_y1_minus_sigma_sqrt_T_prime*(S**(-2*lambda_+2))*dy1_ds + (S**(-2*lambda_+2))*_norm_pdf(y1-sigma*np.sqrt(tau))*dy12_ds2 + _norm_pdf(y1-sigma*np.sqrt(tau))*dy1_ds*(-2*lambda_+2)*S**(-2*lambda_+1)))
        gamma_Pdi = gamma_Pdi+passed*gamma_P
        return gamma_Pdi
    
    def h_greater_k():
        gamma_Pdi = passed*gamma_P
        gamma_Pdi = gamma_Pdi+not_passed*gamma_P
        return gamma_Pdi
    
    return np.where(H < K, h_less_k(), h_greater_k())