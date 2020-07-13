import numpy as np
import math
from scipy.stats import norm
from numba import jit
import pandas as pd
import streamlit as st
class StreamlitBSModel:
    def __init__(self):
        pass

        
    def run(self):
        # Define Parameters
        st.sidebar.title('Black-Scholes parameterization')
        seed = st.sidebar.slider('Seed', min_value = 1, max_value = 999999, step = 1, value = 123456)
        s0 = st.sidebar.number_input('Initial value s0', value = 100., step = 10.)
        r = st.sidebar.slider('Interest rate r', value = 0.05, min_value = -0.5, max_value = 0.5, step = 0.01)
        sigma = st.sidebar.slider('Volatility', value = 0.5, min_value = 0.01, max_value = 4., step = 0.01)
        strike = st.sidebar.slider('European Strike', value = 100, min_value = 0, max_value = 1000, step = 10)

        st.sidebar.title('Grid parameterization')
        maturity = st.sidebar.number_input('Maturity in days', value = 1., step = 0.1)
        timestep = st.sidebar.slider('Timestep', value = 0.001, min_value = 0.0001, step = 0.0001, max_value = 0.01)

        bs_model = BlackScholesModel(seed, s0, r, sigma, timestep, maturity)
        bs_model.simulate_trajectory()
        bs_model.compute_call_trajectory(K = strike)

        s = bs_model.get_Series()
        c = bs_model.get_Call_Series()

        st.title('Black Scholes asset trajectory')
        st.line_chart(s)
        st.title('Black Scholes Call price')
        st.line_chart(c)

        st.title('Black Scholes Call curve')
        time_t = st.slider('Time', min_value = 0., max_value = maturity, value = 0., step = 0.0001)
        call_curve = bs_model.call_intrinsic_curve(K = strike, S_min = strike - 0.8 * strike, S_max = strike + 0.8 * strike, t = time_t)
        st.line_chart(call_curve)


        
class BlackScholesModel:
    __slots__ = ['seed', 's0', 'r', 'sigma', 'timestep', 'maturity', 'values', 'time_grid', 'call_values']
    def __init__(self, seed, s0, r, sigma, timestep, maturity):
        self.seed = seed
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.timestep = timestep
        self.maturity = maturity
        self.values = None

    def simulate_trajectory(self):
        self.time_grid = np.arange(0, self.maturity, self.timestep)
        self.values = self._simulate_trajectory(self.seed, self.s0, self.r, self.sigma, len(self.time_grid), self.timestep)
  
    def get_Series(self):
        if any(self.values) :
            return pd.Series(data = self.values, index = self.time_grid)
        else :
            return None

    def get_Call_Series(self):
        if any(self.call_values):
            return pd.Series(data = self.call_values, index = self.time_grid)
        else :
            return None

    @staticmethod
    @jit(nopython = True)
    def _simulate_trajectory(seed, s0, r, sigma, length, timestep):
        np.random.seed(seed)
        S = np.empty(length)
        location = (r - 0.5 * sigma**2) * timestep
        scale = np.sqrt(timestep) * sigma
        S[0] = s0
        for i in range(1, length):
            # S[i] = S[i - 1] * np.random.lognormal(location, scale)
            # S[i] = S[i - 1] * np.exp(location + scale * np.random.normal(0, 1))
            S[i] = S[i - 1] + r * S[i - 1] * timestep + sigma * S[i - 1] * np.random.normal(0, 1) * np.sqrt(timestep)
        return S
                    
    def compute_call_trajectory(self, K):
        time_to_maturity = self.maturity - self.time_grid
        
        d1 = np.divide(np.log(self.values / K) + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity, (self.sigma * np.sqrt(time_to_maturity)))
        d2 = d1 - self.sigma * np.sqrt(time_to_maturity)
        self.call_values = self.values * norm.cdf(d1) - norm.cdf(d2) * K * np.exp(-self.r * time_to_maturity)       

    def call_intrinsic_curve(self, K, S_min, S_max, t):
        S = np.linspace(start = S_min, stop = S_max, num = 100)
        if t == self.maturity :
            x = (S - K)
            x[x<0]=0
            call_curve = x
        else :
            d1 = (np.log(S / K) + (self.r + 0.5 * self.sigma ** 2) * (self.maturity - t)) / (self.sigma * np.sqrt(self.maturity - t))
            d2 = d1 - self.sigma * np.sqrt(self.maturity - t)
            call_curve = S * norm.cdf(d1) - norm.cdf(d2) * K * np.exp(-self.r * (self.maturity - t)) 
        return pd.Series(call_curve, index = S)

if __name__ == "__main__":
    app = StreamlitBSModel()
    app.run()