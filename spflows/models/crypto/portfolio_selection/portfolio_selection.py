import torch
import numpy as np


def ml_estimates_black_scholes_from_predictors(output):
    batch_size = output.shape[0]
    seq_lenght = output.shape[1]
    steps_ahead = output.shape[2]
    dimension = output.shape[3]
    prices = output.reshape(batch_size*seq_lenght,steps_ahead,dimension)
    prices = prices[:,:,0]

    X = torch.log(prices)
    log_initial_prices = torch.log(prices[:,0]).detach()
    log_final_prices = torch.log(prices[:,-1]).detach()

    dX = log_final_prices - log_initial_prices
    DX = X[:,1:] - X[:,:-1]
    DX = DX**2.
    DX[DX != DX] = 0.
    DX[DX == np.inf] = 0.
    DX[DX == -np.inf] = 0.
    DX = DX.sum(axis=1)

    sigma_square_ml = DX/steps_ahead - (dX**2)/steps_ahead**2
    mu_ml = DX/steps_ahead + 0.5*sigma_square_ml

    mu_ml = mu_ml.reshape(batch_size,seq_lenght)
    sigma_square_ml = sigma_square_ml.reshape(batch_size,seq_lenght)
    return mu_ml, sigma_square_ml

def birth_and_death_indices(prices_below):
    portfolio_size = prices_below.shape[0]

    where_not_zero = (prices_below != 0.).float()
    column_index = torch.arange(0, prices_below.shape[1], 1).long().unsqueeze(0)
    column_index = column_index.repeat(portfolio_size, 1)
    birth_index = where_not_zero * column_index + (1. - where_not_zero) * (prices_below.shape[1] + 1)
    birth_index = birth_index.long()
    birth_index = birth_index.min(axis=1).values

    death_index = where_not_zero * column_index
    death_index = death_index.long()
    death_index = death_index.max(axis=1).values

    return birth_index, death_index

def equally_weighted_portfolio(prices_below,prices_ahead):
    """
    :param data_loader:
    :return: policy [portfolio_size,number_of_steps,]
    """
    portfolio_size = prices_below.shape[0]
    pi_N = 1. / portfolio_size
    policy = torch.full_like(prices_below, pi_N)
    return policy