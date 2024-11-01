import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
import statsmodels.api as sm
import optuna
import optuna.logging
optuna.logging.set_verbosity(optuna.logging.WARNING)
import quantstats

def calculate_var(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = sorted_returns[index]
    return var

def calculate_cvar(returns, var):
    cvar = returns[returns <= var].mean()
    return cvar

def loss_function(ret,weight,gamma,lamb,theta,lo = 0):
    final_ret = (ret * weight).sum(axis=1)
    cum_ret_ser = (1+final_ret).cumprod()
    drawdown = max((np.maximum.accumulate(cum_ret_ser) - cum_ret_ser) / np.maximum.accumulate(cum_ret_ser))
    dd = (np.maximum.accumulate(cum_ret_ser) - cum_ret_ser) / np.maximum.accumulate(cum_ret_ser)
    # 计算rss
    X = cum_ret_ser.reset_index().index
    y = cum_ret_ser
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    residuals = model.resid
    rss = (residuals ** 2).sum()
    slope = model.params[1]
    if lo==1:
        loss = -slope*cum_ret_ser.iloc[-1]/drawdown + lamb * np.linalg.norm(theta,1)
    else:
        loss = -slope*cum_ret_ser.iloc[-1]/drawdown
    return loss

def train(ret,factor_lis,gamma,lamb):
    def objective(trial):
        theta = [trial.suggest_float(f'theta_{i}', 1e-5, 1e-2) for i in range(1)]
        the_w = []
        for i in range(0,len(theta)):
            the_w.append(theta[i] * factor_lis[i])
        n_w = sum(the_w)
        n_w = (n_w.T/n_w.sum(axis=1)).T
        cal_w = n_w +  the_w[0].notna().astype(int).div(the_w[0].notna().astype(int).sum(axis=1), axis=0)
        norm_w = (cal_w.T / cal_w.abs().sum(axis=1)).T
        loss = loss_function(ret,norm_w,gamma,lamb,theta)
        return loss
    Pop_size = 10
    Tmax = 10
    dim = 20  # Number of hyperparameters to optimize
    lb = [1e-5]*20  # Lower bounds of hyperparameters
    ub = [1e-2]*20  # Upper bounds of hyperparameters
    study = optuna.create_study(direction='minimize')  # Or 'maximize' for maximization
    study.optimize(objective, n_trials=100)
    theta = study.best_params
    #_, theta, _ = CPO(Pop_size, Tmax, lb, ub, dim, fobj,factor_lis)
    return list(theta.values())

def calc_theta(ret_v,factor_lis_v,ret_t,factor_lis_t,gamma,lamb_lis):
    ori_loss = 1e10
    best_lamb = 0
    for lamb in lamb_lis:
        theta = train(ret_t,factor_lis_t,gamma,lamb)
        the_w_v = []
        for i in range(0,len(theta)):
            the_w_v.append(theta[i] * factor_lis_v[i])
        n_w = sum(the_w_v)
        n_w = (n_w.T/n_w.sum(axis=1)).T
        cal_w = n_w + the_w_v[0].notna().astype(int).div(the_w_v[0].notna().astype(int).sum(axis=1), axis=0)
        weight_v = (cal_w.T / cal_w.abs().sum(axis=1)).T
        loss = loss_function(ret_v,weight_v,gamma,lamb,theta,lo=0)
        if loss < ori_loss:
            ori_loss = loss
            best_lamb = lamb
    return train(ret_t,factor_lis_t,gamma,best_lamb)

def calc_weight(ret_p,factor_lis_p,theta):
    the_w = []
    for i in range(len(theta)):
        the_w.append(theta[i] * factor_lis_p[i])
    n_w = sum(the_w)
    n_w = (n_w.T/n_w.sum(axis=1)).T
    cal_w = n_w + the_w[0].notna().astype(int).div(the_w[0].notna().astype(int).sum(axis=1), axis=0)
    weight = (cal_w.T / cal_w.abs().sum(axis=1)).T
    return weight

def pivot_tran(factor):
    factor_lis = []
    for i in factor.columns[1:-1]:
        fac = factor.pivot_table(index='date',columns='permno',values=i)
        factor_lis.append(fac)
    return factor_lis

def rolling_opt(ret,factor,gamma,lamb_lis):
    trw = 5
    vaw = 2
    tew = 1
    weight = pd.DataFrame()
    factor_lis = pivot_tran(factor)
    theta_lis = []
    for i in tqdm(range(0,len(ret)-(trw+vaw+tew)*12+13,12)):
        ret_t,factor_lis_t = ret.iloc[i:i+12*trw,:], [factor_lis[j].iloc[i:i+12*trw,:] for j in range(len(factor_lis))]
        ret_v, factor_lis_v = ret.iloc[i+12*trw:i+12*(trw+vaw),:], [factor_lis[j].iloc[i+12*trw:i+12*(trw+vaw),:] for j in range(len(factor_lis))]
        ret_p, factor_lis_p = ret.iloc[i+12*(trw+vaw):i+12*(trw+vaw+1),:],[factor_lis[j].iloc[i+12*(trw+vaw):i+12*(trw+vaw+1),:] for j in range(len(factor_lis))]
        theta = calc_theta(ret_v,factor_lis_v,ret_t,factor_lis_t,gamma,lamb_lis)
        testw = calc_weight(ret_p,factor_lis_p,theta)
        weight = pd.concat([weight,testw])
    return weight

def get_all_data(config):
    ret = pd.read_parquet(config['ret_file']).loc['2010':]
    factor = pd.read_parquet(config['factor_file'])
    factor['date'] = factor.index
    factor = factor.reset_index(drop=True)
    return ret, factor

def calc_pnl(weight,ret):
    pnl = (1+(weight*(ret.shift(-1).loc['2017':])).sum(axis=1)).cumprod()
    bench = (1+(ret.shift(-1).loc['2017':]).mean(axis=1)).cumprod()
    excess = (1+(weight*(ret.shift(-1).loc['2017':])).sum(axis=1) - (ret.shift(-1).loc['2017':]).mean(axis=1)).cumprod()
    p = pd.DataFrame(pnl)#.reset_index()
    p.columns = ['pnl']
    p['pnl'] = p['pnl'].shift(1).fillna(1)
    p['base'] = bench
    p['base'] = p['base'].shift(1).fillna(1)
    p['excess'] = excess
    p['excess'] = p['excess'].shift(1).fillna(1)
    p[['pnl','base']].plot(figsize=(20,5))
    return p

def get_quantstat_metric(pnl):
    stret_stats = quantstats.reports.metrics(pnl['pnl'],periods_per_year=12,mode='base', display=False)
    stret_stats.columns = ['PO']
    baret_stat = quantstats.reports.metrics(pnl['base'],periods_per_year=12,mode='base', display=False)
    baret_stat.columns = ['BASE']
    df_concat = pd.concat([stret_stats, baret_stat], axis=1)
    return df_concat
