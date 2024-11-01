from config import config
from utils import *
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']

def main():
    ret, factor = get_all_data(config)
    weight = rolling_opt(ret.shift(-1), factor, config['gamma'], config['lamb_lis'])
    pnl = calc_pnl(weight, ret)
    df_concat = get_quantstat_metric(pnl)
    print(df_concat)

if __name__ == "__main__":
    main()