import pandas as pd
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


def get_df(df):
    df.columns = ['time','bid','bs','ask','as']
    df = df.dropna()
    df['time']=df['time'].astype(float)
    df['bid']=df['bid'].astype(float)
    df['ask']=df['ask'].astype(float)
    df['bs']=df['bs'].astype(float)
    df['as']=df['as'].astype(float)
    df['mid']=(df['bid'].astype(float)+df['ask'].astype(float))/2
    df['imb']=df['bs'].astype(float)/(df['bs'].astype(float)+df['as'].astype(float))
    df['wmid']=df['ask'].astype(float)*df['imb']+df['bid'].astype(float)*(1-df['imb'])
    return df


def prep_data_sym(T,n_imb,dt,n_spread):
    spread = T.ask-T.bid
    ticksize = np.round(min(spread.loc[spread>0])*1000)/1000
    T.spread = T.ask-T.bid
    # adds the spread and mid prices
    T['spread']=np.round((T['ask']-T['bid'])/ticksize)*ticksize
    T['mid']=(T['bid']+T['ask'])/2
    #filter out spreads >= n_spread
    T = T.loc[(T.spread <= n_spread*ticksize) & (T.spread>0)]
    T['imb']=T['bs']/(T['bs']+T['as'])
    #discretize imbalance into percentiles
    T['imb_bucket'] = pd.qcut(T['imb'], n_imb, labels=False)
    T['next_mid']=T['mid'].shift(-dt)
    #step ahead state variables
    T['next_spread']=T['spread'].shift(-dt)
    T['next_time']=T['time'].shift(-dt)
    T['next_imb_bucket']=T['imb_bucket'].shift(-dt)
    # step ahead change in price
    T['dM']=np.round((T['next_mid']-T['mid'])/ticksize*2)*ticksize/2
    T = T.loc[(T.dM <= ticksize*1.1) & (T.dM>=-ticksize*1.1)]
    # symetrize data
    T2 = T.copy(deep=True)
    T2['imb_bucket']=n_imb-1-T2['imb_bucket']
    T2['next_imb_bucket']=n_imb-1-T2['next_imb_bucket']
    T2['dM']=-T2['dM']
    T2['mid']=-T2['mid']
    T3=pd.concat([T,T2])
    T3.index = pd.RangeIndex(len(T3.index))
    return T3,ticksize


def estimate(T):
    no_move = T[T['dM'] == 0]
    no_move_counts = no_move.pivot_table(index=['next_imb_bucket'],
                                         columns=['spread', 'imb_bucket'],
                                         values='time',
                                         fill_value=0,
                                         aggfunc='count').unstack()
    Q_counts = np.resize(np.array(no_move_counts[0:(n_imb * n_imb)]), (n_imb, n_imb))
    # loop over all spreads and add block matrices
    for i in range(1, n_spread):
        Qi = np.resize(np.array(no_move_counts[(i * n_imb * n_imb):(i + 1) * (n_imb * n_imb)]), (n_imb, n_imb))
        Q_counts = block_diag(Q_counts, Qi)
    # print Q_counts
    move_counts = T[(T['dM'] != 0)].pivot_table(index=['dM'],
                                                columns=['spread', 'imb_bucket'],
                                                values='time',
                                                fill_value=0,
                                                aggfunc='count').unstack()

    R_counts = np.resize(np.array(move_counts), (n_imb * n_spread, 4))
    T1 = np.concatenate((Q_counts, R_counts), axis=1).astype(float)
    for i in range(0, n_imb * n_spread):
        T1[i] = T1[i] / T1[i].sum()
    Q = T1[:, 0:(n_imb * n_spread)]
    R1 = T1[:, (n_imb * n_spread):]

    K = np.array([-0.01, -0.005, 0.005, 0.01])
    move_counts = T[(T['dM'] != 0)].pivot_table(index=['spread', 'imb_bucket'],
                                                columns=['next_spread', 'next_imb_bucket'],
                                                values='time',
                                                fill_value=0,
                                                aggfunc='count')  # .unstack()

    R2_counts = np.resize(np.array(move_counts), (n_imb * n_spread, n_imb * n_spread))
    T2 = np.concatenate((Q_counts, R2_counts), axis=1).astype(float)

    for i in range(0, n_imb * n_spread):
        T2[i] = T2[i] / T2[i].sum()
    R2 = T2[:, (n_imb * n_spread):]
    Q2 = T2[:, 0:(n_imb * n_spread)]
    G1 = np.dot(np.dot(np.linalg.inv(np.eye(n_imb * n_spread) - Q), R1), K)
    B = np.dot(np.linalg.inv(np.eye(n_imb * n_spread) - Q), R2)

    return G1, B, Q, Q2, R1, R2, K


def get_Gstar(G1, B):
    G2=np.dot(B,G1)+G1
    G3=G2+np.dot(np.dot(B,B),G1)
    G4=G3+np.dot(np.dot(np.dot(B,B),B),G1)
    G5=G4+np.dot(np.dot(np.dot(np.dot(B,B),B),B),G1)
    G6=G5+np.dot(np.dot(np.dot(np.dot(np.dot(B,B),B),B),B),G1)
    return G6


def calc(G, data, ticksize):
    mid = (data["ask"] + data["bid"]) / 2
    imb = data["bs"] /(data["as"] + data["bs"])
    imb_idx = pd.qcut(imb, n_imb, labels=False)
    spread = np.round((data["ask"] - data["bid"]) / ticksize)
    spread[spread > n_spread] = n_spread
    spread[spread < 1] = 1
    spread_idx = spread - 1

    micro_price = pd.Series(range(len(data)), dtype=np.int64)
    micro_price = micro_price.apply(lambda x: mid[x] + G[int(imb_idx[x] + n_imb * spread_idx[x])])
    return micro_price


def calc_micro_price(df, imbs, spreads, dts):
    global n_imb, n_spread, dt
    n_imb = imbs
    n_spread = spreads
    dt = dts
    data = get_df(df)
    pd.set_option('mode.chained_assignment', None)
    T, ticksize = prep_data_sym(data, n_imb, dt, n_spread)
    G1, B, Q, Q2, R1, R2, K = estimate(T)
    G6 = get_Gstar(G1, B)
    micro_price = calc(G6, data, ticksize)
    return micro_price