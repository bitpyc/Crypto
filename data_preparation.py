import time
import dateutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.spot import Spot
import argparse


def calc_start_time(df, args):
    start_time = args.start_timestamp + ((df["timestamp"] // 1000 - args.start_timestamp) // args.interval) * args.interval
    return start_time.astype(np.int64)


# calculate weighted average price
def calc_wap(df, bid_price, bid_size, ask_price, ask_size):
    wap = (df[bid_price] * df[ask_size] + df[ask_price] * df[bid_size]) / (df[bid_size] + df[ask_size])
    return wap


# calculate log return for wap series
def calc_log_return(df, wap):
    log_return = np.log(df[wap]).diff()
    return log_return


def calc_volume_imbalance(df, args):
    top_k = args.top_k
    bids = ["bids[{}].amount".format(i) for i in range(top_k)]
    ask = ["asks[{}].amount".format(i) for i in range(top_k)]
    volume_imbalance = abs(df[bids].sum(axis=1) - df[ask].sum(axis=1))
    return volume_imbalance


def process_book_data(args):

    start_date_utc = datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=1)
    current_day = start_date_utc
    df_list = []
    while current_day < datetime.strptime(args.end_date, "%Y-%m-%d"):
        df_list.append(pd.read_csv(args.book_files.format(current_day.strftime("%Y-%m-%d"))))
        current_day = current_day + timedelta(days=1)
    df = pd.concat(df_list, ignore_index=True)

    # target dataframe to be created.
    book_info = pd.DataFrame()
    book_info["start_time"] = calc_start_time(df, args)

    '''
    print("start:")
    print(datetime.fromtimestamp(book_info["start_time"].values[0] // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    print(datetime.fromtimestamp(args.start_timestamp // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    print("end:")
    print(datetime.fromtimestamp(book_info["start_time"].values[-1] // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    print(datetime.fromtimestamp(args.end_timestamp // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    '''

    # calculate top k wap and log return
    for i in range(args.top_k):
        wap_name = "wap" + str(i)
        log_return_name = "log_return" + str(i)
        book_info[wap_name] = calc_wap(df, "bids[{}].price".format(i), "bids[{}].amount".format(i), "asks[{}].price".format(i), "asks[{}].amount".format(i))
        book_info[log_return_name] = calc_log_return(book_info, wap_name)
    book_info["volume_imbalance"] = calc_volume_imbalance(df, args)

    # aggregate the variables witch have the same "start_time"
    variables = book_info.columns.values.tolist()
    variables.remove("start_time")
    variable_agg_functions = {v:"mean" for v in variables}
    book_info = book_info[(args.start_timestamp <= book_info["start_time"]) & (book_info["start_time"] < args.end_timestamp)].groupby("start_time").agg(variable_agg_functions)
    '''
    print("\nbefore_reindex:")
    print("length={}".format(len(book_info)))
    print((book_info.index[1:] - book_info.index[:-1]).value_counts())
    '''
    book_info = book_info.reindex(range(args.start_timestamp, args.end_timestamp, args.interval), method='pad')
    book_info = book_info.reset_index()
    '''
    print("\nafter_reindex:")
    print("length={}".format(len(book_info)))
    print(book_info["start_time"].diff().value_counts())
    print(datetime.fromtimestamp(book_info["start_time"].values[-1] // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    '''
    return book_info


def process_kline_data(args):
    # the meaning of each kline data accessed from binance spot
    kline_description = [
        "start_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "end_time",
        "quote_v",
        "trades",
        "taker_base_v",
        "taker_quote_v",
        "unused"
    ]
    client = Spot()
    print(client.time())
    kline_info_list = []

    # binance api max limit is 500 data per request. need to access full data in an increment way
    increment_num = 500 * args.interval
    for st in range(args.start_timestamp, args.end_timestamp, increment_num):
        ed = min(st + increment_num, args.end_timestamp - args.interval)
        kline_info_list.extend(client.klines(symbol="SOLUSDT", interval="1s", startTime=st, endTime=ed))

    kline_info = pd.DataFrame(data=kline_info_list, columns=kline_description)
    kline_info = kline_info.drop(labels=["end_time", "unused"], axis=1)
    '''
    print(kline_info["start_time"].diff().value_counts())
    print(datetime.fromtimestamp(kline_info["start_time"].values[0] // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    print(datetime.fromtimestamp(kline_info["start_time"].values[-1] // 1000).strftime("%Y-%m-%d %H:%M:%S"))
    '''
    return kline_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_files", type=str, default="datasets/binance-futures_book_snapshot_5_{}_SOLUSDT.csv")
    parser.add_argument("--start_date", type=str, default="2023-12-16")
    parser.add_argument("--end_date", type=str, default="2023-12-17")
    parser.add_argument("--interval", type=int, default=1000, help="interval of timestep, 'ms' based")
    parser.add_argument("--top_k", type=int, default=2, help="order book bid/ask top_k leval selected as variables")
    args = parser.parse_args()
    args.start_timestamp = np.int64(datetime.strptime(args.start_date, "%Y-%m-%d").timestamp() * 1000)
    args.end_timestamp = np.int64(datetime.strptime(args.end_date, "%Y-%m-%d").timestamp() * 1000)

    book_info = process_book_data(args)

    kline_info = process_kline_data(args)

    solana_dataset = pd.merge(kline_info, book_info, on="start_time")

    solana_dataset.describe()
    print(solana_dataset.columns.values)
    print(solana_dataset["start_time"].diff().value_counts())

    col_missing = solana_dataset.isnull().sum()

    print(col_missing)

    solana_dataset.to_csv("datasets/top{}_{}_to_{}_local.csv".format(args.top_k, "2023-12-16", "2023-12-17"), index=False)