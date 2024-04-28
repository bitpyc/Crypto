import time
import dateutil
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from binance.spot import Spot
from micro_price import calc_micro_price
import argparse


def calc_start_time(df, args):
    start_time = args.start_timestamp + ((df["timestamp"] // 1000 - args.start_timestamp) // args.interval) * args.interval
    return start_time.astype(np.int64)


# calculate weighted average price
def calc_wap(df, bid_price, bid_size, ask_price, ask_size, time_col="start_time"):
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


def calc_micro(df, bid_price, bid_size, ask_price, ask_size, time_col="start_time"):
    df = df[[time_col, bid_price, bid_size, ask_price, ask_size]]
    micro_price = calc_micro_price(df, 10, 2, 1)
    return micro_price


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
    df["start_time"] = book_info["start_time"]

    if args.price == "MicroPrice":
        calc_price = calc_micro
    else:
        calc_price = calc_wap
    # calculate top k wap and log return
    for i in range(args.top_k):
        price_name = args.price + str(i)
        log_return_name = "log_return" + str(i)
        book_info[price_name] = calc_price(df, "bids[{}].price".format(i), "bids[{}].amount".format(i), "asks[{}].price".format(i), "asks[{}].amount".format(i))
        book_info[log_return_name] = calc_log_return(book_info, price_name)
    book_info["volume_imbalance"] = calc_volume_imbalance(df, args)

    # df["{}0".format(args.price)] = book_info["{}0".format(args.price)]
    # print(df[["bids[0].price", "bids[0].amount", "{}0".format(args.price), "asks[0].price", "asks[0].amount"]][:20])

    # aggregate the variables which have the same "start_time"
    variables = book_info.columns.values.tolist()
    variables.remove("start_time")
    variable_agg_functions = {v:"mean" for v in variables}
    book_info = book_info[(args.start_timestamp <= book_info["start_time"]) & (book_info["start_time"] < args.end_timestamp)].groupby("start_time").agg(variable_agg_functions)
    book_info = book_info.reindex(range(args.start_timestamp, args.end_timestamp, args.interval), method='pad')
    book_info = book_info.reset_index()
    return book_info


def process_liquid_data(args):

    start_date_utc = datetime.strptime(args.start_date, "%Y-%m-%d") - timedelta(days=1)
    current_day = start_date_utc
    df_list = []
    while current_day < datetime.strptime(args.end_date, "%Y-%m-%d"):
        df_list.append(pd.read_csv(args.liquid_files.format(current_day.strftime("%Y-%m-%d"))))
        current_day = current_day + timedelta(days=1)
    df = pd.concat(df_list, ignore_index=True)

    # target dataframe to be created.
    liquid_info = pd.DataFrame()
    liquid_info["start_time"] = calc_start_time(df, args)
    df["start_time"] = liquid_info["start_time"]
    liquid_info["price"] = df["price"]
    liquid_info["amount"] = df["amount"]
    variable_agg_functions = {"price":"mean", "amount":"sum"}
    liquid_info = liquid_info[(args.start_timestamp <= liquid_info["start_time"]) & (liquid_info["start_time"] < args.end_timestamp)].groupby("start_time").agg(variable_agg_functions)
    liquid_info = liquid_info.reindex(range(args.start_timestamp, args.end_timestamp, args.interval), fill_value=0)
    liquid_info = liquid_info.reset_index()
    liquid_info.columns = ["start_time", "liquid_price", "liquid_amount"]
    return liquid_info


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
    return kline_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--book_files", type=str, default="datasets/binance-futures_book_snapshot_5_{}_SOLUSDT.csv")
    parser.add_argument("--liquid_files", type=str, default="datasets/binance-futures_liquidations_{}_SOLUSDT.csv")
    parser.add_argument("--start_date", type=str, default="2023-12-16")
    parser.add_argument("--end_date", type=str, default="2023-12-17")
    parser.add_argument("--interval", type=int, default=1000, help="interval of timestep, 'ms' based")
    parser.add_argument("--top_k", type=int, default=2, help="order book bid/ask top_k leval selected as variables")
    parser.add_argument("--price", type=str, default="MicroPrice")

    args = parser.parse_args()
    args.start_timestamp = np.int64(datetime.strptime(args.start_date, "%Y-%m-%d").timestamp() * 1000)
    args.end_timestamp = np.int64(datetime.strptime(args.end_date, "%Y-%m-%d").timestamp() * 1000)

    kline_info = process_kline_data(args)
    liquid_info = process_liquid_data(args)
    book_info = process_book_data(args)
    solana_dataset = pd.merge(kline_info, book_info, on="start_time")
    solana_dataset = pd.merge(solana_dataset, liquid_info, on="start_time")
    solana_dataset.describe()

    col_missing = solana_dataset.isnull().sum()
    print("missing data:")
    print(col_missing)

    solana_dataset.to_csv("datasets/top{}_{}_{}_to_{}_local.csv".format(args.top_k, args.price, "2023-12-16", "2023-12-17"), index=False)