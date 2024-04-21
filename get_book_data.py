from tardis_dev import datasets
from datetime import datetime
import pytz
import gzip
import os


# the date here is in utc zone!
def get_book_data(from_date, to_date):
    datasets.download(
        exchange="binance-futures",
        data_types=[
            "incremental_book_L2",
        ],
        from_date=from_date,
        to_date=to_date,
        symbols=["SOLUSDT"],
        api_key=os.environ.get("CRYPTO_KEY"),
    )


def decompress_gz(file_name):
    target_name = file_name.replace(".gz", "")
    with open(target_name, "wb") as f:
        with gzip.open(file_name, "rb") as g:
            f.write(g.read())


if __name__ == '__main__':
    get_book_data("2023-12-17", "2024-04-21")
    # decompress_gz("datasets/binance-futures_book_snapshot_5_2023-12-15_SOLUSDT.csv.gz")