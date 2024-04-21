# Crypto

- crypto可能用于构造的时间特征：
  - Kline: 按照时间段划分好的数据
    - Open
    - High
    - Low
    - Close
    - Volume
    - Quote asset volume:报价资产交易总量
    - Taker buy base asset volume:加密货币资产通过Taker角色（以当前市场价格执行交易）的购买量
    - Taker buy quote asset volume:报价资产通过Taker角色的购买量
  - order book:流式数据。需要将一个时间段内的所有相应信息进行融合。这里采用mean的方式，因此以下所有因子后缀都有mean。
    - wap1_mean:按照如下公式计算。1下标表示最佳，对于买价而言是最高，对于卖价而言是最低。
[图片]
    - wap2_mean:上面公式的1（最佳）变为2（次优）
    - log_return1_mean：下为log return计算公式。这里公式中的S用wap1表示
[图片]
    - log_return2_mean: log return公式中的S用wap2表示
    - volume_imbalance_mean:(ask_size1 + ask_size2) - (bid_size1 + bid_size2)
