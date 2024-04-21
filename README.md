# Crypto

- used variables：
  - Kline variables:
    - Open
    - High
    - Low
    - Close
    - Volume
    - Quote asset volume
    - Taker buy base asset volume
    - Taker buy quote asset volume
  - order book variables:
    - wap1_mean
    - wap2_mean
    - log_return1_mean
    - log_return2_mean
    - volume_imbalance_mean

- inputs：(sequence_length, variables) "sequence_length" seconds, each second contains all the used variables

- outputs：(30) predict up/down of each second in the next 30 seconds
