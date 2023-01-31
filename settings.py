import random
import time


CSV_DATA_PATH = 'btcusdt_1294651_1m_candles.csv'
DATA_SIZE = 20000
N_STEPS = 70
LOOKUP_STEP = 1
SCALE = True
SHUFFLE = True
TEST_SIZE = 0.2
FEATURE_COLUMNS = ["close", "volume", "open", "high", "low"]
N_LAYERS = 2
UNITS = 256
DROPOUT = 0.4
BIDIRECTIONAL = False
LOSS = "huber_loss"
OPTIMIZER = "adam"
BATCH_SIZE = 64
EPOCHS = 222
SESSION_ID = [i for i in
              "PyVBUkDEKMxcwGIoJOuiTXgenlFrSYWdqsZzjHRhtvfmCLbNapAQ"[random.randrange(0, 51)] +
              "SmauEfRHijrtZnoqwsVOYxIQAMlTgydbCWXeNzhUKFPpDGkJBcvL"[random.randrange(0, 51)] +
              "lvnobfOiPBFMZJHyLpWRkGEQsYNwgDdUtuqIejCzKrXATVSxahmc"[random.randrange(0, 51)] +
              "uXwkfdijHJhaZAxzCUFEmeNOynLSQtlPqBGRTKVvIosWYMDpcgbr"[random.randrange(0, 51)] +
              "GWfbgTuRCocvSwmUiesMQLhXYIKElAVkOJFHdyxzDPqantZrjBNp"[random.randrange(0, 51)] +
              "LzlSHVywOEeIpBQmoRXWFNkPfvhZKgjGTsaUJxMcrbiqnutYDCAd"[random.randrange(0, 51)] +
              "rWiklbmBQHROVdqtJxuMwcahzGsSpefDNYZjEgUKXTIAovLPFCny"[random.randrange(0, 51)] +
              "MtIlFwnvEcjQmPrdSgJxhyKZsbGoLfVzRAWkTCuOiUYXeDaBNHpq"[random.randrange(0, 51)] +
              str(random.randrange(1000000000, 9999999999))]
for i in range(random.randrange(2, 10)):
    random.shuffle(SESSION_ID)
SESSION_ID = "".join(SESSION_ID)
MODEL_NAME = f"datetime={time.strftime('%Y-%m-%d')}-SESSION_ID={SESSION_ID}-CSV_DATA_PATH=" \
             f"{CSV_DATA_PATH.split('.')[0]}-DATA_SIZE={DATA_SIZE}-N_STEPS={N_STEPS}-LOOKUP_STEP={LOOKUP_STEP}-" \
             f"SCALE={SCALE}-SHUFFLE={SHUFFLE}-N_LAYERS={N_LAYERS}-UNITS={UNITS}-BIDIRECTIONAL={BIDIRECTIONAL}-LOSS=" \
             f"{LOSS}-OPTIMIZER={OPTIMIZER}-BATCH_SIZE={BATCH_SIZE}-EPOCHS={EPOCHS}"
