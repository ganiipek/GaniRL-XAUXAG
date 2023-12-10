import numpy as np
import pandas as pd
import itertools
from environments.loader_xauxag import Loader
from enum import Enum
from collections import deque


class Action(Enum):
    HOLD = 0
    BUY_SYMBOL1_SELL_SYMBOL2 = 1
    SELL_SYMBOL1_BUY_SYMBOL2 = 2
    CLOSE = 3

class Symbol:
    def __init__(self, name, digits, contract_size, volume_min=0.01, volume_step=0.01, commission_per_100k_volume=0, leverage=100):
        self.name = name
        self.digits = digits
        self.contract_size = contract_size
        self.volume_min = volume_min
        self.volume_step = volume_step
        self.commission_per_100k_volume = commission_per_100k_volume
        self.leverage = leverage
        self.price = None

class PositionType(Enum):
    BUY = 0
    SELL = 1

class Position:
    id_iter = itertools.count()
    def __init__(self, _type:PositionType, symbol:Symbol, volume, open_price, open_time, commission=0):
        self.id = next(self.id_iter)
        self.type = _type
        self.symbol = symbol
        self.volume = volume
        self.open_price = open_price
        self.open_time = open_time
        self.commission = commission
        self.close_price = None
        self.close_time = None
        self.profit = None
        self.total_profit = None

    def get_profit(self):
        if self.close_time is None:
            self.close_price = self.symbol.price

        if self.type == PositionType.BUY:
            self.profit = (self.close_price - self.open_price) * self.volume * self.symbol.contract_size
        elif self.type == PositionType.SELL:
            self.profit = (self.open_price - self.close_price) * self.volume * self.symbol.contract_size

        return self.profit

    def get_total_profit(self):
        self.total_profit = self.get_profit() - self.commission
        return self.total_profit

    def close(self, close_price, close_time):
        self.close_price = close_price
        self.close_time = close_time

        self.profit = self.get_profit()
        self.total_profit = self.get_total_profit()

class Transaction:
    id_iter = itertools.count()
    def __init__(self, verbose=False):
        self.verbose = verbose

        self.id = next(self.id_iter)
        self.action = None
        self.position1 = None
        self.position2 = None
        self.open_ratio = None
        self.open_time = None
        self.close_ratio = None
        self.close_time = None
        self.profit = None
        self.total_profit = None
        self.ratio_diff = None

    def calc_volume(self, margin, symbol:Symbol):
        volume = margin / symbol.price / symbol.contract_size * symbol.leverage
        volume = np.round(volume, 2)
        volume = np.floor(volume / symbol.volume_step) * symbol.volume_step
        volume = np.max([volume, symbol.volume_min])
        return volume

    def calc_commission(self, margin, symbol:Symbol):
        commission = ((margin * symbol.leverage / 100000) * symbol.commission_per_100k_volume) * 2 # DEAL IN/OUT
        commission = np.round(commission, 2)
        return commission

    def open(self, action, margin, symbol1:Symbol, symbol2:Symbol, ratio, time):
        self.action = action

        symbol1_volume = self.calc_volume(margin/2, symbol1)
        symbol2_volume = self.calc_volume(margin/2, symbol2)

        symbol1_commission = self.calc_commission(margin, symbol1)
        symbol2_commission = self.calc_commission(margin, symbol2)

        if action == Action.BUY_SYMBOL1_SELL_SYMBOL2.value:
            position1 = Position(PositionType.BUY, symbol1, symbol1_volume, symbol1.price, time, symbol1_commission)
            position2 = Position(PositionType.SELL, symbol2, symbol2_volume, symbol2.price, time, symbol2_commission)
        elif action == Action.SELL_SYMBOL1_BUY_SYMBOL2.value:
            position1 = Position(PositionType.SELL, symbol1, symbol1_volume, symbol1.price, time, symbol1_commission)
            position2 = Position(PositionType.BUY, symbol2, symbol2_volume, symbol2.price, time, symbol2_commission)
        else:
            raise Exception("Invalid action")

        self.position1 = position1
        self.position2 = position2
        self.open_ratio = ratio
        self.open_time = time

        if self.verbose:
            print(f"")
            print(f"[Open] Transaction #{self.id} - {self.action}")
            print(f"Position1: {self.position1.type.name}, {self.position1.volume} {self.position1.symbol.name}, {self.position1.open_price}, {self.position1.open_time}")
            print(f"Position2: {self.position2.type.name}, {self.position2.volume} {self.position2.symbol.name}, {self.position2.open_price}, {self.position2.open_time}")
            print(f"Open ratio: {self.open_ratio}")
            print(f"Open time: {self.open_time}")
            print(f"")

    def get_profit(self):
        self.profit = self.position1.get_profit() + self.position2.get_profit()
        return self.profit

    def get_total_profit(self):
        self.total_profit = self.position1.get_total_profit() + self.position2.get_total_profit()
        return self.total_profit

    def close(self, position1_price, position2_price, close_ratio, close_time):
        self.close_ratio = close_ratio
        self.close_time = close_time
        self.position1.close(position1_price, close_time)
        self.position2.close(position2_price, close_time)

        self.profit = self.get_profit()
        self.total_profit = self.get_total_profit()
        self.ratio_diff = self.close_ratio - self.open_ratio

        if self.verbose:
            print(f"")
            print(f"[Close] Transaction #{self.id} - {self.action}")
            print(f"Position1: {self.position1.type.name}, {self.position1.volume} {self.position1.symbol.name}, {self.position1.close_price}, {self.position1.close_time} | Total Profit: {self.position1.get_total_profit()}")
            print(f"Position2: {self.position2.type.name}, {self.position2.volume} {self.position2.symbol.name}, {self.position2.close_price}, {self.position2.close_time} | Total Profit: {self.position2.get_total_profit()}")
            print(f"Close ratio: {self.close_ratio}")
            print(f"Close time: {self.close_time}")
            print(f"Profit: {self.profit}")
            print(f"Total profit: {self.total_profit}")
            print(f"Ratio diff: {self.ratio_diff}")
            print(f"")

class RatioEnv:
    def __init__(
            self, data,
            initial_balance=100000, commission_per_100k_volume=3,
            lookback_window_size=50, verbose=False
        ):
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.verbose=verbose

        self.historical_data = data

        self.symbol1 = Symbol(
            name="XAUUSD", digits=2, contract_size=100, volume_min=0.01, volume_step=0.01, commission_per_100k_volume=commission_per_100k_volume
        )
        self.symbol2 = Symbol(
            name="XAGUSD", digits=2, contract_size=5000, volume_min=0.01, volume_step=0.01, commission_per_100k_volume=commission_per_100k_volume
        )

        self.current_step = lookback_window_size
        self.current_balance = initial_balance
        self.current_equity = initial_balance
        self.current_margin = 0
        self.current_transaction = None

        self.transaction_history = []
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.total_rewards = 0
        self.max_drawdown = 0
        self.punish_value = 0

        self.hold_count = 0

    def update_market_history(self, data):
        # self.market_history = pd.concat([self.market_history, data], axis=1)
        pass

    def update_transaction_history(self, transaction:Transaction):
        self.transaction_history.append(transaction)

    def state_shape(self):
        # Observation boyutunu belirtir
        # obs = np.concatenate([
        #     self.trade_histroy.tail(self.lookback_window_size).to_numpy(),
        #     self.market_history.tail(self.lookback_window_size).to_numpy()
        # ], axis=1)
        # return len(self.historical_data.columns),
        return self.lookback_window_size,

    def action_shape(self):
        # Action boyutunu belirtir
        # return len([e.value for e in Action]),
        return 1,

    def reset(self):
        self.current_balance = self.initial_balance
        self.current_equity = self.initial_balance
        self.current_step = self.lookback_window_size
        self.current_transaction = None
        
        self.transaction_history = []
        self.market_history = deque(maxlen=self.lookback_window_size)
        for index, row in self.historical_data[self.current_step-self.lookback_window_size : self.current_step].iterrows():
            self.market_history.append([
                row["XAUUSD"],
                row["XAGUSD"],
                row["ratio"]
            ])

        self.max_drawdown = 0
        self.punish_value = 0

        self.hold_count = 0
        self.total_rewards = 0

        return self.next_observation()

    def is_finished(self):
        if self.current_step >= self.historical_data.index.size - 1:
            return True
        elif self.current_equity < self.initial_balance * 0.1:
            return True

        return False
    
    def get_intervals(self, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
        index = self.historical_data.index

        size = len(index)
        train_begin = 0
        train_end = int(np.round(train_ratio * size - 1))
        valid_begin = train_end + 1
        valid_end = valid_begin + int(np.round(valid_ratio * size - 1))
        test_begin = valid_end + 1
        test_end = -1
        
        intervals = {'training': (index[train_begin], index[train_end]),
             'validation': (index[valid_begin], index[valid_end]),
             'testing': (index[test_begin], index[test_end])}

        return intervals
        
    def calculate_reward(self, action):
        # Ajanın doğru pozisyonu alıp almadığını kontrol et
        reward = 0
        current_ratio = self.historical_data["ratio"][self.current_step]
        previous_ratio = self.historical_data["ratio"][self.current_step - 1]

        if self.current_transaction is not None:
            self.current_equity = self.current_balance + self.current_transaction.get_total_profit() 

            return_perc = (self.current_equity - self.current_balance) / self.current_balance * 100

            if return_perc < -0.1:
                reward = return_perc
        else:
            reward += -0.01 * self.hold_count // 10




        # Karlılık kontrolü
        # ... (ticaret karlıysa reward artır)

        # Risk kontrolü
        # ... (aşırı risk durumunda negatif reward)

        # İşlem frekans kontrolü
        # ... (çok sık veya çok nadir işlem durumunda negatif reward)

        return reward
        # return (self.current_equity - self.current_balance) / self.current_balance * 100
        # return np.random.randint(-20, 20, size=1)[0]

    def close_trade(self, action, ratio, time):
        self.punish_value += self.current_equity * 0.00001
        self.punish_value = 0
        
        if self.current_transaction is not None:
            if self.current_transaction.action == Action.SELL_SYMBOL1_BUY_SYMBOL2.value:
                self.current_transaction.close(self.symbol1.price, self.symbol2.price, ratio, time)
                self.update_transaction_history(self.current_transaction)

                self.current_balance += self.current_transaction.get_total_profit()
                self.current_equity = self.current_balance

                reward = self.current_transaction.get_total_profit()
                reward -= self.punish_value
                self.punish_value = 0

                self.current_transaction = None
                return reward
        
            elif self.current_transaction.action == Action.BUY_SYMBOL1_SELL_SYMBOL2.value:
                self.current_transaction.close(self.symbol1.price, self.symbol2.price, ratio, time)
                self.update_transaction_history(self.current_transaction)

                self.current_balance += self.current_transaction.get_total_profit()
                self.current_equity = self.current_balance
                
                reward = self.current_transaction.get_total_profit()
                reward -= self.punish_value
                self.punish_value = 0

                self.current_transaction = None
                return reward
        else:
            return 0 - self.punish_value
        

    def evaluate_trade(self, action, ratio, time):
        # Ticaret sonuçlarını değerlendir
        reward = 0
        margin = self.current_balance * 0.01

        if action == Action.CLOSE.value:
            if self.current_transaction is not None:
                reward = self.close_trade(action, ratio, time)

        if action == Action.BUY_SYMBOL1_SELL_SYMBOL2.value:
            if self.current_transaction is not None:
                if self.current_transaction.action is not Action.BUY_SYMBOL1_SELL_SYMBOL2.value:
                    reward = self.close_trade(action, ratio, time)

            if self.current_transaction is None:   
                transaction = Transaction(verbose=self.verbose)
                transaction.open(action, margin, self.symbol1, self.symbol2, ratio, time)
                self.current_transaction = transaction

        elif action == Action.SELL_SYMBOL1_BUY_SYMBOL2.value:
            if self.current_transaction is not None:
                if self.current_transaction.action is not Action.SELL_SYMBOL1_BUY_SYMBOL2.value:
                    reward = self.close_trade(action, ratio, time)

            if self.current_transaction is None:   
                transaction = Transaction(verbose=self.verbose)
                transaction.open(action, margin, self.symbol1, self.symbol2, ratio, time)
                self.current_transaction = transaction

        elif action == Action.HOLD.value:
            # reward = self.close_trade(action, ratio, time)
            pass

        if self.current_transaction is not None:
            self.current_equity = self.current_balance + self.current_transaction.get_total_profit()

        return reward

    def next_observation(self):
        # Gözlemi güncelle
        # self.update_market_history(self.historical_data.at[self.current_step])
        self.market_history.append([
            self.historical_data.at[self.historical_data.index[self.current_step], "XAUUSD"],
            self.historical_data.at[self.historical_data.index[self.current_step], "XAGUSD"],
            self.historical_data.at[self.historical_data.index[self.current_step], "ratio"]
        ])
        # data = self.historical_data[self.current_step-self.lookback_window_size : self.current_step]["ratio"]
        # observation = np.array(data).T
        obs = np.array(self.market_history)
        return obs

    def step(self, action:int):
        self.current_step += 1

        # Ajanın doğru pozisyonu alıp almadığını kontrol et
        current_ratio = self.historical_data.at[self.historical_data.index[self.current_step], "ratio"]
        previous_ratio = self.historical_data.at[self.historical_data.index[self.current_step - 1], "ratio"]

        self.symbol1.price = self.historical_data.at[self.historical_data.index[self.current_step], "XAUUSD"]
        self.symbol2.price = self.historical_data.at[self.historical_data.index[self.current_step], "XAGUSD"]
        
        # 1. İşlemi yap
        reward = self.evaluate_trade(action, current_ratio, self.historical_data.index[self.current_step])
        self.total_rewards += reward

        # 3. Reward hesapla
        if action == 0:
            self.hold_count += 1
        else: 
            self.hold_count = 0

        
        # 4. Observation hesapla    
        observation = self.next_observation()
        # 5. Done hesapla
        done = self.is_finished()
        # 6. Info hesapla
        info = {
            "timestamp": self.historical_data.index[self.current_step].timestamp(),
            "datetime": self.historical_data.index[self.current_step],
            "current_balance": self.current_balance, 
            "current_equity": self.current_equity,
            "current_margin": self.current_margin,
            "total_rewards": self.total_rewards,
            "transaction_count": len(self.transaction_history)
            }

        if self.verbose:
            print(f"Date: {info['datetime']},\tBalance: {int(info['current_balance'])},\tEquity: {int(info['current_equity'])}")
            print(f"Action: {action},\tReward: {reward},\tTotal Reward: {self.total_rewards}")
            print("-----------------\n")

        return observation, reward, done, info

    