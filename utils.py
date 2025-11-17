import numpy as np
import tensorflow as tf
import json
import pandas as pd
from collections import deque
import collections
import random
import csv
from collections import defaultdict


def generate_data(name, length, data, value):
    if name in ['history', 'previous_offer', 'previous_trade', 'values']:
        
        data = tf.where(tf.equal(data, 18401), -3.0, data)
        data = tf.where(tf.equal(data, 18402), 311.0, data)
        
        if name == 'values':
            data = data / 150.0
        
        current_length = tf.shape(data)[0]
        data = tf.cond(
            current_length < length,
            true_fn=lambda: (
                
                tf.pad(data, [[0, length - current_length]], constant_values=-3.0)
                if name == 'values'
                
                else tf.pad(data, [[length - current_length, 0]], constant_values=-3.0)
            ),
            false_fn=lambda: data[-length:]  
        )
        data = tf.expand_dims(data, axis=0)

    elif name in ['bids', 'asks']:
        
        key_list = tf.convert_to_tensor(list(data.values()), dtype=tf.float32)  
        key_list = tf.sort(key_list)  
       
        current_length = tf.shape(key_list)[0]
        if current_length < length:
            padding_shape = length - current_length
            padding = tf.fill([padding_shape], -3.0)
            key_list = tf.concat([key_list, padding], axis=0)
        elif current_length > length:
            if name == 'bids':
                key_list = key_list[-length:]  
            else:
                key_list = key_list[:length]  
        data = key_list
        data = tf.expand_dims(data, axis=0)
    else:
        if name == 'value':
            data = data / 150.0  

        elif name == 'value_ratio':
            
            if not data.values():
                data = tf.constant(0.0)  
            else:
                
                key_list = tf.sort(tf.convert_to_tensor(list(data.values()), dtype=tf.float32))
                data = key_list[0] / value

            
        elif name == 'risk_level':
            
            if not data.values():
                data = tf.constant(0.0)  
            else:
                
                key_list = tf.sort(tf.convert_to_tensor(list(data.values()), dtype=tf.float32))
                min_value = key_list[0]
                
                data = tf.case([
                    (tf.greater(min_value, value * 1.5), lambda: tf.constant(2.0)),
                    (tf.greater(min_value, value), lambda: tf.constant(1.0))
                ], default=lambda: tf.constant(0.0))
        data = tf.reshape(data, [1])
    
    mask = tf.equal(data, -3)
    data_float = tf.cast(data, tf.float32)
    data_float = tf.where(mask, tf.constant(np.nan, dtype=tf.float32), data_float)

    if name not in ['unit', 'time', 'role', 'value', 'values', 'value_ratio', 'risk_level']:
        
        mean = tf.convert_to_tensor(np.load('saved_model/' + name + '_mean.npy'), dtype=tf.float32)
        std = tf.convert_to_tensor(np.load('saved_model/' + name + '_std.npy'), dtype=tf.float32)
        data_normalized = (data_float - mean) / std
        data_normalized = tf.where(mask, -3.0, data_normalized)
    else:
        data_normalized = data_float

    return data_normalized


def generate_data1(name,length,data):
    if name in ['history', 'previous_offer', 'previous_trade']:
        temp = []
        for h in data:
            if h == 18401:
                temp.append(-3)
            elif h == 18402:
                temp.append(311)
            else:
                temp.append(h)
        if len(temp) <= length:
            new = [-3] * (length - len(temp))
            data = new + temp
        else:
            data = temp[-length:]

    elif name in ['bids', 'asks']:
        key_list = list(data.values())
        key_list = [h for h in key_list]
        key_list.sort()
        if len(key_list) < length:
            new = [-3] * (length - len(key_list))
            data = key_list + new
        elif len(key_list) > length:
            if name == 'bids':
                data = key_list[-3:]  
            else:
                data = key_list[0:3]  
        else:
            data = key_list
    else:
        data = [data]
    data = np.array([data])
    
    mask = (data == -3)
    data = data.astype(float)  
    data[mask] = np.nan
    
    if name not in ['unit', 'time', 'role']:
        
        mean = np.load('saved_model/' + name + '_mean.npy')
        std = np.load('saved_model/' + name + '_std.npy')
        data[~mask] = (data[~mask] - mean) / std
    
    data[mask] = -3
    
    return data

def generate_rl_data(data,value,total_time):
    bids_data = []
    asks_data = []
    trade_count_data = []
    trade_max_data = []
    trade_min_data = []
    trade_avg_data = []
    trade_mid_data = []
    buy_offer_count_data = []
    buy_offer_max_data = []
    buy_offer_min_data = []
    buy_offer_avg_data = []
    buy_offer_mid_data = []
    hist_data = []
    unit_data = []
    pre_offer_data = []
    pre_trade_data = []
    time_data=[]
    action_data=[]
    value_data=[]
    value_ratio=[]
    risk_level=[]
    for line in data:
        bids_data.append(generate_data('bids', 3, line['bids'],value)[0])
        asks_data.append(generate_data('asks', 3, line['asks'],value)[0])
        hist_data.append(generate_data('history', 15, line['history'],value)[0])
        unit_data.append(generate_data('unit', 0, line['unit'],value))
        pre_offer_data.append(generate_data('previous_offer', 2, line['previous_offer'],value)[0])
        pre_trade_data.append(generate_data('previous_trade', 2, line['previous_trade'],value)[0])
        trade_count_data.append(generate_data('trade_count', 0, line['trade_count'],value)[0])
        trade_avg_data.append(generate_data('trade_avg', 0, line['trade_avg'],value)[0])
        trade_max_data.append(generate_data('trade_max', 0, line['trade_max'],value)[0])
        trade_min_data.append(generate_data('trade_min', 0, line['trade_min'],value)[0])
        trade_mid_data.append(generate_data('trade_mid', 0, line['trade_mid'],value)[0])
        buy_offer_count_data.append(generate_data('offer_count', 0, line['offer_count'],value)[0])
        buy_offer_max_data.append(generate_data('offer_max', 0, line['offer_max'],value)[0])
        buy_offer_min_data.append(generate_data('offer_min', 0, line['offer_min'],value)[0])
        buy_offer_avg_data.append(generate_data('offer_avg', 0, line['offer_avg'],value)[0])
        buy_offer_mid_data.append(generate_data('offer_mid', 0, line['offer_mid'],value)[0])
        time_data.append(generate_data('time', 0, line['time']/total_time,value))
        value_data.append(generate_data('value', 0, value,value)[0])
        value_ratio.append(generate_data('value_ratio', 0,line['asks'] ,value)[0])
        risk_level.append(generate_data('risk_level', 0,line['asks'] ,value)[0])
        action_data.append([line['behavior1'],line['behavior2']])
    bids_data = tf.convert_to_tensor(bids_data, dtype=tf.float32)
    asks_data = tf.convert_to_tensor(asks_data, dtype=tf.float32)
    trade_count_data = tf.convert_to_tensor(trade_count_data, dtype=tf.float32)
    trade_count_data = tf.reshape(trade_count_data, (-1, 1))
    trade_max_data = tf.convert_to_tensor(trade_max_data, dtype=tf.float32)
    trade_max_data = tf.reshape(trade_max_data, (-1, 1))
    trade_min_data = tf.convert_to_tensor(trade_min_data, dtype=tf.float32)
    trade_min_data = tf.reshape(trade_min_data, (-1, 1))
    trade_avg_data = tf.convert_to_tensor(trade_avg_data, dtype=tf.float32)
    trade_avg_data = tf.reshape(trade_avg_data, (-1, 1))
    trade_mid_data = tf.convert_to_tensor(trade_mid_data, dtype=tf.float32)
    trade_mid_data = tf.reshape(trade_mid_data, (-1, 1))
    buy_offer_count_data = tf.convert_to_tensor(buy_offer_count_data, dtype=tf.float32)
    buy_offer_count_data = tf.reshape(buy_offer_count_data, (-1, 1))
    buy_offer_max_data = tf.convert_to_tensor(buy_offer_max_data, dtype=tf.float32)
    buy_offer_max_data = tf.reshape(buy_offer_max_data, (-1, 1))
    buy_offer_min_data = tf.convert_to_tensor(buy_offer_min_data, dtype=tf.float32)
    buy_offer_min_data = tf.reshape(buy_offer_min_data, (-1, 1))
    buy_offer_avg_data = tf.convert_to_tensor(buy_offer_avg_data, dtype=tf.float32)
    buy_offer_avg_data = tf.reshape(buy_offer_avg_data, (-1, 1))
    buy_offer_mid_data = tf.convert_to_tensor(buy_offer_mid_data, dtype=tf.float32)
    buy_offer_mid_data = tf.reshape(buy_offer_mid_data, (-1, 1))
    hist_data = tf.convert_to_tensor(hist_data, dtype=tf.float32)
    unit_data = tf.convert_to_tensor(unit_data, dtype=tf.float32)
    unit_data =tf.reshape(unit_data, (-1, 1))
    pre_offer_data = tf.convert_to_tensor(pre_offer_data, dtype=tf.float32)
    pre_trade_data = tf.convert_to_tensor(pre_trade_data, dtype=tf.float32)
    time_data = tf.convert_to_tensor(time_data, dtype=tf.float32)
    time_data = tf.reshape(time_data, (-1, 1))
    value_data = tf.convert_to_tensor(value_data, dtype=tf.float32)
    value_data = tf.reshape(value_data, (-1, 1))
    value_ratio = tf.convert_to_tensor(value_ratio, dtype=tf.float32)
    value_ratio = tf.reshape(value_ratio, (-1, 1))
    risk_level = tf.convert_to_tensor(risk_level, dtype=tf.float32)
    risk_level = tf.reshape(risk_level, (-1, 1))
    return hist_data, bids_data, asks_data, unit_data, pre_offer_data,pre_trade_data,\
           trade_count_data, trade_max_data, trade_min_data,trade_avg_data, trade_mid_data,\
           buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, \
           buy_offer_mid_data,action_data,time_data,value_data,value_ratio,risk_level

def generate_rl_data_vector(data,value_vector,total_time):
    bids_data = []
    asks_data = []
    trade_count_data = []
    trade_max_data = []
    trade_min_data = []
    trade_avg_data = []
    trade_mid_data = []
    buy_offer_count_data = []
    buy_offer_max_data = []
    buy_offer_min_data = []
    buy_offer_avg_data = []
    buy_offer_mid_data = []
    hist_data = []
    unit_data = []
    pre_offer_data = []
    pre_trade_data = []
    time_data=[]
    action_data=[]
    value_data=[]
    value_ratio=[]
    risk_level=[]
    for line in data:
        unit = line['unit']  
        value = tf.gather(value_vector, unit - 1)
        bids_data.append(generate_data('bids', 3, line['bids'],value)[0])
        asks_data.append(generate_data('asks', 3, line['asks'],value)[0])
        hist_data.append(generate_data('history', 15, line['history'],value)[0])
        unit_data.append(generate_data('unit', 0, line['unit'],value))
        pre_offer_data.append(generate_data('previous_offer', 2, line['previous_offer'],value)[0])
        pre_trade_data.append(generate_data('previous_trade', 2, line['previous_trade'],value)[0])
        trade_count_data.append(generate_data('trade_count', 0, line['trade_count'],value)[0])
        trade_avg_data.append(generate_data('trade_avg', 0, line['trade_avg'],value)[0])
        trade_max_data.append(generate_data('trade_max', 0, line['trade_max'],value)[0])
        trade_min_data.append(generate_data('trade_min', 0, line['trade_min'],value)[0])
        trade_mid_data.append(generate_data('trade_mid', 0, line['trade_mid'],value)[0])
        buy_offer_count_data.append(generate_data('offer_count', 0, line['offer_count'],value)[0])
        buy_offer_max_data.append(generate_data('offer_max', 0, line['offer_max'],value)[0])
        buy_offer_min_data.append(generate_data('offer_min', 0, line['offer_min'],value)[0])
        buy_offer_avg_data.append(generate_data('offer_avg', 0, line['offer_avg'],value)[0])
        buy_offer_mid_data.append(generate_data('offer_mid', 0, line['offer_mid'],value)[0])
        time_data.append(generate_data('time', 0, line['time']/total_time,value))
        value_data.append(generate_data('value', 0, value,value)[0])
        value_ratio.append(generate_data('value_ratio', 0,line['asks'] ,value)[0])
        risk_level.append(generate_data('risk_level', 0,line['asks'] ,value)[0])
        action_data.append([line['behavior1'],line['behavior2']])
    bids_data = tf.convert_to_tensor(bids_data, dtype=tf.float32)
    asks_data = tf.convert_to_tensor(asks_data, dtype=tf.float32)
    trade_count_data = tf.convert_to_tensor(trade_count_data, dtype=tf.float32)
    trade_count_data = tf.reshape(trade_count_data, (-1, 1))
    trade_max_data = tf.convert_to_tensor(trade_max_data, dtype=tf.float32)
    trade_max_data = tf.reshape(trade_max_data, (-1, 1))
    trade_min_data = tf.convert_to_tensor(trade_min_data, dtype=tf.float32)
    trade_min_data = tf.reshape(trade_min_data, (-1, 1))
    trade_avg_data = tf.convert_to_tensor(trade_avg_data, dtype=tf.float32)
    trade_avg_data = tf.reshape(trade_avg_data, (-1, 1))
    trade_mid_data = tf.convert_to_tensor(trade_mid_data, dtype=tf.float32)
    trade_mid_data = tf.reshape(trade_mid_data, (-1, 1))
    buy_offer_count_data = tf.convert_to_tensor(buy_offer_count_data, dtype=tf.float32)
    buy_offer_count_data = tf.reshape(buy_offer_count_data, (-1, 1))
    buy_offer_max_data = tf.convert_to_tensor(buy_offer_max_data, dtype=tf.float32)
    buy_offer_max_data = tf.reshape(buy_offer_max_data, (-1, 1))
    buy_offer_min_data = tf.convert_to_tensor(buy_offer_min_data, dtype=tf.float32)
    buy_offer_min_data = tf.reshape(buy_offer_min_data, (-1, 1))
    buy_offer_avg_data = tf.convert_to_tensor(buy_offer_avg_data, dtype=tf.float32)
    buy_offer_avg_data = tf.reshape(buy_offer_avg_data, (-1, 1))
    buy_offer_mid_data = tf.convert_to_tensor(buy_offer_mid_data, dtype=tf.float32)
    buy_offer_mid_data = tf.reshape(buy_offer_mid_data, (-1, 1))
    hist_data = tf.convert_to_tensor(hist_data, dtype=tf.float32)
    unit_data = tf.convert_to_tensor(unit_data, dtype=tf.float32)
    unit_data =tf.reshape(unit_data, (-1, 1))
    pre_offer_data = tf.convert_to_tensor(pre_offer_data, dtype=tf.float32)
    pre_trade_data = tf.convert_to_tensor(pre_trade_data, dtype=tf.float32)
    time_data = tf.convert_to_tensor(time_data, dtype=tf.float32)
    time_data = tf.reshape(time_data, (-1, 1))
    value_data = tf.convert_to_tensor(value_data, dtype=tf.float32)
    value_data = tf.reshape(value_data, (-1, 1))
    value_ratio = tf.convert_to_tensor(value_ratio, dtype=tf.float32)
    value_ratio = tf.reshape(value_ratio, (-1, 1))
    risk_level = tf.convert_to_tensor(risk_level, dtype=tf.float32)
    risk_level = tf.reshape(risk_level, (-1, 1))
    return hist_data, bids_data, asks_data, unit_data, pre_offer_data,pre_trade_data,\
           trade_count_data, trade_max_data, trade_min_data,trade_avg_data, trade_mid_data,\
           buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, \
           buy_offer_mid_data,action_data,time_data,value_data,value_ratio,risk_level

class MarketEnvironment:
    def __init__(self, buyers, sellers, model_classification, model_classification_price, model_regression,
                 current_unit, bids, asks, transactions, buy_offers, sell_offers, env_history, env_pre_offer,
                 env_pre_trade, userid,time,total_time):
        self.buyers = buyers
        self.sellers = sellers
        self.bids = bids.copy()
        self.asks = asks.copy()
        self.transactions = transactions.copy()
        self.buy_offers = buy_offers.copy()
        self.sell_offers = sell_offers.copy()
        self.history = env_history.copy()
        self.previous_offer = env_pre_offer.copy()
        self.previous_trade = env_pre_trade.copy()
        self.count = time
        self.userid = userid
        self.current_unit = current_unit
        model_classification.reset_states()
        model_regression.reset_states()
        self.model_classification = model_classification
        self.model_classification_price = model_classification_price
        self.model_regression = model_regression
        self.total_time = total_time

    def step(self, price,change):
        
        trade,price=self._update_market(price,change)
        
        new_state = self.get_state()
        print('-------------count--------------')
        print(self.count)
        self.count+=1
        if self.count==self.total_time:
            done = True  
        else:
            done=False
        return new_state, done, trade,price

    def get_state(self):
        
        # with open('test_newIRL_data.txt', 'a+', encoding='utf-8') as file:
        #     data = {
        #         'bids': self.bids,
        #         'asks': self.asks,
        #         'transactions': self.transactions,
        #         'buy_offers': self.buy_offers,
        #         'sell_offers': self.sell_offers,
        #         'time': self.count
        #     }
        #     json.dump(data, file, ensure_ascii=False, indent=4)
        return (self.bids,self.asks,self.transactions,self.buy_offers,self.sell_offers)


    def _update_market(self, price,change):
        trade = 0
        transaction = 0

        change_flag = {}
        for buyer in self.buyers:
            change_flag[buyer] = 0
        for seller in self.sellers:
            change_flag[seller] = 0
        change_flag[self.userid] = change

        if change==1:
            self.buy_offers.append(price)
        min_ask = -1
        #if self.asks != {}:
            # if price > min(self.asks.values()):
            #     min_ask = min(self.asks.values())
            #     trade_price = min_ask
            #condition = price > min_ask
            #trade_price = tf.where(condition, min_ask, price)
        if min_ask == -1:
            self.bids[self.userid] = price
        # else:
        #     trade = 1
        #     transaction = trade_price
        #     print('11111')
        #     print(transaction)
        #     for key in self.asks.keys():
        #         if self.asks[key] == min_ask:
        #             min_ask_id = key
        #             break
        #     del self.asks[min_ask_id]
        #     if self.userid in self.bids.keys():
        #         del self.bids[self.userid]

            # self.transactions.append(trade_price)
            # self.previous_offer[min_ask_id].append(self.history[min_ask_id][1][-1])
            # self.previous_trade[min_ask_id].append(trade_price)
            # ask_unit = self.history[min_ask_id][0]
            # self.history[min_ask_id] = (ask_unit + 1, deque(maxlen=20))

        bids_data = generate_data1('bids', 3, self.bids)
        asks_data = generate_data1('asks', 3, self.asks)
        if self.transactions == []:
            trade_max = -1
            trade_min = -1
            trade_avg = -1
            trade_mid = -1
        else:
            trade_max = max(self.transactions)
            trade_min = min(self.transactions)
            trade_avg = np.mean(self.transactions)
            trade_mid = np.median(self.transactions)
        trade_count_data = generate_data1('trade_count', 0, len(self.transactions))
        trade_max_data = generate_data1('trade_max', 0, trade_max)
        trade_min_data = generate_data1('trade_min', 0, trade_min)
        trade_avg_data = generate_data1('trade_avg', 0, trade_avg)
        trade_mid_data = generate_data1('trade_mid', 0, trade_mid)
        if self.buy_offers == []:
            buy_offer_max = -1
            buy_offer_min = -1
            buy_offer_avg = -1
            buy_offer_mid = -1
        else:
            buy_offer_max = max(self.buy_offers)
            buy_offer_min = min(self.buy_offers)
            buy_offer_avg = np.mean(self.buy_offers)
            buy_offer_mid = np.median(self.buy_offers)
        buy_offer_count_data = generate_data1('offer_count', 0, len(self.buy_offers))
        buy_offer_max_data = generate_data1('offer_max', 0, buy_offer_max)
        buy_offer_min_data = generate_data1('offer_min', 0, buy_offer_min)
        buy_offer_avg_data = generate_data1('offer_avg', 0, buy_offer_avg)
        buy_offer_mid_data = generate_data1('offer_mid', 0, buy_offer_mid)
        if self.sell_offers == []:
            sell_offer_max = -1
            sell_offer_min = -1
            sell_offer_avg = -1
            sell_offer_mid = -1
        else:
            sell_offer_max = max(self.sell_offers)
            sell_offer_min = min(self.sell_offers)
            sell_offer_avg = np.mean(self.sell_offers)
            sell_offer_mid = np.median(self.sell_offers)
        sell_offer_count_data = generate_data1('offer_count', 0, len(self.sell_offers))
        sell_offer_max_data = generate_data1('offer_max', 0, sell_offer_max)
        sell_offer_min_data = generate_data1('offer_min', 0, sell_offer_min)
        sell_offer_avg_data = generate_data1('offer_avg', 0, sell_offer_avg)
        sell_offer_mid_data = generate_data1('offer_mid', 0, sell_offer_mid)
        for buyer in self.buyers:
            hist = self.history[buyer][1]
            
            if self.history[buyer][0] > self.current_unit:
                continue
            unit_data = generate_data1('unit', 0, self.history[buyer][0])
            hist_data = generate_data1('history', 5, list(hist))
            pre_offer_data = generate_data1('previous_offer', 2, self.previous_offer[buyer])
            pre_trade_data = generate_data1('previous_trade', 2, self.previous_trade[buyer])
            role_data = np.array([0])
            price = self.model_regression.predict(
                [hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data,
                 trade_count_data, trade_max_data, trade_min_data, trade_avg_data, trade_mid_data,
                 buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data,
                 buy_offer_mid_data, role_data])[
                0][0]
            price = round(price)
            if price < 0:
                price = 0
            if price > 300:
                price = 300

            change_flag[buyer] = 1
            # if len(hist) != 0:
            #     if self.history[buyer][1][-1]//10 == price//10:
            #         change_flag[buyer] = 0
            #         self.history[buyer][1].append(hist[-1])

            if change_flag[buyer] == 1:
                if len(hist) != 0:
                    if self.history[buyer][1][-1] // 10 == price // 10:
                        change_flag[buyer] = 0
                    else:
                        self.buy_offers.append(price)
                else:
                    self.buy_offers.append(price)
                self.history[buyer][1].append(price)
                self.bids[buyer] = price

        for seller in self.sellers:
            hist = self.history[seller][1]
            
            if self.history[seller][0] > self.current_unit:
                continue
            unit_data = generate_data1('unit', 0, self.history[seller][0])
            hist_data = generate_data1('history', 5, list(hist))
            pre_offer_data = generate_data1('previous_offer', 2, self.previous_offer[seller])
            pre_trade_data = generate_data1('previous_trade', 2, self.previous_trade[seller])
            role_data = np.array([1])

            price = self.model_regression.predict(
                [hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data,
                 trade_count_data, trade_max_data, trade_min_data, trade_avg_data, trade_mid_data,
                 sell_offer_count_data, sell_offer_max_data, sell_offer_min_data, sell_offer_avg_data,
                 sell_offer_mid_data, role_data])[0][0]
            price = round(price)
            if price < 0:
                price = 0
            if price > 300:
                price = 300

            change_flag[seller] = 1
            # if len(hist) != 0:
            #     if self.history[seller][1][-1] //10 == price//10:
            #         change_flag[seller] = 0
            #         self.history[seller][1].append(hist[-1])

            if change_flag[seller] == 1:
                if len(hist) != 0:
                    if self.history[seller][1][-1] // 10 == price // 10:
                        change_flag[seller] = 0
                    else:
                        self.sell_offers.append(price)
                else:
                    self.sell_offers.append(price)
                self.history[seller][1].append(price)
                self.asks[seller] = price

        if self.bids and self.asks:
            max_bid = max(self.bids.values())
            min_ask = min(self.asks.values())
            while (max_bid >= min_ask):
                for key in self.bids.keys():
                    if self.bids[key] == max_bid:
                        max_bid_id = key
                        break
                for key in self.asks.keys():
                    if self.asks[key] == min_ask:
                        min_ask_id = key
                        break
                del self.bids[max_bid_id]
                del self.asks[min_ask_id]

                transaction_price = (min_ask + max_bid) // 2
                # if change_flag[max_bid_id] == 1:
                #     if change_flag[min_ask_id] == 1:
                #         transaction_price = (min_ask + max_bid) / 2
                #     else:
                #         transaction_price = min_ask
                # else:
                #     if change_flag[min_ask_id] == 1:
                #         transaction_price = max_bid

                self.transactions.append(transaction_price)
                if max_bid_id == self.userid:
                    trade = 1
                    transaction = transaction_price
                else:
                    self.previous_offer[max_bid_id].append(self.history[max_bid_id][1][-1])
                    self.previous_trade[max_bid_id].append(transaction_price)
                    bid_unit = self.history[max_bid_id][0]
                    self.history[max_bid_id] = (bid_unit + 1, [])
                print(min_ask_id)
                print(self.previous_offer)
                print(self.history)
                self.previous_offer[min_ask_id].append(self.history[min_ask_id][1][-1])
                self.previous_trade[min_ask_id].append(transaction_price)
                ask_unit = self.history[min_ask_id][0]
                self.history[min_ask_id] = (ask_unit + 1, [])

                time_info = {
                    'time': self.count,
                    'bid_id': max_bid_id,
                    'bid_offer': max_bid,
                    'ask_id': min_ask_id,
                    'ask_offer': min_ask,
                    'price': transaction_price
                }


                if self.bids == {} or self.asks == {}:
                    break
                else:
                    max_bid = max(self.bids.values())
                    min_ask = min(self.asks.values())

        return trade, transaction

def initialize_market(current_marketID_period,initial_time):
    user = pd.read_csv('data_processing/user.csv')
    user = user.dropna()
    user_count = 0
    buyers = []
    sellers = []
    all_cost = {}
    while user_count < user.shape[0]:
        user_row = user.iloc[user_count]
        user_count += 1
        marketID_period = str(user_row['MarketID_period'])
        if int(marketID_period) > int(current_marketID_period):
            break
        elif int(marketID_period) < int(current_marketID_period):
            continue
        userid = str(int(user_row['userid_profile']))
        role = user_row['role']
        valuecost = user_row['valuecost']
        all_cost[userid] = valuecost.split('-')
        current_unit = valuecost.count('-') + 1
        if role == 'buyer':
            buyers.append(userid)
        else:
            sellers.append(userid)

    whole_history = {}
    whole_previous_offer = {}
    whole_previous_trade = {}
    bids = {}
    asks = {}
    transactions = []
    buy_offers = []
    sell_offers = []

    for buyer in buyers:
        whole_history[buyer] = (1, deque(maxlen=20))
        whole_previous_offer[buyer] = []
        whole_previous_trade[buyer] = []
    for seller in sellers:
        whole_history[seller] = (1, deque(maxlen=20))
        whole_previous_offer[seller] = []
        whole_previous_trade[seller] = []

    
    df = pd.read_csv('data_processing/transaction.csv')
    data = df.dropna()
    row_count = 0
    with open('data_processing/time.jsonl', 'r') as f:
        for line in f:
            l = json.loads(line)
            marketID_period = l['MarketID_period']
            if int(marketID_period) < int(current_marketID_period):
                row_count += l['time_count']
            else:
                break

    for i in range(initial_time):
        row = data.iloc[row_count]
        row_count += 1
        bidask = row['bidask']
        if bidask == 'MARKET ORDER':
            if row['status'] == 'EXPIRED':
                continue
            if row['transactedprice'] == '--':
                bidask = 18402
            else:
                bidask = int(row['transactedprice'])
        else:
            bidask = int(bidask)
        user_id = str(int(row['userid']))

        role = row['playerrole'].lower()
        for user, unit_hist in whole_history.items():
            unit = unit_hist[0]
            if unit > current_unit:
                continue
            hist = unit_hist[1]
            if user == user_id:
                whole_history[user_id][1].append(bidask)
            else:
                if len(whole_history[user][1]) == 0:
                    whole_history[user][1].append(18401)
                else:
                    whole_history[user][1].append(hist[-1])
        if role == 'buyer':
            bids[user_id] = (bidask, i)
            buy_offers.append(bidask)
        else:
            asks[user_id] = (bidask, i)
            sell_offers.append(bidask)

        if bids and asks:
            
            bid_prices = [price for price, _ in bids.values()]
            ask_prices = [price for price, _ in asks.values()]

            
            if 18402 in bid_prices:
                max_bid_price = 18402
            else:
                max_bid_price = max(bid_prices) if bid_prices else -1

            if 18402 in ask_prices:
                min_ask_price = 18402
            else:
                min_ask_price = min(ask_prices) if ask_prices else float('inf')

            
            while (min_ask_price == 18402) or (max_bid_price == 18402) or (max_bid_price >= min_ask_price):
                
                candidates = [(time, uid) for uid, (price, time) in bids.items() if price == max_bid_price]
                if candidates:
                    
                    candidates.sort()
                    max_bid_id = candidates[0][1]
                else:
                    break  
                
                candidates = [(time, uid) for uid, (price, time) in asks.items() if price == min_ask_price]
                if candidates:
                    candidates.sort()
                    min_ask_id = candidates[0][1]
                else:
                    break
                
                if min_ask_price == 18402:
                    transaction_price = max_bid_price
                elif max_bid_price == 18402:
                    transaction_price = min_ask_price
                else:
                   
                    max_bid_time = bids[max_bid_id][1]
                    min_ask_time = asks[min_ask_id][1]
                    transaction_price = max_bid_price if max_bid_time < min_ask_time else min_ask_price

                
                transactions.append(transaction_price)

                
                whole_previous_offer[max_bid_id].append(whole_history[max_bid_id][1][-1])
                whole_previous_trade[max_bid_id].append(transaction_price)
                whole_previous_offer[min_ask_id].append(whole_history[min_ask_id][1][-1])
                whole_previous_trade[min_ask_id].append(transaction_price)

                
                bid_unit = whole_history[max_bid_id][0]
                ask_unit = whole_history[min_ask_id][0]
                whole_history[max_bid_id] = (bid_unit + 1, deque(maxlen=20))  
                whole_history[min_ask_id] = (ask_unit + 1, deque(maxlen=20))

                
                del bids[max_bid_id]
                del asks[min_ask_id]

                
                if not bids or not asks:
                    break

                
                bid_prices = [price for price, _ in bids.values()]
                ask_prices = [price for price, _ in asks.values()]

                max_bid_price = 18402 if 18402 in bid_prices else (max(bid_prices) if bid_prices else -1)
                min_ask_price = 18402 if 18402 in ask_prices else (min(ask_prices) if ask_prices else float('inf'))
        final_bids = {userid: price for userid, (price, _) in bids.items()}
        final_asks = {userid: price for userid, (price, _) in asks.items()}
    return buyers,whole_history,whole_previous_offer,whole_previous_trade,sellers,current_unit,final_bids,final_asks,transactions,buy_offers,sell_offers

def find_time_count(jsonl_file, current_marketID_period):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())  
            if data["MarketID_period"] == current_marketID_period:
                return data["time_count"]
    return None  

def find_last_time(csv_file, market):
   
    df = pd.read_csv(csv_file, dtype={'MarketID_period': str})
    
    match = df[df['MarketID_period'] == market]
    
    if not match.empty:
        return match['offer_time'].iloc[-1]  
    return None  

class ReplayBuffer:
    
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  
        self.decay_factor = 0.00005
        self.decay=1.0

    def add(self, state, action, reward, advantage,next_state, done):  
        #if (reward>=0) | (random.random()<self.decay) :
        self.buffer.append((state, action, reward, advantage,next_state, done))
        self.decay-=self.decay_factor

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, advantage,next_state, done = zip(*transitions)
        return state, action, reward, advantage,next_state, done

    def size(self):  
        return len(self.buffer)

    def get_state(self):
        
        return {
            'buffer': self.buffer,
            'buffer_size': self.buffer_size,
            'count': self.count
        }

    def set_state(self, state):
        
        self.buffer = state['buffer']
        self.buffer_size = state['buffer_size']
        self.count = state['count']

def generate_statistics(data, prefix):
    if not data:
        return {
            f'{prefix}_max': -1,
            f'{prefix}_min': -1,
            f'{prefix}_avg': -1,
            f'{prefix}_mid': -1,
            f'{prefix}_count': 0
        }
    return {
        f'{prefix}_max': max(data),
        f'{prefix}_min': min(data),
        f'{prefix}_avg': np.mean(data),
        f'{prefix}_mid': np.median(data),
        f'{prefix}_count': len(data)
    }

def generate_state_data(state, cost, unit, history, previous_offer, previous_trade, time, total_time):
    hist_data = generate_data('history', 15, history, cost[unit - 1])
    pre_offer_data = generate_data('previous_offer', 2, previous_offer, cost[unit - 1])
    pre_trade_data = generate_data('previous_trade', 2, previous_trade, cost[unit - 1])
    bids_data = generate_data('bids', 3, state[0], cost[unit - 1])
    asks_data = generate_data('asks', 3, state[1], cost[unit - 1])
    unit_data = generate_data('unit', 0, unit, cost[unit - 1])

    trade_stats = generate_statistics(state[2], 'trade')
    buy_offer_stats = generate_statistics(state[3], 'buy_offer')
    trade_count_data = generate_data('trade_count', 0, trade_stats['trade_count'], cost[unit - 1])
    trade_max_data = generate_data('trade_max', 0, trade_stats['trade_max'], cost[unit - 1])
    trade_min_data = generate_data('trade_min', 0, trade_stats['trade_min'], cost[unit - 1])
    trade_avg_data = generate_data('trade_avg', 0, trade_stats['trade_avg'], cost[unit - 1])
    trade_mid_data = generate_data('trade_mid', 0, trade_stats['trade_mid'], cost[unit - 1])

    buy_offer_count_data = generate_data('offer_count', 0, buy_offer_stats['buy_offer_count'], cost[unit - 1])
    buy_offer_max_data = generate_data('offer_max', 0, buy_offer_stats['buy_offer_max'], cost[unit - 1])
    buy_offer_min_data = generate_data('offer_min', 0, buy_offer_stats['buy_offer_min'], cost[unit - 1])
    buy_offer_avg_data = generate_data('offer_avg', 0, buy_offer_stats['buy_offer_avg'], cost[unit - 1])
    buy_offer_mid_data = generate_data('offer_mid', 0, buy_offer_stats['buy_offer_mid'], cost[unit - 1])

    role_data = np.array([[0]])  # buyer
    value_data = generate_data('value', 3, cost[unit - 1], cost[unit - 1])
    time_data = generate_data('time', 0, time / total_time, cost[unit - 1])
    value_ratio=generate_data('value_ratio', 0, state[1], cost[unit - 1])
    risk_level=generate_data('risk_level', 0, state[1], cost[unit - 1])

    state_data = [
        hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data,
        trade_count_data, trade_max_data, trade_min_data, trade_avg_data, trade_mid_data,
        buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
        role_data, value_data, time_data,value_ratio,risk_level
    ]
    return state_data

def generate_state_data_for_expert(state, cost, unit, history, previous_offer, previous_trade, time, total_time):
    hist_data = generate_data('history', 5, history, cost[unit - 1])
    pre_offer_data = generate_data('previous_offer', 2, previous_offer, cost[unit - 1])
    pre_trade_data = generate_data('previous_trade', 2, previous_trade, cost[unit - 1])
    bids_data = generate_data('bids', 3, state[0], cost[unit - 1])
    asks_data = generate_data('asks', 3, state[1], cost[unit - 1])
    unit_data = generate_data('unit', 0, unit, cost[unit - 1])

    trade_stats = generate_statistics(state[2], 'trade')
    buy_offer_stats = generate_statistics(state[3], 'buy_offer')
    trade_count_data = generate_data('trade_count', 0, trade_stats['trade_count'], cost[unit - 1])
    trade_max_data = generate_data('trade_max', 0, trade_stats['trade_max'], cost[unit - 1])
    trade_min_data = generate_data('trade_min', 0, trade_stats['trade_min'], cost[unit - 1])
    trade_avg_data = generate_data('trade_avg', 0, trade_stats['trade_avg'], cost[unit - 1])
    trade_mid_data = generate_data('trade_mid', 0, trade_stats['trade_mid'], cost[unit - 1])

    buy_offer_count_data = generate_data('offer_count', 0, buy_offer_stats['buy_offer_count'], cost[unit - 1])
    buy_offer_max_data = generate_data('offer_max', 0, buy_offer_stats['buy_offer_max'], cost[unit - 1])
    buy_offer_min_data = generate_data('offer_min', 0, buy_offer_stats['buy_offer_min'], cost[unit - 1])
    buy_offer_avg_data = generate_data('offer_avg', 0, buy_offer_stats['buy_offer_avg'], cost[unit - 1])
    buy_offer_mid_data = generate_data('offer_mid', 0, buy_offer_stats['buy_offer_mid'], cost[unit - 1])

    role_data = np.array([[0]])  # buyer

    state_data = [
        hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data,
        trade_count_data, trade_max_data, trade_min_data, trade_avg_data, trade_mid_data,
        buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
        role_data]
    return state_data

def calculate_reward(trade, offer, price, cost, unit, state_values, next_state_values):
    if trade == 1:
        reward = cost[unit - 1] - price
    else:
        reward=0

    safety_penalty = 0
    if state_values:
        best_ask = min(state_values)
        if best_ask >= cost[unit - 1] : 
            if best_ask > offer: 
                if cost[unit - 1] > offer: 
                    safety_penalty = 1
                else:
                    safety_penalty = (cost[unit - 1] - offer) * 0.1
                if next_state_values:
                    next_best_ask = min(next_state_values)
                    safety_penalty += (best_ask - next_best_ask)* 0.5
            else: 
                safety_penalty = (best_ask - offer) * 0.1
        elif best_ask < cost[unit - 1]: 
            if best_ask > offer:
                safety_penalty = (offer - best_ask) * 1.0
                if next_state_values:  
                    next_best_ask = min(next_state_values)
                    if next_best_ask <= best_ask:
                        safety_penalty = (best_ask - next_best_ask)* 0.5
            else:
                safety_penalty = (best_ask - offer) * 0.1
    else:
        if cost[unit - 1] > offer:
            safety_penalty = 1
        else:
            safety_penalty = (cost[unit - 1] - offer) * 0.1

    if reward>0:
        total_reward = reward*unit + safety_penalty + 10
    else:
        total_reward = reward + safety_penalty
    return total_reward, reward


def calculate_reward1(trade, offer, price, cost, unit, state_values, next_state_values):
    if trade == 1:
        reward = cost[unit - 1] - price
    else:
        reward=0

    safety_penalty = 0
    if state_values:
        best_ask = min(state_values)
        if best_ask >= cost[unit - 1] : 
            if best_ask > offer: 
                if cost[unit - 1] > offer: 
                    safety_penalty = 1
                else:
                    safety_penalty = (cost[unit - 1] - offer) * 0.1
                if next_state_values:
                    next_best_ask = min(next_state_values)
                    safety_penalty += (best_ask - next_best_ask)* 0.5
            else: 
                safety_penalty = (best_ask - offer) * 0.1
        elif best_ask < cost[unit - 1]: 
            if best_ask > offer:
                safety_penalty = (offer - best_ask) * 1.0
                if next_state_values:  
                    next_best_ask = min(next_state_values)
                    if next_best_ask <= best_ask:
                        safety_penalty = (best_ask - next_best_ask)* 0.5
            else:
                safety_penalty = (best_ask - offer) * 0.1
    else:
        if cost[unit - 1] > offer:
            safety_penalty = 1
        else:
            safety_penalty = (cost[unit - 1] - offer) * 0.1

    if reward>0:
        total_reward = reward + safety_penalty
    else:
        total_reward = reward + safety_penalty
    return total_reward, reward

def process_tensors(tensor_list,feature_num):
    
    all_features = []
    for row in tensor_list:
        features = []
        for tensor in row:
            if tensor.ndim == 2:
                features.extend(tensor)  
            elif tensor.ndim == 1:
                features.extend(tensor)
            elif tensor.ndim == 0:
                features.append(tensor)
        all_features.append(features)

    inputs = []
    for i in range(feature_num):
        input_tensors = [row[i] for row in all_features]
        input_tensor = tf.stack(input_tensors)
        inputs.append(input_tensor)
    return inputs

def smoothness_loss(action_probs):
    
    diff = action_probs[:, 1:] - action_probs[:, :-1]  
    return tf.reduce_mean(diff ** 2)  

def get_random_time(total_time):
    lower_bound = total_time // 4
    upper_bound = 3 * total_time // 4
    return random.randint(lower_bound, upper_bound)

def constrained_params(params):
    clipped = tf.clip_by_value(params, 20.0, 250.0)
    sorted_params = -tf.sort(-clipped)
    params.assign(sorted_params)
    return sorted_params

def train_step(hist_data, bids_data, asks_data, unit_data, pre_offer_data,
                               pre_trade_data,trade_count_data, trade_max_data, trade_min_data,
                               trade_avg_data, trade_mid_data,buy_offer_count_data, buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
                               role_data,action, time_data,valuation_data,value_ratio,risk_level,model_actor3):

    actions_second_elements = [sublist[1]//10 for sublist in action]
    actions_second_elements = tf.convert_to_tensor(actions_second_elements)
    predicted_regress = model_actor3([hist_data, bids_data, asks_data, unit_data, pre_offer_data,
									  pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
									  trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
									  buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
									  role_data, valuation_data
                                         ,time_data, value_ratio, risk_level])
    # print('-----------------predicted_regress-----------------')
    # print(predicted_regress)
    one_hot_labels = tf.one_hot(actions_second_elements, depth=tf.shape(predicted_regress)[-1])
    cross_entropy = -tf.reduce_sum(one_hot_labels * tf.math.log(predicted_regress + 1e-30), axis=-1)

    loss=tf.reduce_mean(cross_entropy)

    
    batch_size = tf.shape(cross_entropy)[0]
    weights = tf.exp(tf.linspace(
        start=tf.math.log(0.01),  
        stop=tf.math.log(1.0),  
        num=batch_size
    ))
    loss = tf.reduce_sum(cross_entropy * weights)

    return loss

def objective(value_vector,market,buyer,model_actor,value_len):
    loss = 0.0
    num = 0
    for i in range(value_len):
        data_path = 'data_processing/inference_data.jsonl'
        data = []

        with open(data_path, 'r') as file:
            for line in file:
                l = json.loads(line)
                if (l["MarketID_period"] == market) and (l['unit'] == i + 1) and (l['userid'] == int(buyer)):
                    if l['behavior'] != 18401:
                        l['behavior2'] = l['behavior']
                        l['behavior1'] = l['class']
                        data.append(l)
                        num+=1

        if data != []:
            last_time = find_last_time("data_processing/market-stat-data.csv", market)
            hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data, trade_count_data, \
                trade_max_data, trade_min_data, trade_avg_data, trade_mid_data, buy_offer_count_data, \
                buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data, action_data, time_data, value_data, value_ratio, risk_level = generate_rl_data(
                data, value_vector[i], last_time)
            role_data = np.zeros((hist_data.shape[0], 1))

            loss += train_step(hist_data, bids_data, asks_data, unit_data, pre_offer_data,
                               pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
                               trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
                               buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data, role_data,
                               action_data, time_data, value_data, value_ratio, risk_level, model_actor)
    if num <= 5:
        return 0.0
    return loss

def apply_momentum(grad, momentum_var, beta=0.9):
    
    momentum_var.assign(beta * momentum_var + (1 - beta) * grad)

def optimize_parameters(market_id, buyer_id,  model,value_len,patience=50, threshold=1e-2):
    """Optimize parameters with early stopping mechanism"""
    defaults = {
        3: [200.0, 125.0, 75.0],
        2: [150.0, 50.0],
        1: [100.0]
    }
    initial_values = tf.constant(defaults[value_len], dtype=tf.float32)
    params = tf.Variable(initial_values, trainable=True)
    #optimizer = tf.optimizers.Adam(learning_rate=0.1)
    optimizer = tf.keras.optimizers.Nadam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=5.0,  
            decay_steps=500,
            decay_rate=0.9
        ),
        beta_1=0.9,
        beta_2=0.999
    )
    momentum = tf.Variable(tf.zeros_like(params), trainable=False)  
    # Training monitoring variables
    best_params = None
    min_loss = float('inf')
    no_improvement_count = 0
    for iteration in range(500):  # Set upper limit
        with tf.GradientTape() as tape:
            current_params = constrained_params(params)
            current_loss = objective(current_params,market_id,buyer_id,model,value_len)
            if current_loss==0.0:
                return None,None
        
        if current_loss < min_loss - threshold:
            min_loss = current_loss
            best_params = current_params.numpy()
            no_improvement_count = 0
        else:
            no_improvement_count += 1
        if no_improvement_count >= patience:
            print(f"Early stopping at iteration {iteration}, no improvement for {patience} consecutive steps")
            break
        # Gradient update
        grads = tape.gradient(current_loss, [params])
        apply_momentum(grads[0], momentum)  
        optimizer.apply_gradients(zip(grads, [params]))
        if iteration % 100 == 0:
            print(f"Iter {iteration:04d} | LR: {optimizer.learning_rate.numpy():.3f} | "
                  f"Params: {current_params.numpy().round(1)} | Loss: {current_loss.numpy():.2f}")
    return best_params if best_params is not None else current_params.numpy(),min_loss

def load_user_profile(user_csv_path,inference_market_path):
    
    simulate_markets = set()
    try:
        with open(inference_market_path, 'r', encoding='utf-8') as f:
            for line in f:
                market = line.strip()
                if market:  
                    simulate_markets.add(market)
    except FileNotFoundError:
        print("Warning: simulate_market.txt not found, will include all markets")
        simulate_markets = None  
    user_profile = {}
    with open(user_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not all(row.get(key) for key in ["userid_profile", "role", "MarketID_period", "valuecost"]):
                continue
            if row['role'] != 'buyer':
                continue
            market_id = str(row['MarketID_period'])
            if market_id in simulate_markets:
                valuecost = list(map(int, row["valuecost"].split('-')))
                key = (market_id, int(row['userid_profile']))
                user_profile[key] = valuecost
    return user_profile


def get_user_unit(market_id, user_id, filepath='data_processing/user_unit.jsonl'):
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['market'] == market_id:
                return data['user_unit'].get(str(user_id))
    return None