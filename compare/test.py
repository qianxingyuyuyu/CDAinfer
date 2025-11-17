import numpy as np
import tensorflow as tf
import json
import pandas as pd
import csv
from collections import defaultdict
import random


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
        if name == 'value' or name =='reward':
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

    if name not in ['unit', 'time', 'role', 'value', 'values', 'value_ratio', 'risk_level','reward']:

        mean = tf.convert_to_tensor(np.load('../saved_model/' + name + '_mean.npy'), dtype=tf.float32)
        std = tf.convert_to_tensor(np.load('../saved_model/' + name + '_std.npy'), dtype=tf.float32)
        data_normalized = (data_float - mean) / std
        data_normalized = tf.where(mask, -3.0, data_normalized)
    else:
        data_normalized = data_float

    return data_normalized

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
									  role_data, valuation_data])
                                         #,time_data, value_ratio, risk_level])
    # print(predicted_regress)
    # print(actions_second_elements)
    one_hot_labels = tf.one_hot(actions_second_elements, depth=tf.shape(predicted_regress)[-1])
    cross_entropy = -tf.reduce_sum(one_hot_labels * tf.math.log(predicted_regress + 1e-30), axis=-1)
    #loss=tf.reduce_mean(cross_entropy)


    batch_size = tf.shape(cross_entropy)[0]
    weights = tf.exp(tf.linspace(
        start=tf.math.log(0.01),
        stop=tf.math.log(1.0),
        num=batch_size
    ))
    # print("weights")
    # print(weights)
    # print(cross_entropy)


    loss = tf.reduce_sum(cross_entropy * weights)
    return loss


def objective(value_vector,market,buyer,model_actor,value_len):
    loss = 0.0
    num=0
    for i in range(value_len):
        data_path = '../data_processing/inference_data.jsonl'
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
            last_time = find_last_time("../data_processing/market-stat-data.csv", market)
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
    if num<5:
        loss = 0.0
    return loss

def load_user_profile(user_csv_path):

    user_profile = {}
    with open(user_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not all(row.get(key) for key in ["userid_profile", "role", "MarketID_period", "valuecost"]):
                continue
            valuecost = list(map(int, row["valuecost"].split('-')))
            key = (str(row['MarketID_period']), int(row['userid_profile']))
            user_profile[key] = valuecost
    return user_profile

def process_buyer_bounds(file_path,user_profile):

    with open(file_path, 'r', encoding='utf-8') as f:
        bound_data = json.load(f)

    grouped = defaultdict(dict)
    for record in bound_data:
        if record.get('role') == 'buyer':
            key = (record['MarketID_period'], record['userid'])
            grouped[key][record['unit']] = record['bound']
    result = []
    for (market_id, user_id), valuecost in user_profile.items():
        unit_count= len(valuecost)
        if (market_id, user_id) in grouped:
            user_bounds = grouped.get((market_id, user_id), {})
            unit_bounds = {}
            for unit in range(1, unit_count + 1):
                unit_bounds[unit] = user_bounds.get(unit, 0)
            sorted_bounds = [unit_bounds[u] for u in sorted(unit_bounds)]
            result.append({
                "MarketID_period": market_id,
                "userid": user_id,
                "required_units": unit_count,
                "cost": valuecost,
                "unit_bounds": sorted_bounds
            })
    return result

def generate_sample_cost(bound_list):

    if any(b >= 250 for b in bound_list):
        return None
    sample_cost = []
    previous = 0

    for b in reversed(bound_list):
        upper = 250
        lower = max(b,previous)
        if lower > upper:
            return None

        s = random.randint(lower, upper)
        sample_cost.insert(0, s)
        previous = s
    return sample_cost

def generate_sample_cost_dl(bound_list):
    sample_cost = []
    previous = 0

    for b in reversed(bound_list):
        if b >= 250:
            s = 250
        else:
            upper = 250
            lower = max(b, previous)

            s = random.randint(lower, upper)
        sample_cost.insert(0, s)
        previous = s
    return sample_cost

def generate_sample_cost_mean(bound_list):
    if any(b >= 250 for b in bound_list):
        return None
    sample_cost = []
    previous = 0

    for b in reversed(bound_list):
        upper = 250
        lower = max(b,previous)
        if lower > upper:
            return None

        s = (lower + upper) // 2
        sample_cost.insert(0, s)
        previous = s
    return sample_cost

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
        # Print progress every 500 iterations
        if iteration % 100 == 0:
            print(f"Iter {iteration:04d} | LR: {optimizer.learning_rate.numpy():.3f} | "
                  f"Params: {current_params.numpy().round(1)} | Loss: {current_loss.numpy():.2f}")
    return best_params if best_params is not None else current_params.numpy(),min_loss

def optimize_parameters_reward(market_id, buyer_id,  model,value_len,patience=50, threshold=1e-2):
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
            current_loss = objective_reward(current_params,market_id,buyer_id,model,value_len)
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
        # Print progress every 500 iterations
        if iteration % 100 == 0:
            print(f"Iter {iteration:04d} | LR: {optimizer.learning_rate.numpy():.3f} | "
                  f"Params: {current_params.numpy().round(1)} | Loss: {current_loss.numpy():.2f}")
    return best_params if best_params is not None else current_params.numpy(),min_loss


def filter_buyer_data(user_csv_path, inference_market_path):
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

def calculate_global_mse(optimization_results):
    """Calculate global MSE across all buyers"""
    all_params = []
    all_costs = []
    for result in optimization_results:
        rounded_params = np.round(result['optimized_params'])
        all_params.extend(rounded_params)
        all_costs.extend(result['cost_values'])

    mse = np.mean((np.array(all_params) - np.array(all_costs)) ** 2)
    return mse

def get_user_unit(market_id, user_id, filepath='../data_processing/user_unit.jsonl'):

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data['market'] == market_id:
                return data['user_unit'].get(str(user_id))
    return None

def calculate_reward1(trade, offer, price, cost,  state_values, next_state_values):
    if trade == 1:
        reward = cost - price
    else:
        reward=0

    safety_penalty = 0
    if state_values:
        best_ask = min(state_values)
        if best_ask >= cost :
            if best_ask > offer:
                if cost > offer:
                    safety_penalty = 1
                else:
                    safety_penalty = (cost - offer) * 0.1
                if next_state_values:
                    next_best_ask = min(next_state_values)
                    safety_penalty += (best_ask - next_best_ask)* 0.5
            else:
                safety_penalty = (best_ask - offer) * 0.1
        elif best_ask < cost:
            if best_ask > offer:
                safety_penalty = (offer - best_ask) * 1.0
                if next_state_values:
                    next_best_ask = min(next_state_values)
                    if next_best_ask <= best_ask:
                        safety_penalty = (best_ask - next_best_ask)* 0.5
            else:
                safety_penalty = (best_ask - offer) * 0.1
    else:
        if cost > offer:
            safety_penalty = 1
        else:
            safety_penalty = (cost - offer) * 0.1

    if reward>0:
        total_reward = reward + safety_penalty
    else:
        total_reward = reward + safety_penalty
    return total_reward, reward


def objective_reward(value_vector,market,buyer,model_actor,value_len):
    loss = 0.0
    num=0
    for i in range(value_len):
        data_path = '../data_processing/inference_data.jsonl'
        data = []
        with open(data_path, 'r') as file:
            for line in file:
                l = json.loads(line)
                if (l["MarketID_period"] == market) and (l['unit'] == i + 1) and (l['userid'] == int(buyer)):
                    if l['behavior'] != 18401:
                        l['behavior2'] = l['behavior']
                        l['behavior1'] = l['class']
                        l['reward']= random.randint(0, 100)
                        data.append(l)
                        num+=1

        if data != []:
            last_time = find_last_time("../data_processing/market-stat-data.csv", market)
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
            time_data = []
            action_data = []
            value_data = []
            value_ratio = []
            risk_level = []
            reward_data=[]
            for line in data:
                bids_data.append(generate_data('bids', 3, line['bids'], value_vector[i])[0])
                asks_data.append(generate_data('asks', 3, line['asks'], value_vector[i])[0])
                hist_data.append(generate_data('history', 15, line['history'], value_vector[i])[0])
                unit_data.append(generate_data('unit', 0, line['unit'], value_vector[i]))
                pre_offer_data.append(generate_data('previous_offer', 2, line['previous_offer'], value_vector[i])[0])
                pre_trade_data.append(generate_data('previous_trade', 2, line['previous_trade'], value_vector[i])[0])
                trade_count_data.append(generate_data('trade_count', 0, line['trade_count'], value_vector[i])[0])
                trade_avg_data.append(generate_data('trade_avg', 0, line['trade_avg'], value_vector[i])[0])
                trade_max_data.append(generate_data('trade_max', 0, line['trade_max'], value_vector[i])[0])
                trade_min_data.append(generate_data('trade_min', 0, line['trade_min'], value_vector[i])[0])
                trade_mid_data.append(generate_data('trade_mid', 0, line['trade_mid'], value_vector[i])[0])
                buy_offer_count_data.append(generate_data('offer_count', 0, line['offer_count'], value_vector[i])[0])
                buy_offer_max_data.append(generate_data('offer_max', 0, line['offer_max'], value_vector[i])[0])
                buy_offer_min_data.append(generate_data('offer_min', 0, line['offer_min'], value_vector[i])[0])
                buy_offer_avg_data.append(generate_data('offer_avg', 0, line['offer_avg'], value_vector[i])[0])
                buy_offer_mid_data.append(generate_data('offer_mid', 0, line['offer_mid'], value_vector[i])[0])
                time_data.append(generate_data('time', 0, line['time'] / last_time, value_vector[i]))
                value_data.append(generate_data('value', 0, value_vector[i], value_vector[i])[0])
                value_ratio.append(generate_data('value_ratio', 0, line['asks'], value_vector[i])[0])
                risk_level.append(generate_data('risk_level', 0, line['asks'], value_vector[i])[0])
                action_data.append([line['behavior1'], line['behavior2']])
                reward_data.append(generate_data('reward', 0, line['reward'], value_vector[i])[0])
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
            unit_data = tf.reshape(unit_data, (-1, 1))
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
            reward_data = tf.convert_to_tensor(reward_data, dtype=tf.float32)
            reward_data = tf.reshape(reward_data, (-1, 1))
            role_data = np.zeros((hist_data.shape[0], 1))

            actions_second_elements = [sublist[1] // 10 for sublist in action_data]
            actions_second_elements = tf.convert_to_tensor(actions_second_elements)
            predicted_regress = model_actor([hist_data, bids_data, asks_data, unit_data, pre_offer_data,
                                              pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
                                              trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
                                              buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
                                              role_data, value_data,reward_data])
            # ,time_data, value_ratio, risk_level])
            # print(predicted_regress)
            # print(actions_second_elements)
            one_hot_labels = tf.one_hot(actions_second_elements, depth=tf.shape(predicted_regress)[-1])
            cross_entropy = -tf.reduce_sum(one_hot_labels * tf.math.log(predicted_regress + 1e-30), axis=-1)
            # loss=tf.reduce_mean(cross_entropy)

            batch_size = tf.shape(cross_entropy)[0]
            weights = tf.exp(tf.linspace(
                start=tf.math.log(0.01),
                stop=tf.math.log(1.0),
                num=batch_size
            ))
            # print("weights")
            # print(weights)
            # print(cross_entropy)

            loss += tf.reduce_sum(cross_entropy * weights)
    if num<5:
        loss = 0.0
    return loss