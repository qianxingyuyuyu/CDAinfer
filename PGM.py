import tensorflow as tf
from keras.layers import Input, LSTM, Concatenate, Dense,Dropout,Masking, BatchNormalization
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
from keras.models import load_model
from training.utils import F1_score,generate_data
from utils import MarketEnvironment, initialize_market, calculate_reward, generate_state_data_for_expert, find_time_count, \
    get_user_unit, load_user_profile, generate_state_data_for_expert
import copy
import os
from skopt.space import Integer
from skopt import Optimizer
import tensorflow_probability as tfp
from collections import Counter
import logging
import pandas as pd

data = []
count=0

behavior_counter = Counter()
data_path='data_processing/inference_data.jsonl'
with open(data_path, 'r') as file:
    for line in file:
        try:
            l=json.loads(line)
            if l['behavior']<300:
                count += 1
                data.append(l)
                
                category = l['behavior'] // 10
                
                behavior_counter[category] += 1
        except:
            print('error')

log_dir = "logs_gail/live_multiple_lists"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


summary_writer = tf.summary.create_file_writer(log_dir)


tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)  
train_data, other_data= train_test_split(data, test_size=0.2,random_state=42)  
val_data, test_data = train_test_split(other_data, test_size=0.5,random_state=42) 


X_train_history,X_val_history,X_test_history=generate_data('history',train_data,val_data, test_data)
X_train_bids,X_val_bids,X_test_bids=generate_data('bids',train_data,val_data, test_data)
X_train_asks,X_val_asks,X_test_asks=generate_data('asks',train_data,val_data, test_data)
X_train_unit,X_val_unit,X_test_unit=generate_data('unit',train_data,val_data, test_data)
X_train_previous_offer,X_val_previous_offer,X_test_previous_offer=generate_data('previous_offer',train_data,val_data, test_data)
X_train_previous_trade,X_val_previous_trade,X_test_previous_trade=generate_data('previous_trade',train_data,val_data, test_data)
X_train_trade_count,X_val_trade_count,X_test_trade_count=generate_data('trade_count',train_data,val_data, test_data)
X_train_trade_max,X_val_trade_max,X_test_trade_max=generate_data('trade_max',train_data,val_data, test_data)
X_train_trade_min,X_val_trade_min,X_test_trade_min=generate_data('trade_min',train_data,val_data, test_data)
X_train_trade_avg,X_val_trade_avg,X_test_trade_avg=generate_data('trade_avg',train_data,val_data, test_data)
X_train_trade_mid,X_val_trade_mid,X_test_trade_mid=generate_data('trade_mid',train_data,val_data, test_data)
X_train_offer_count,X_val_offer_count,X_test_offer_count=generate_data('offer_count',train_data,val_data, test_data)
X_train_offer_max,X_val_offer_max,X_test_offer_max=generate_data('offer_max',train_data,val_data, test_data)
X_train_offer_min,X_val_offer_min,X_test_offer_min=generate_data('offer_min',train_data,val_data, test_data)
X_train_offer_avg,X_val_offer_avg,X_test_offer_avg=generate_data('offer_avg',train_data,val_data, test_data)
X_train_offer_mid,X_val_offer_mid,X_test_offer_mid=generate_data('offer_mid',train_data,val_data, test_data)
X_train_role,X_val_role,X_test_role=generate_data('role',train_data,val_data, test_data)
y_regression_train,y_regression_val,y_regression_test=generate_data('behavior',train_data,val_data, test_data)


input_history = Input(shape=(5, ))
input_bids = Input(shape=(3, ))
input_asks = Input(shape=(3, ))
input_unit = Input(shape=(1,))
input_previous_offer=Input(shape=(2, ))
input_previous_trade=Input(shape=(2, ))
input_trade_count=Input(shape=(1,))
input_trade_max=Input(shape=(1,))
input_trade_min=Input(shape=(1,))
input_trade_avg=Input(shape=(1,))
input_trade_mid=Input(shape=(1,))
input_offer_count=Input(shape=(1,))
input_offer_max=Input(shape=(1,))
input_offer_min=Input(shape=(1,))
input_offer_avg=Input(shape=(1,))
input_offer_mid=Input(shape=(1,))
input_role=Input(shape=(1,))

masking_layer = Masking(mask_value=-3)

masked_input_history = masking_layer(input_history)
masked_input_bids = masking_layer(input_bids)
masked_input_asks = masking_layer(input_asks)
masked_input_previous_offer=masking_layer(input_previous_offer)
masked_input_previous_trade=masking_layer(input_previous_trade)

concatenated = Concatenate()([masked_input_history, masked_input_previous_offer,masked_input_previous_trade, input_unit, masked_input_bids,masked_input_asks, input_trade_count,input_trade_max,input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,input_role])
dense1 = Dense(32, activation='relu')(concatenated)
#dense1=BatchNormalization()(dense1)
dense2 = Dense(16, activation='relu')(dense1)
dense2 = Dense(8, activation='relu')(dense2)
#dense2=BatchNormalization()(dense2)
output = Dense(30, activation='softmax')(dense2)


# expert_model = tf.keras.Model(inputs=[input_history, input_bids, input_asks, input_unit,input_previous_offer,
#                                input_previous_trade,input_trade_count,input_trade_max,
#                                input_trade_min,input_trade_avg,input_trade_mid,
#                                input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,
#                                input_role],
#                        outputs=output)
# print(expert_model.summary())
#
# from keras.callbacks import TensorBoard
# tbCallBack = TensorBoard(log_dir='./regre_logs',  
#                  histogram_freq=1,  
#                  write_graph=True,  
#                  write_grads=True, 
#                  write_images=True,
#                  embeddings_freq=0,
#                  embeddings_layer_names=None,
#                  embeddings_metadata=None)
#
# expert_model.compile(optimizer=Adam(lr=0.0005), loss='sparse_categorical_crossentropy')
# expert_model.fit([X_train_history, X_train_bids, X_train_asks, X_train_unit, X_train_previous_offer,X_train_previous_trade,
#            X_train_trade_count,X_train_trade_max,X_train_trade_min,X_train_trade_avg,
#            X_train_trade_mid,X_train_offer_count,X_train_offer_max,X_train_offer_min,
#            X_train_offer_avg,X_train_offer_mid,X_train_role],  y_regression_train, epochs=1000, batch_size=128,
#           callbacks=[tbCallBack],
#           validation_data=([X_val_history, X_val_bids, X_val_asks, X_val_unit,X_val_previous_offer,
#                             X_val_previous_trade,X_val_trade_count,X_val_trade_max,
#                             X_val_trade_min,X_val_trade_avg,X_val_trade_mid,X_val_offer_count,
#                             X_val_offer_max,X_val_offer_min,X_val_offer_avg,X_val_offer_mid,X_val_role],
#                             y_regression_val))
#
# y_pred1=expert_model.predict([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
#                        X_test_previous_trade,X_test_trade_count,X_test_trade_max,
#                        X_test_trade_min,X_test_trade_avg,X_test_trade_mid,X_test_offer_count,
#                        X_test_offer_max,X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role])
# y_pred_classes = np.argmax(y_pred1, axis=1)
# y_regression_test = np.array(y_regression_test)
# y_pred_classes = np.array(y_pred_classes)
# expert_model.save('saved_model/expert_model.h5')


# x_axis = np.arange(len(y_regression_test))
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 6))
#

# plt.plot(x_axis, y_regression_test, label='True Values', color='blue')
#

# plt.plot(x_axis, y_pred_classes, label='Predicted Values', color='red', linestyle='--')
#

# plt.legend()
#

# plt.title('True vs Predicted Values')
# plt.xlabel('Sample Index')
# plt.ylabel('Value')

# plt.show()

expert_model=load_model('saved_model/expert_model.h5')
y_pred1=expert_model.predict([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
                       X_test_previous_trade,X_test_trade_count,X_test_trade_max,
                       X_test_trade_min,X_test_trade_avg,X_test_trade_mid,X_test_offer_count,
                       X_test_offer_max,X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role])
#print(np.array2string(y_pred1, threshold=np.inf, precision=8, formatter={'float_kind': lambda x: "%.8f" % x}))

y_pred_classes = np.argmax(y_pred1, axis=1)
y_regression_test = np.array(y_regression_test)
y_pred_classes = np.array(y_pred_classes)


x_axis = np.arange(len(y_regression_test))
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))

plt.plot(x_axis, y_regression_test, label='True Values', color='blue')


plt.plot(x_axis, y_pred_classes, label='Predicted Values', color='red', linestyle='--')


plt.legend()


plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')


plt.show()

model_classification = load_model('saved_model/classification_model.h5', custom_objects={'F1_score': F1_score})
model_classification.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score])
model_classification_price=load_model('saved_model/classification_price_model.h5', custom_objects={'F1_score': F1_score})
model_classification_price.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score])
model_regression = load_model('saved_model/regression_model_new.h5')



# data_path = 'data_processing/inference_data.jsonl'
# data = []
# with open(data_path, 'r') as file:
#     for line in file:
#         l = json.loads(line)
#         if (l["MarketID_period"] == current_marketID_period) and (l['userid'] == int(buyer)):
#             if l['behavior'] != 18401:
#                 l['behavior2'] = l['behavior']
#                 l['behavior1'] = l['class']
#                 data.append(l)
# if data != []:
#     time_count = find_time_count(current_marketID_period)
#     hist_data, bids_data, asks_data, unit_data, pre_offer_data, pre_trade_data, trade_count_data, \
#         trade_max_data, trade_min_data, trade_avg_data, trade_mid_data, buy_offer_count_data, \
#         buy_offer_max_data, buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data, action_data, time_data = generate_rl_data(data, 0,time_count)
#     role_data = np.zeros((hist_data.shape[0], 1))
#     actions_second_elements = [sublist[1] // 10 for sublist in action_data]
#     actions_second_elements = tf.convert_to_tensor(actions_second_elements)
#
#     y_pred1 = expert_model.predict([hist_data, bids_data, asks_data, unit_data, pre_offer_data,
#                                     pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
#                                     trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
#                                     buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
#                                     role_data])
#     # print(y_pred1)
#     # print(actions_second_elements)
#
#     y_pred_classes = np.argmax(y_pred1, axis=1)
#     y_regression_test = np.array(actions_second_elements)
#     y_pred_classes = np.array(y_pred_classes)

#     x_axis = np.arange(len(y_regression_test))
#     import matplotlib.pyplot as plt

#     plt.figure(figsize=(10, 6))

#     plt.plot(x_axis, y_regression_test, label='True Values', color='blue')

#     plt.plot(x_axis, y_pred_classes, label='Predicted Values', color='red', linestyle='--')

#     plt.legend()

#     plt.title('True vs Predicted Values')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Value')

#     plt.show()
#
#     expert_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='sparse_categorical_crossentropy')
#
#     expert_model.fit([hist_data, bids_data, asks_data, unit_data, pre_offer_data,
#                                   pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
#                                   trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
#                                   buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
#                                   role_data], actions_second_elements, epochs=100,batch_size=128)
#
#     y_pred1=expert_model.predict([hist_data, bids_data, asks_data, unit_data, pre_offer_data,
#                                       pre_trade_data, trade_count_data, trade_max_data, trade_min_data,
#                                       trade_avg_data, trade_mid_data, buy_offer_count_data, buy_offer_max_data,
#                                       buy_offer_min_data, buy_offer_avg_data, buy_offer_mid_data,
#                                       role_data])
#     # print(y_pred1)
#     # print(actions_second_elements)
#
#     y_pred_classes = np.argmax(y_pred1, axis=1)
#     y_regression_test = np.array(actions_second_elements)
#     y_pred_classes = np.array(y_pred_classes)

#     x_axis = np.arange(len(y_regression_test))
#     plt.figure(figsize=(10, 6))

#     plt.plot(x_axis, y_regression_test, label='True Values', color='blue')

#     plt.plot(x_axis, y_pred_classes, label='Predicted Values', color='red', linestyle='--')

#     plt.legend()

#     plt.title('True vs Predicted Values')
#     plt.xlabel('Sample Index')
#     plt.ylabel('Value')

#     plt.show()

log_filename= 'PGM.txt'
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simpler format without timestamps
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)


def objective(value_vector, current_marketID_period, buyer, expert_model, current_unit):
    target_total_steps = 50
    cost = [value_vector[0], value_vector[1], value_vector[2]]

    total_time = find_time_count("data_processing/time.jsonl", current_marketID_period)

    initial_time_points = np.linspace(total_time // 4, 3 * total_time // 4, num=5, dtype=int).tolist()

    all_state_list = []
    all_action_list = []
    all_reward_list = []
    total_steps = 0

    for initial_time in initial_time_points:
        buyers, whole_history, whole_previous_offer, whole_previous_trade, sellers, current_unit, bids, asks, transactions, buy_offers, sell_offers = initialize_market(
            current_marketID_period, initial_time)

        new_buyers = buyers.copy()
        new_buyers.remove(buyer)
        env_history = copy.deepcopy(whole_history)
        del env_history[buyer]
        env_pre_offer = copy.deepcopy(whole_previous_offer)
        del env_pre_offer[buyer]
        env_pre_trade = copy.deepcopy(whole_previous_trade)
        del env_pre_trade[buyer]

        env = MarketEnvironment(
            new_buyers, sellers, model_classification, model_classification_price, model_regression,
            current_unit, bids.copy(), asks.copy(), transactions.copy(), buy_offers.copy(), sell_offers.copy(),
            env_history.copy(), env_pre_offer.copy(), env_pre_trade.copy(), buyer, initial_time, total_time
        )
        state = copy.deepcopy(env.get_state())
        done = False

        history = list(whole_history[buyer][1]).copy()
        previous_offer = whole_previous_offer[buyer].copy()
        previous_trade = whole_previous_trade[buyer].copy()
        unit = whole_history[buyer][0]

        state_list = []
        action_list = []
        reward_list = []
        time= initial_time
        while not done and unit <= current_unit:
            
            state_data = generate_state_data_for_expert(state, cost, unit, history, previous_offer, previous_trade, time,
                                             total_time)
            states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in state_data]
            probs = expert_model(states)
            action_dist = tfp.distributions.Categorical(probs=probs)
            action = action_dist.sample()
            price = np.random.randint(10 * action.numpy().item(), 10 * action.numpy().item() + 10)
            print(price)
            if history != []:
                if price // 10 == history[-1] // 10:
                    change = 0
                else:
                    change = 1
            else:
                change = 1
            history.append(price)

           
            offer = price
            next_state, done, trade, price = env.step(price, change)
            if total_steps >= target_total_steps:
                break

            state_values = copy.deepcopy(list(state[1].values()))
            next_state_values = copy.deepcopy(list(next_state[1].values()))

            total_reward, reward = calculate_reward(trade, offer, price, cost, unit, state_values, next_state_values)

            time += 1

            if unit > current_unit:
                done = True

            
            state_list.append(states)
            reward_list.append(total_reward)
            action_list.append(offer // 10)
            state = copy.deepcopy(next_state)

            
            total_steps += 1


        
        if state_list:
            all_state_list.extend(state_list)
            all_action_list.extend(action_list)
            all_reward_list.extend(reward_list)

        if total_steps >= target_total_steps:
            break

    if all_reward_list==[]:
        return None

    G = 0
    gamma = 0.81
    with tf.GradientTape() as tape:
        for i in reversed(range(len(all_reward_list))):  
            reward = all_reward_list[i]
            action = tf.convert_to_tensor([all_action_list[i]], dtype=tf.int32)
            action_probs = expert_model(all_state_list[i])
            log_prob = tf.math.log(tf.gather(action_probs, action, axis=1))
            G = gamma * G + reward
            loss = -log_prob * G

        
        total_loss = tf.reduce_sum(loss)
        total_loss = total_loss / len(all_reward_list)

    gradients = tape.gradient(total_loss, expert_model.trainable_variables)
    gradient_norm = tf.linalg.global_norm(gradients)

    #global episode
    # with summary_writer.as_default():
    #     tf.summary.scalar("gradient_norm", gradient_norm, step=episode)
    #     for i, c in enumerate(cost):
    #         tf.summary.scalar(f"cost_{i}", c, step=episode)
    #episode += 1

    return float(gradient_norm.numpy())

def sample_valid_point():
    x0 = np.random.randint(27, 150)
    x1 = np.random.randint(26, x0)
    x2 = np.random.randint(25, x1)
    return [x0, x1, x2]

def optimize_buyer_value(market_id, buyer_id, expert_model, unit):
    
    space = [Integer(25, 250), Integer(25, 250), Integer(25, 250)]
    optimizer = Optimizer(space, n_initial_points=10, random_state=0)
    n_calls = 10
    
    for num in range(n_calls):
        x = sample_valid_point()
        y = objective(x, market_id, str(buyer_id), expert_model, unit)
        if y is None:
            logging.info(f"Skipping optimization for Market {market_id}, Buyer {buyer_id} due to None return.")
            return None, None
        optimizer.tell(x, y)

    res = optimizer.get_result()
    return res.x, res.fun


def main():
    
    user_profile = load_user_profile('data_processing/user.csv', 'data_processing/inference_market.txt')
    expert_model = load_model('saved_model/expert_model.h5')  

    total_squared_error = 0.0
    total_params = 0
    results = []

    for (market_id, buyer_id), true_valuecost in user_profile.items():
        logging.info(f"\nOptimizing parameters for Market {market_id}, Buyer {buyer_id}...")
        
        unit = get_user_unit(market_id, buyer_id)
        user_df = pd.read_csv('data_processing/user.csv')
        user_df = user_df.dropna()
        query = user_df[
            (user_df['MarketID_period'].astype(str) == market_id) &
            (user_df['userid_profile'].astype(int) == buyer_id)
            ]
        payoff = query['payoff'].values[0]
        if payoff < -10:
            logging.info(f"market_id={market_id}, user_id={buyer_id} payoff={payoff}<-10, skipping")
            continue
        
        optimized_params, loss = optimize_buyer_value(
            market_id, buyer_id, expert_model, unit
        )
        if optimized_params is None:
            logging.info(f"Skipping optimization for Market {market_id}, Buyer {buyer_id} due to None return.")
            continue
        logging.info(f"Optimized parameters for Market {market_id}, Buyer {buyer_id}: {optimized_params}")

        
        opt_array = np.array(optimized_params[:unit])
        true_array = np.array(true_valuecost[:unit])
        mse = np.mean((opt_array - true_array) ** 2)

        
        total_squared_error += np.sum((opt_array - true_array) ** 2)
        total_params += unit

        logging.info(f"MSE for Market {market_id}, Buyer {buyer_id}: {mse:.4f}")

    
    global_mse = total_squared_error / total_params if total_params > 0 else 0
    logging.info(f"\nGlobal MSE: {global_mse:.4f}")


if __name__ == "__main__":
    main()