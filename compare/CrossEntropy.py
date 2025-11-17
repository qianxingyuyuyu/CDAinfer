import tensorflow as tf
from keras.layers import Input, LSTM, Concatenate, Dense,Dropout,Masking, BatchNormalization
from keras.optimizers import Adam
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
from training.utils import generate_data
from collections import defaultdict
import csv
import pandas as pd
from keras.models import load_model
from test import  load_user_profile, process_buyer_bounds, filter_buyer_data, calculate_global_mse, optimize_parameters, generate_sample_cost, get_user_unit

user_profile = load_user_profile('../data_processing/user.csv')
buyer_data = process_buyer_bounds('../data_processing/bound.jsonl', user_profile)

result = []
for entry in buyer_data:
    bound = entry['unit_bounds']
    sc = generate_sample_cost(bound)
    if sc: 
        new_entry = entry.copy()
        new_entry["sample_cost"] = sc
        result.append(new_entry)
result_dict = {(item["MarketID_period"], item["userid"]): item["sample_cost"] for item in result}
#print(result_dict)
data = []
count=0
data_path='../data_processing/newnewdata_nn.jsonl'
with open(data_path, 'r') as file:
    for line in file:
        l=json.loads(line)
        if l['behavior']<300 and l['role']=='buyer':
            
            key = (l["MarketID_period"], l["userid"])
            
            if key in result_dict:
                sc_list = result_dict[key]
                unit_idx = l["unit"] - 1  
                l["sample_cost_value"] = sc_list[unit_idx]
                count += 1
                data.append(l)

print(count)


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
X_train_cost,X_val_cost,X_test_cost=generate_data('sample_cost_value',train_data,val_data, test_data)

y_regression_train,y_regression_val,y_regression_test=generate_data('behavior',train_data,val_data, test_data)


input_history = Input(shape=(15, ))
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
input_cost=Input(shape=(1,))

masking_layer = Masking(mask_value=-3)

masked_input_history = masking_layer(input_history)
masked_input_bids = masking_layer(input_bids)
masked_input_asks = masking_layer(input_asks)
masked_input_previous_offer=masking_layer(input_previous_offer)
masked_input_previous_trade=masking_layer(input_previous_trade)

concatenated = Concatenate()([input_history, masked_input_previous_offer,masked_input_previous_trade, input_unit, masked_input_bids,masked_input_asks, input_trade_count,input_trade_max,input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,input_role,input_cost])
dense1 = Dense(32, activation='relu')(concatenated)
#dense1=BatchNormalization()(dense1)
dense2 = Dense(16, activation='relu')(dense1)
#dense2=BatchNormalization()(dense2)
dense3 = Dense(8, activation='relu')(dense2)
regression_output = Dense(30, activation='softmax')(dense3)


model = tf.keras.Model(inputs=[input_history, input_bids, input_asks, input_unit,input_previous_offer,
                               input_previous_trade,input_trade_count,input_trade_max,
                               input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,
                               input_role,input_cost],
                       outputs=regression_output)

from keras.callbacks import TensorBoard
tbCallBack = TensorBoard(log_dir='regre_logs',  
                 histogram_freq=1,  
                 write_graph=True,  
                 write_grads=True, 
                 write_images=True,
                 embeddings_freq=0,
                 embeddings_layer_names=None,
                 embeddings_metadata=None)

optimizer = Adam(learning_rate=0.001, amsgrad=True)  # RMSprop
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy')

model.fit([X_train_history, X_train_bids, X_train_asks, X_train_unit, X_train_previous_offer,X_train_previous_trade,
           X_train_trade_count,X_train_trade_max,X_train_trade_min,X_train_trade_avg,
           X_train_trade_mid,X_train_offer_count,X_train_offer_max,X_train_offer_min,
           X_train_offer_avg,X_train_offer_mid,X_train_role,X_train_cost],  y_regression_train, epochs=50, batch_size=128,
          callbacks=[tbCallBack],
          validation_data=([X_val_history, X_val_bids, X_val_asks, X_val_unit,X_val_previous_offer,
                            X_val_previous_trade,X_val_trade_count,X_val_trade_max,
                            X_val_trade_min,X_val_trade_avg,X_val_trade_mid,X_val_offer_count,
                            X_val_offer_max,X_val_offer_min,X_val_offer_avg,X_val_offer_mid,X_val_role,X_val_cost],  
                            y_regression_val))
model.evaluate([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
                X_test_previous_trade,X_test_trade_count,X_test_trade_max,X_test_trade_min,
                X_test_trade_avg,X_test_trade_mid,X_test_offer_count,X_test_offer_max,
                X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role,X_test_cost],  y_regression_test)
# model.save('ce_model_250.h5')
#
# 
# model = load_model('ce_model_250.h5')
optimization_results = []
user_profile = filter_buyer_data(
    '../data_processing/user.csv',
    '../data_processing/inference_market.txt'
)
total_squared_error = 0.0  
total_samples = 0  
for (market_id, user_id), valuecost in user_profile.items():
    print(f"\nOptimizing parameters for Market {market_id}, Buyer {user_id}...")
    user_df = pd.read_csv('../data_processing/user.csv')
    user_df = user_df.dropna()
    query = user_df[
        (user_df['MarketID_period'].astype(str) == market_id) &
        (user_df['userid_profile'].astype(int) == user_id)
        ]
    payoff = query['payoff'].values[0]
    if payoff < -10:
        print(f"market_id={market_id}, user_id={user_id}  payoff={payoff}<10, skipping")
        continue
    unit = get_user_unit(market_id, user_id)
    optimized_params, loss = optimize_parameters(market_id, user_id,model,unit)
    print(f"Optimized parameters for Market {market_id}, Buyer {user_id}: {optimized_params}")
    print(f"Loss for Market {market_id}, Buyer {user_id}: {loss if loss is not None else 'N/A'}")
    if loss != None:
        optimized_array = np.asarray(optimized_params, dtype=np.float32)
        valuecost_array = np.asarray(valuecost[:unit], dtype=np.float32)
        if optimized_array.shape != valuecost_array.shape:
            raise ValueError(f"Shape mismatch: {optimized_array.shape} vs {valuecost_array.shape}")
        squared_error = np.sum((optimized_array - valuecost_array) ** 2)
        total_squared_error += squared_error
        total_samples += optimized_array.size
        mse = squared_error / optimized_array.size 
        print(f"MSE for Market {market_id}, Buyer {user_id}: {mse:.4f}")
global_mse = total_squared_error / total_samples
print(f"Global MSE: {global_mse:.4f}")

optimization_results = []
user_profile = filter_buyer_data(
    '../data_processing/user.csv',
    '../data_processing/inference_market.txt'
)
total_absolute_error = 0.0  
total_samples = 0  
for (market_id, user_id), valuecost in user_profile.items():
    print(f"\nOptimizing parameters for Market {market_id}, Buyer {user_id}...")
    user_df = pd.read_csv('../data_processing/user.csv')
    user_df = user_df.dropna()
    query = user_df[
        (user_df['MarketID_period'].astype(str) == market_id) &
        (user_df['userid_profile'].astype(int) == user_id)
        ]
    payoff = query['payoff'].values[0]
    if payoff < -10:
        print(f"market_id={market_id}, user_id={user_id} payoff={payoff}<10, skipping")
        continue
    unit = get_user_unit(market_id, user_id)
    optimized_params, loss = optimize_parameters(market_id, user_id,model,unit)
    print(f"Optimized parameters for Market {market_id}, Buyer {user_id}: {optimized_params}")
    print(f"Loss for Market {market_id}, Buyer {user_id}: {loss if loss is not None else 'N/A'}")
    if loss != None:
        optimized_array = np.asarray(optimized_params, dtype=np.float32)
        valuecost_array = np.asarray(valuecost[:unit], dtype=np.float32)
        if optimized_array.shape != valuecost_array.shape:
            raise ValueError(f"Shape mismatch: {optimized_array.shape} vs {valuecost_array.shape}")
        absolute_error = np.sum(np.abs(optimized_array - valuecost_array))  
        total_absolute_error += absolute_error
        total_samples += optimized_array.size
        mae = absolute_error / optimized_array.size  
        print(f"MAE for Market {market_id}, Buyer {user_id}: {mae:.4f}")
global_mae = total_absolute_error / total_samples  
print(f"Global MAE: {global_mae:.4f}")

