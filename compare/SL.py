import tensorflow as tf
from keras.layers import Input, LSTM, Concatenate, Dense,Dropout,Masking, BatchNormalization
from keras.optimizers import Adam
import json
from sklearn.metrics import mean_squared_error,mean_absolute_error
import random
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from training.utils import generate_data
from test import  load_user_profile, process_buyer_bounds, filter_buyer_data, calculate_global_mse, optimize_parameters, generate_sample_cost, get_user_unit

user_profile = load_user_profile('../data_processing/user.csv')
buyer_data = process_buyer_bounds('../data_processing/bound.jsonl', user_profile)
unit_bounds_dict = {}
for entry in buyer_data:
    key = (entry["MarketID_period"], entry["userid"])
    unit_bounds_dict[key] = entry["unit_bounds"]

data = []
count=0
data_path='../data_processing/newnewdata_nn.jsonl'
with open(data_path, 'r') as file:
    for line in file:
        l=json.loads(line)
        if l['MarketID_period']=='185800101':
            continue
        if l['behavior']<300 and l['role']=='buyer':
                count += 1
                l['cost']=user_profile[(l['MarketID_period'],l['userid'])][l['unit']-1]
                data.append(l)
for l in data:
    key = (l["MarketID_period"], l["userid"])
    unit_idx = l["unit"] - 1
    l["unit_bound"] = unit_bounds_dict[key][unit_idx]

# 划分数据
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
X_train_behavior,X_val_behavior,X_test_behavior=generate_data('behavior',train_data,val_data, test_data)
X_train_role,X_val_role,X_test_role=generate_data('role',train_data,val_data, test_data)

y_regression_train,y_regression_val,y_regression_test=generate_data('cost',train_data,val_data, test_data)


def create_dataset(features, labels, original_data, batch_size=128):
    
    unit_bounds = [item['unit_bound'] for item in original_data]
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (tuple(features),  
         (labels, unit_bounds)))  
    return dataset.batch(batch_size)

train_dataset = create_dataset([X_train_history, X_train_bids, X_train_asks, X_train_unit, X_train_previous_offer,X_train_previous_trade,
           X_train_trade_count,X_train_trade_max,X_train_trade_min,X_train_trade_avg,
           X_train_trade_mid,X_train_offer_count,X_train_offer_max,X_train_offer_min,
           X_train_offer_avg,X_train_offer_mid,X_train_role,X_train_behavior], y_regression_train, train_data)
val_dataset = create_dataset([X_val_history, X_val_bids, X_val_asks, X_val_unit,X_val_previous_offer,
                            X_val_previous_trade,X_val_trade_count,X_val_trade_max,
                            X_val_trade_min,X_val_trade_avg,X_val_trade_mid,X_val_offer_count,
                            X_val_offer_max,X_val_offer_min,X_val_offer_avg,X_val_offer_mid,X_val_role,X_val_behavior], y_regression_val, val_data)



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
input_behavior=Input(shape=(1,))

masking_layer = Masking(mask_value=-3)

masked_input_history = masking_layer(input_history)
masked_input_bids = masking_layer(input_bids)
masked_input_asks = masking_layer(input_asks)
masked_input_previous_offer=masking_layer(input_previous_offer)
masked_input_previous_trade=masking_layer(input_previous_trade)

concatenated = Concatenate()([input_history, masked_input_previous_offer,masked_input_previous_trade, input_unit, masked_input_bids,masked_input_asks, input_trade_count,input_trade_max,input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,input_role,input_behavior])
dense1 = Dense(32, activation='relu')(concatenated)
#dense1=BatchNormalization()(dense1)
dense2 = Dense(16, activation='relu')(dense1)
#dense2=BatchNormalization()(dense2)
dense3 = Dense(8, activation='relu')(dense2)
regression_output = Dense(1, activation='linear', name='regression_output')(dense3)

class CustomTrainer(tf.keras.Model):
    def __init__(self, model, alpha=1.0):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def call(self, inputs, training=None, mask=None):
        return self.model(inputs, training=training)

    def train_step(self, data):
        features, (y_true, unit_bounds) = data

        with tf.GradientTape() as tape:
            y_pred = self.model(features, training=True)
            y_pred = tf.reshape(y_pred, [-1])  

            
            unit_bounds = tf.cast(unit_bounds, dtype=tf.float32)
            unit_bounds = tf.reshape(unit_bounds, [-1])  

            
            lower_violation = tf.nn.relu(unit_bounds - y_pred)  
            lower_penalty = tf.reduce_mean(tf.square(lower_violation))

           
            upper_violation = tf.nn.relu(y_pred - 250.0)  
            upper_penalty = tf.reduce_mean(tf.square(upper_violation))

            
            bound_penalty = lower_penalty + upper_penalty

        
        gradients = tape.gradient(bound_penalty, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"loss": bound_penalty}

model = tf.keras.Model(inputs=[input_history, input_bids, input_asks, input_unit,input_previous_offer,
                               input_previous_trade,input_trade_count,input_trade_max,
                               input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,
                               input_role,input_behavior],
                       outputs=regression_output)

trainer = CustomTrainer(model, alpha=0.5)
trainer.compile(optimizer=Adam(learning_rate=0.001))
history = trainer.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20
)

y_pred1 = model.predict([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
                       X_test_previous_trade,X_test_trade_count,X_test_trade_max,
                       X_test_trade_min,X_test_trade_avg,X_test_trade_mid,X_test_offer_count,
                       X_test_offer_max,X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role,X_test_behavior])


y_regression_test = np.array(y_regression_test)
y_pred1 = np.array(y_pred1).flatten()  


# test_mse = mean_squared_error(y_regression_test, y_pred1)
# print(f"\nTest MSE: {test_mse:.4f}")

test_mae = mean_absolute_error(y_regression_test, y_pred1)
print(f"\nTest MAE: {test_mae:.4f}")