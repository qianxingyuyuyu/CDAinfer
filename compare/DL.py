import tensorflow as tf
from keras.layers import Input, Concatenate, Dense, Masking
from keras.optimizers import Adam
import json
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gradients_impl import gradients
from sklearn.metrics import mean_squared_error,mean_absolute_error
from training.utils import generate_data
from test import load_user_profile, process_buyer_bounds, generate_sample_cost_dl
import matplotlib.pyplot as plt


def load_and_preprocess_data():
   
    user_profile = load_user_profile('../data_processing/user.csv')
    buyer_data = process_buyer_bounds('../data_processing/bound.jsonl', user_profile)

    
    unit_bounds_dict = {}
    for entry in buyer_data:
        key = (entry["MarketID_period"], entry["userid"])
        unit_bounds_dict[key] = entry["unit_bounds"]

    
    data = []
    data_path = '../data_processing/newnewdata_nn.jsonl'
    with open(data_path, 'r') as file:
        for line in file:
            l = json.loads(line)
            if l['behavior'] < 300 and l['role'] == 'buyer':
                l['cost'] = user_profile[(l['MarketID_period'], l['userid'])][l['unit'] - 1]
                data.append(l)

    
    for l in data:
        key = (l["MarketID_period"], l["userid"])
        unit_idx = l["unit"] - 1
        l["unit_bound"] = unit_bounds_dict[key][unit_idx]

    
    result_dict = {}
    for entry in buyer_data:
        bound = entry['unit_bounds']
        sc = generate_sample_cost_dl(bound)
        if sc:
            new_entry = entry.copy()
            new_entry["sample_cost"] = sc
            key = (new_entry["MarketID_period"], new_entry["userid"])
            result_dict[key] = new_entry["sample_cost"]
        else:
            print(entry)

    
    for l in data:
        key = (l["MarketID_period"], l["userid"])
        if key in result_dict:
            unit_idx = l["unit"] - 1
            l["sample_cost_value"] = result_dict[key][unit_idx]
    #print(data)
    return data



def build_cost_model():
    
    inputs = [
        Input(shape=(15,)),  # history
        Input(shape=(3,)),  # bids
        Input(shape=(3,)),  # asks
        Input(shape=(1,)),  # unit
        Input(shape=(2,)),  # previous_offer
        Input(shape=(2,)),  # previous_trade
        Input(shape=(1,)),  # trade_count
        Input(shape=(1,)),  # trade_max
        Input(shape=(1,)),  # trade_min
        Input(shape=(1,)),  # trade_avg
        Input(shape=(1,)),  # trade_mid
        Input(shape=(1,)),  # offer_count
        Input(shape=(1,)),  # offer_max
        Input(shape=(1,)),  # offer_min
        Input(shape=(1,)),  # offer_avg
        Input(shape=(1,)),  # offer_mid
        Input(shape=(1,)),  # role
        Input(shape=(1,))  # cost
    ]

    
    masked = [Masking(mask_value=-3)(inp) if len(inp.shape) > 1 else inp for inp in inputs[:5]]
    concatenated = Concatenate()(masked + inputs[3:])

    
    x = Dense(32, activation='relu')(concatenated)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(1, activation='linear')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


def build_behavior_model():
    
    inputs = [
        Input(shape=(15,)),  # history
        Input(shape=(3,)),  # bids
        Input(shape=(3,)),  # asks
        Input(shape=(1,)),  # unit
        Input(shape=(2,)),  # previous_offer
        Input(shape=(2,)),  # previous_trade
        Input(shape=(1,)),  # trade_count
        Input(shape=(1,)),  # trade_max
        Input(shape=(1,)),  # trade_min
        Input(shape=(1,)),  # trade_avg
        Input(shape=(1,)),  # trade_mid
        Input(shape=(1,)),  # offer_count
        Input(shape=(1,)),  # offer_max
        Input(shape=(1,)),  # offer_min
        Input(shape=(1,)),  # offer_avg
        Input(shape=(1,)),  # offer_mid
        Input(shape=(1,)),  # role
        Input(shape=(1,))  # behavior
    ]

    
    masked = [Masking(mask_value=-3)(inp) if len(inp.shape) > 1 else inp for inp in inputs[:5]]
    concatenated = Concatenate()(masked + inputs[3:])

    
    x = Dense(32, activation='relu')(concatenated)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    output = Dense(30, activation='softmax')(x)

    return tf.keras.Model(inputs=inputs, outputs=output)


class JointTrainer(tf.keras.Model):
    def __init__(self, cost_model, behavior_model):
        super().__init__()
        self.cost_model = cost_model
        self.behavior_model = behavior_model
        self.cost_optimizer = Adam(0.001)
        self.behavior_optimizer = Adam(0.001)

    def call(self, inputs, training=None, mask=None):
        
        base_features = inputs[:17]  
        cost_feature = inputs[17]  
        behavior_feature = inputs[18]  

        
        cost_inputs = base_features + [cost_feature]  # list + list

        cost_output = self.cost_model(cost_inputs, training=training)

        behavior_inputs = base_features + [behavior_feature]  # list + list

        behavior_output = self.behavior_model(behavior_inputs, training=training)

        return (cost_output, behavior_output)

    def train_step(self, data):
        
        base_features, (sample_cost, behavior_target, unit_bounds) = data

        
        sample_cost = tf.convert_to_tensor(sample_cost, dtype=tf.float32)
        behavior_target = tf.convert_to_tensor(behavior_target, dtype=tf.float32)
        unit_bounds = tf.cast(unit_bounds, tf.float32)

        
        with tf.GradientTape() as cost_tape1:  
            
            cost_inputs = list(base_features) + [tf.reshape(sample_cost, (-1, 1))]+[tf.reshape(tf.cast(behavior_target, tf.float32), (-1, 1))]

            y_pred_cost,_ = self.call(cost_inputs, training=True)
            y_pred_cost = tf.reshape(y_pred_cost, [-1])

            unit_bounds = tf.reshape(unit_bounds, [-1])
            lower_violation = tf.nn.relu(unit_bounds - y_pred_cost)
            upper_violation = tf.nn.relu(y_pred_cost - 250.0)
            bound_penalty1 = tf.reduce_mean(lower_violation ** 2 + upper_violation ** 2)

        cost_gradients = cost_tape1.gradient(bound_penalty1, self.cost_model.trainable_variables)
        self.cost_optimizer.apply_gradients(zip(cost_gradients, self.cost_model.trainable_variables))

        
        with tf.GradientTape() as behavior_tape_pred:
            
            behavior_inputs_pred = list(base_features) + [tf.reshape(y_pred_cost / 150, (-1, 1))]+ [tf.reshape(tf.cast(behavior_target, tf.float32), (-1, 1))]
            _,y_pred_behavior = self.call(behavior_inputs_pred, training=True)
            loss_pred2 = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    behavior_target, y_pred_behavior))

        gradients_pred = behavior_tape_pred.gradient(loss_pred2, self.behavior_model.trainable_variables)
        self.behavior_optimizer.apply_gradients(zip(gradients_pred, self.behavior_model.trainable_variables))

       
        with tf.GradientTape() as behavior_tape_real:
            behavior_inputs_real = list(base_features) + [tf.reshape(sample_cost, (-1, 1))]+ [tf.reshape(tf.cast(behavior_target, tf.float32), (-1, 1))]
            _,y_real_behavior = self.call(behavior_inputs_real, training=True)
            loss_real3 = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    behavior_target, y_real_behavior))

        gradients_real = behavior_tape_real.gradient(loss_real3, self.behavior_model.trainable_variables)
        self.behavior_optimizer.apply_gradients(zip(gradients_real, self.behavior_model.trainable_variables))

        
        with tf.GradientTape() as cost_tape4:  
            
            behavior_labels = tf.argmax(y_real_behavior, axis=1)
            enhanced_inputs = list(base_features) + [tf.reshape(sample_cost, (-1, 1))] + [tf.reshape(tf.cast(behavior_labels, tf.float32), (-1, 1))]
            y_final_cost,_ = self.call(enhanced_inputs, training=True)
            y_final_cost = tf.reshape(y_final_cost, [-1])

            
            lower_violation = tf.nn.relu(unit_bounds - y_final_cost)
            upper_violation = tf.nn.relu(y_final_cost - 250.0)
            bound_penalty4 = tf.reduce_mean(lower_violation ** 2 + upper_violation ** 2)

        final_gradients = cost_tape4.gradient(bound_penalty4, self.cost_model.trainable_variables)
        self.cost_optimizer.apply_gradients(zip(final_gradients, self.cost_model.trainable_variables))

        
        return {
            "cost_loss_step1": bound_penalty1,
            "behavior_loss_pred": loss_pred2,
            "behavior_loss_real": loss_real3,
            "cost_loss_step4": bound_penalty4,
        }

def create_dataset(features, sample_cost, labels, original_data, batch_size=128):
    
    unit_bounds = [item['unit_bound'] for item in original_data]
    
    dataset = tf.data.Dataset.from_tensor_slices(
        (tuple(features), 
         (sample_cost,labels, unit_bounds)))  
    return dataset.batch(batch_size)


if __name__ == "__main__":
    
    data = load_and_preprocess_data()

    
    train_data, other_data = train_test_split(data, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(other_data, test_size=0.5, random_state=42)
    print(train_data[0])
    
    feature_names = [
        'history', 'bids', 'asks', 'unit', 'previous_offer', 'previous_trade',
        'trade_count', 'trade_max', 'trade_min', 'trade_avg', 'trade_mid',
        'offer_count', 'offer_max', 'offer_min', 'offer_avg', 'offer_mid',
        'role', 'behavior', 'sample_cost_value'
    ]

    X_train_history, X_val_history, X_test_history = generate_data('history', train_data, val_data, test_data)
    X_train_bids, X_val_bids, X_test_bids = generate_data('bids', train_data, val_data, test_data)
    X_train_asks, X_val_asks, X_test_asks = generate_data('asks', train_data, val_data, test_data)
    X_train_unit, X_val_unit, X_test_unit = generate_data('unit', train_data, val_data, test_data)
    X_train_previous_offer, X_val_previous_offer, X_test_previous_offer = generate_data('previous_offer', train_data,
                                                                                        val_data, test_data)
    X_train_previous_trade, X_val_previous_trade, X_test_previous_trade = generate_data('previous_trade', train_data,
                                                                                        val_data, test_data)
    X_train_trade_count, X_val_trade_count, X_test_trade_count = generate_data('trade_count', train_data, val_data,
                                                                               test_data)
    X_train_trade_max, X_val_trade_max, X_test_trade_max = generate_data('trade_max', train_data, val_data, test_data)
    X_train_trade_min, X_val_trade_min, X_test_trade_min = generate_data('trade_min', train_data, val_data, test_data)
    X_train_trade_avg, X_val_trade_avg, X_test_trade_avg = generate_data('trade_avg', train_data, val_data, test_data)
    X_train_trade_mid, X_val_trade_mid, X_test_trade_mid = generate_data('trade_mid', train_data, val_data, test_data)
    X_train_offer_count, X_val_offer_count, X_test_offer_count = generate_data('offer_count', train_data, val_data,
                                                                               test_data)
    X_train_offer_max, X_val_offer_max, X_test_offer_max = generate_data('offer_max', train_data, val_data, test_data)
    X_train_offer_min, X_val_offer_min, X_test_offer_min = generate_data('offer_min', train_data, val_data, test_data)
    X_train_offer_avg, X_val_offer_avg, X_test_offer_avg = generate_data('offer_avg', train_data, val_data, test_data)
    X_train_offer_mid, X_val_offer_mid, X_test_offer_mid = generate_data('offer_mid', train_data, val_data, test_data)
    X_train_role, X_val_role, X_test_role = generate_data('role', train_data, val_data, test_data)
    X_train_cost, X_val_cost, X_test_cost = generate_data('sample_cost_value', train_data, val_data, test_data)

    y_train_behavior, y_val_behavior, y_test_behavior = generate_data('behavior', train_data, val_data, test_data)
    y_regression_train, y_regression_val, y_regression_test = generate_data('cost', train_data, val_data, test_data)

    train_ds = create_dataset(
        [X_train_history, X_train_bids, X_train_asks, X_train_unit, X_train_previous_offer, X_train_previous_trade,
         X_train_trade_count, X_train_trade_max, X_train_trade_min, X_train_trade_avg,
         X_train_trade_mid, X_train_offer_count, X_train_offer_max, X_train_offer_min,
         X_train_offer_avg, X_train_offer_mid, X_train_role], X_train_cost,y_train_behavior, train_data)
    val_ds = create_dataset([X_val_history, X_val_bids, X_val_asks, X_val_unit, X_val_previous_offer,
                                  X_val_previous_trade, X_val_trade_count, X_val_trade_max,
                                  X_val_trade_min, X_val_trade_avg, X_val_trade_mid, X_val_offer_count,
                                  X_val_offer_max, X_val_offer_min, X_val_offer_avg, X_val_offer_mid, X_val_role], X_val_cost, y_val_behavior, val_data)
    test_ds= create_dataset([X_test_history, X_test_bids, X_test_asks, X_test_unit, X_test_previous_offer,
                                    X_test_previous_trade, X_test_trade_count, X_test_trade_max,
                                    X_test_trade_min, X_test_trade_avg, X_test_trade_mid, X_test_offer_count,
                                    X_test_offer_max, X_test_offer_min, X_test_offer_avg, X_test_offer_mid, X_test_role],X_test_cost,y_test_behavior,test_data)

    
    cost_model = build_cost_model()
    behavior_model = build_behavior_model()
    joint_trainer = JointTrainer(cost_model, behavior_model)

    
    joint_trainer.compile()
    history = joint_trainer.fit(
        train_ds,
        epochs=20
        #validation_data=val_ds
    )

    y_pred1,_=joint_trainer.call([X_test_history, X_test_bids, X_test_asks, X_test_unit, X_test_previous_offer,
            X_test_previous_trade, X_test_trade_count, X_test_trade_max, X_test_trade_min, X_test_trade_avg,
            X_test_trade_mid, X_test_offer_count, X_test_offer_max, X_test_offer_min,
            X_test_offer_avg, X_test_offer_mid, X_test_role,X_test_cost,y_test_behavior], training=False)
    
    y_regression_test = np.array(y_regression_test)
    y_pred1 = np.array(y_pred1).flatten()  
    
    # test_mse = mean_squared_error(y_regression_test, y_pred1)
    # print(f"\nTest MSE: {test_mse:.4f}")
    
    test_mae = mean_absolute_error(y_regression_test, y_pred1)
    print(f"\nTest MAE: {test_mae:.4f}")
