import heapq
from sklearn.preprocessing import StandardScaler
import numpy as np
import keras.backend as K
import tensorflow as tf
import os
import pandas as pd

def padding(name,length,data):
    padded=[]
    for entry in data:
        if name in ['history','previous_offer','previous_trade']:
            temp=[]
            for h in entry[name]:
                if h==18401:
                    #temp.append(310)
                    temp.append(-3)
                elif h==18402:
                    temp.append(311)
                else:
                    temp.append(h)
            if len(temp)<=length:
                new = [-3] * (length - len(temp))
                padded.append(new+temp)
            else:
                padded.append(temp[-length:])

        elif name=='bids':
            key_list = list(entry[name].values())
            key_list.sort()
            if len(key_list)<length:
                new = [-3] * (length - len(key_list))
                padded.append(key_list+ new)
            elif len(key_list)>length:
                padded.append(key_list[-3:])
            else:
                padded.append(key_list)
        elif name=='asks':
            key_list = list(entry[name].values())
            key_list.sort()
            if len(key_list) < length:
                new = [-3] * (length - len(key_list))
                padded.append(key_list + new)
            elif len(key_list) > length:
                padded.append(key_list[0:3])
            else:
                padded.append(key_list)
    return padded


def replace_padding_with_nan(data):
    mask = (data == -3)
    data[mask] = np.nan
    return data


def standardize_with_nan(data, scaler, fit=False):
    mask = ~np.isnan(data)
    if fit:
        scaler.fit(data[mask].reshape(-1, 1))
    data[mask] = scaler.transform(data[mask].reshape(-1, 1)).reshape(-1)
    return data


def replace_nan_with_padding(data):
    mask = np.isnan(data)
    data[mask] = -3
    return data

def generate_data(name,train_data,val_data,test_data):
    scaler = StandardScaler()
    #####################################Set the path according to different file directories######################################################
    mean_path = os.path.join('../saved_model', f'{name}_mean.npy')
    std_path = os.path.join('../saved_model', f'{name}_std.npy')
    # mean_path = os.path.join('saved_model', f'{name}_mean.npy')
    # std_path = os.path.join('saved_model', f'{name}_std.npy')

    if name in ['history','bids','asks','previous_offer','previous_trade']:
        if name=='history':
            length=5
        elif name in ['bids','asks']:
            length=3
        else:
            length=2
        X_train=padding(name,length,train_data)
        X_val=padding(name,length,val_data)
        X_test=padding(name,length,test_data)

        X_train = np.array(X_train, dtype=float)
        X_val = np.array(X_val, dtype=float)
        X_test = np.array(X_test, dtype=float)
        #print(X_train[:10])
        X_train = replace_padding_with_nan(X_train)
        X_val = replace_padding_with_nan(X_val)
        X_test = replace_padding_with_nan(X_test)
        #print(X_train[:10])


        if os.path.exists(mean_path) and os.path.exists(std_path):
            print('existing')

            mean = np.load(mean_path)
            std = np.load(std_path)
            scaler.mean_ = mean
            scaler.scale_ = std
            scaler.var_ = std ** 2
            X_train = standardize_with_nan(X_train, scaler)
        else:
            X_train = standardize_with_nan(X_train, scaler, fit=True)
            # np.save('saved_model/' + name + '_mean.npy', mean)
            # np.save('saved_model/' + name + '_std.npy', std)
        X_val = standardize_with_nan(X_val, scaler)
        X_test = standardize_with_nan(X_test, scaler)
        #print(X_train[:10])
        X_train = replace_nan_with_padding(X_train)
        X_val = replace_nan_with_padding(X_val)
        X_test = replace_nan_with_padding(X_test)
        #print(X_train[:10])
        mean = scaler.mean_
        std = scaler.scale_
        print(name)
        print(mean)
        print(std)

    elif name in ['time','eq_price','eq_quantity','optimalconsumersurplus','optimalproducersurplus','optimal_surplus','Number_players','Quantity','ConsumerSurplus','ProducerSurplus','surplus','efficiency']:

        market_stat_data = pd.read_csv('../data_processing/market-stat-data.csv')
        market_stat_data['MarketID_period'] = market_stat_data['MarketID_period'].astype(str)

        train_market_period = [entry['MarketID_period'] for entry in train_data]
        val_market_period = [entry['MarketID_period'] for entry in val_data]
        test_market_period = [entry['MarketID_period'] for entry in test_data]
        if name == 'time':

            market_stat_dict = dict(zip(market_stat_data['MarketID_period'], market_stat_data['offer_time']))

            X_train = [entry[name] / market_stat_dict[train_market_period[i]] for i, entry in enumerate(train_data)]
            X_val = [entry[name] / market_stat_dict[val_market_period[i]] for i, entry in enumerate(val_data)]
            X_test = [entry[name] / market_stat_dict[test_market_period[i]] for i, entry in enumerate(test_data)]
        else:

            market_stat_dict = dict(zip(market_stat_data['MarketID_period'], market_stat_data[name]))

            X_train = [market_stat_dict[train_market_period[i]] for i, entry in enumerate(train_data)]
            X_val = [market_stat_dict[val_market_period[i]] for i, entry in enumerate(val_data)]
            X_test = [market_stat_dict[test_market_period[i]] for i, entry in enumerate(test_data)]

            scaler = StandardScaler()
            X_train = np.array(X_train).reshape(-1, 1)
            X_train = scaler.fit_transform(X_train).flatten()
            X_val = np.array(X_val).reshape(-1, 1)
            X_val = scaler.transform(X_val).flatten()
            X_test = np.array(X_test).reshape(-1, 1)
            X_test = scaler.transform(X_test).flatten()
            mean = scaler.mean_
            std = scaler.scale_
            print(name)
            print(mean)
            print(std)
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    else:
        if name=='role':
            X_train = [1 if entry[name] == 'seller' else 0 for entry in train_data]
            X_val = [1 if entry[name] == 'seller' else 0 for entry in val_data]
            X_test = [1 if entry[name] == 'seller' else 0 for entry in test_data]
        else:
            print(name)
            X_train = [entry[name] for entry in train_data]
            X_val = [entry[name] for entry in val_data]
            X_test = [entry[name] for entry in test_data]

            if name in['behavior','class','class_price','unit','sample_cost_value','reward']:
                X_train = np.array(X_train)
                X_val = np.array(X_val)
                X_test = np.array(X_test)
                ################################Depending on the task output, configure whether behaviour//10.###########################################################
                if name=='behavior':
                    # X_train = X_train // 10
                    # X_val = X_val // 10
                    # X_test = X_test// 10
                    X_train = X_train
                    X_val = X_val
                    X_test = X_test
                if name=='sample_cost_value' or name=='reward':
                    X_train = X_train / 150
                    X_val = X_val / 150
                    X_test = X_test / 150
            else:

                if os.path.exists(mean_path) and os.path.exists(std_path):
                    print('existing')

                    mean = np.load(mean_path)
                    std = np.load(std_path)
                    scaler.mean_ = mean
                    scaler.scale_ = std
                    scaler.var_ = std ** 2

                    X_train = np.array(X_train).reshape(-1, 1)
                    X_train = scaler.transform(X_train)
                else:

                    X_train = np.array(X_train).reshape(-1, 1)
                    X_train = scaler.fit_transform(X_train)
                    # np.save('saved_model/' + name + '_mean.npy', mean)
                    # np.save('saved_model/' + name + '_std.npy', std)
                X_val = np.array(X_val).reshape(-1, 1)
                X_val = scaler.transform(X_val)
                X_test = np.array(X_test).reshape(-1, 1)
                X_test = scaler.transform(X_test)
                mean = scaler.mean_
                std = scaler.scale_
                print(name)
                print(mean)
                print(std)
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    return X_train,X_val,X_test

def F1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))
    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0.0:
        return K.cast(0, K.floatx())
    # How many selected items are relevant?
    precision = c1 / (c2 + K.epsilon())
    #tf.print("Precision:", precision)
    # How many relevant items are selected?
    recall = c1 / (c3 + K.epsilon())
    #tf.print("Recall:", recall)
    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall+ K.epsilon())
    return f1_score

def negative_log_likelihood(y_true, y_pred):
    mu = y_pred[0]
    std = y_pred[1]
    mu = tf.cast(mu, tf.float32)
    std = tf.cast(std, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    return -tf.reduce_mean(-0.5 * tf.math.log(2 * np.pi * tf.pow(std, 2))- tf.pow(y_true - mu, 2) / (2 * tf.pow(std, 2)))
