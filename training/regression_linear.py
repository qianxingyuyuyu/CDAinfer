import tensorflow as tf
from keras.layers import Input, LSTM, Concatenate, Dense,Dropout,Masking, BatchNormalization
from keras.optimizers import Adam
import json
import random
from sklearn.model_selection import train_test_split
import numpy as np
from utils import padding,generate_data

data = []
count=0
data_path='../data_processing/newnewdata_nn.jsonl'
with open(data_path, 'r') as file:
    for line in file:
        try:
            l=json.loads(line)
            if l['behavior']<300:
                    count += 1
                    data.append(l)
        except:
            print('error')
print(count)


np.random.seed(42)
random.seed(42)  
train_data, other_data= train_test_split(data, test_size=0.2,random_state=42)  
val_data, test_data = train_test_split(other_data, test_size=0.5,random_state=42) 

# generate_data '../saved_model' behavior
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

concatenated = Concatenate()([input_history, masked_input_previous_offer,masked_input_previous_trade, input_unit, masked_input_bids,masked_input_asks, input_trade_count,input_trade_max,input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,input_role])
dense1 = Dense(32, activation='relu')(concatenated)
#dense1=BatchNormalization()(dense1)
dense2 = Dense(16, activation='relu')(dense1)
#dense2=BatchNormalization()(dense2)
dense3 = Dense(8, activation='relu')(dense2)
regression_output = Dense(1, activation='linear', name='regression_output')(dense3)


model = tf.keras.Model(inputs=[input_history, input_bids, input_asks, input_unit,input_previous_offer,
                               input_previous_trade,input_trade_count,input_trade_max,
                               input_trade_min,input_trade_avg,input_trade_mid,
                               input_offer_count,input_offer_max,input_offer_min,input_offer_avg,input_offer_mid,
                               input_role],
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

model.compile(optimizer=Adam(lr=0.0005), loss='mean_squared_error')
model.fit([X_train_history, X_train_bids, X_train_asks, X_train_unit, X_train_previous_offer,X_train_previous_trade,
           X_train_trade_count,X_train_trade_max,X_train_trade_min,X_train_trade_avg,
           X_train_trade_mid,X_train_offer_count,X_train_offer_max,X_train_offer_min,
           X_train_offer_avg,X_train_offer_mid,X_train_role],  y_regression_train, epochs=100, batch_size=128,
          callbacks=[tbCallBack],
          validation_data=([X_val_history, X_val_bids, X_val_asks, X_val_unit,X_val_previous_offer,
                            X_val_previous_trade,X_val_trade_count,X_val_trade_max,
                            X_val_trade_min,X_val_trade_avg,X_val_trade_mid,X_val_offer_count,
                            X_val_offer_max,X_val_offer_min,X_val_offer_avg,X_val_offer_mid,X_val_role],
                            y_regression_val))
model.evaluate([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
                X_test_previous_trade,X_test_trade_count,X_test_trade_max,X_test_trade_min,
                X_test_trade_avg,X_test_trade_mid,X_test_offer_count,X_test_offer_max,
                X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role],  y_regression_test)

model.save('../saved_model/regression_model_new.h5')

from keras.models import load_model
#model = load_model('../saved_model/regression_model.h5')
y_pred1=model.predict([X_test_history, X_test_bids, X_test_asks, X_test_unit,X_test_previous_offer,
                       X_test_previous_trade,X_test_trade_count,X_test_trade_max,
                       X_test_trade_min,X_test_trade_avg,X_test_trade_mid,X_test_offer_count,
                       X_test_offer_max,X_test_offer_min,X_test_offer_avg,X_test_offer_mid,X_test_role])
y_regression_test = np.array(y_regression_test)
y_pred1 = np.array(y_pred1)


x_axis = np.arange(len(y_regression_test))
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

plt.plot(x_axis, y_regression_test, label='True Values', color='blue')

plt.plot(x_axis, y_pred1, label='Predicted Values', color='red', linestyle='--')

plt.legend()

plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Value')

plt.show()

# plt.hist(y_regression_test, bins=20, edgecolor='black', density=True)
# plt.title('offer_hist')
# plt.xlabel('offer')
# plt.ylabel('Probability density')
# plt.show()
#
# plt.hist(y_pred1_, bins=20, edgecolor='black', density=True)
# plt.title('pred_hist')
# plt.xlabel('offer')
# plt.ylabel('Probability density')
# plt.show()

