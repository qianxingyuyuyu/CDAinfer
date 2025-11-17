import numpy as np
import pandas as pd
from keras.models import load_model
from test import  filter_buyer_data,  optimize_parameters,  get_user_unit


model = load_model('ce_model4.h5')
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