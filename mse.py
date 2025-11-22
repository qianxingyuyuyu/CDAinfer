from utils import load_user_profile, optimize_parameters, get_user_unit
from keras.models import load_model
import numpy as np
import pandas as pd
import logging
import argparse


parser = argparse.ArgumentParser(description='Run model optimization with custom parameters.')
parser.add_argument('--logfile', type=str,
                    help='Filename for the output log')
parser.add_argument('--modelfile', type=str,
                    help='Filename for the model file')
args = parser.parse_args()


log_filename = args.logfile
model_filename = args.modelfile

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',  # Simpler format without timestamps
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

user_profile = load_user_profile('data_processing/user.csv', 'data_processing/inference_market.txt')
model = load_model(model_filename)
total_squared_error = 0.0
total_samples = 0

for (market_id, user_id), valuecost in user_profile.items():
    logging.info(f"\nOptimizing parameters for Market {market_id}, Buyer {user_id}...")
    user_df = pd.read_csv('data_processing/user.csv')
    user_df = user_df.dropna()
    query = user_df[
        (user_df['MarketID_period'].astype(str) == market_id) &
        (user_df['userid_profile'].astype(int) == user_id)
        ]
    payoff = query['payoff'].values[0]
    if payoff < -10:
        logging.info(f"market_id={market_id}, user_id={user_id} payoff={payoff}<10, skipping")
        continue
    unit = get_user_unit(market_id, user_id)
    optimized_params, loss = optimize_parameters(market_id, user_id, model, unit)
    logging.info(f"Optimized parameters for Market {market_id}, Buyer {user_id}: {optimized_params}")
    logging.info(f"Loss for Market {market_id}, Buyer {user_id}: {loss if loss is not None else 'N/A'}")
    if loss != None:
        optimized_array = np.asarray(optimized_params, dtype=np.float32)
        valuecost_array = np.asarray(valuecost[:unit], dtype=np.float32)
        if optimized_array.shape != valuecost_array.shape:
            raise ValueError(f"Shape mismatch: {optimized_array.shape} vs {valuecost_array.shape}")
        squared_error = np.sum((optimized_array - valuecost_array) ** 2)
        total_squared_error += squared_error
        total_samples += optimized_array.size
        mse = squared_error / optimized_array.size
        logging.info(f"MSE for Market {market_id}, Buyer {user_id}: {mse:.4f}")

global_mse = total_squared_error / total_samples
logging.info(f"\nGlobal MSE: {global_mse:.4f}")

# python mse.py --logfile custom_log.txt --modelfile custom_model.h5
