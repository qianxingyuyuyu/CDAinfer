import pandas as pd
import numpy as np
import json
import random
from collections import deque
from keras.models import load_model
from training.utils import F1_score
from utils import initialize_market, MarketEnvironment, find_time_count


model_classification = load_model('saved_model/classification_model.h5', custom_objects={'F1_score': F1_score})
model_classification.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score])
model_regression = load_model('saved_model/regression_model_new.h5')


with open('data_processing/simulate_market.txt', 'r') as f:
    markets = [int(line.strip()) for line in f]


results = {
    'market_id': [],
    'buyer_prices': [],
    'buyer_price_mean': [],
    'buyer_price_median': [],
    'buyer_price_std': [],
    'buyer_offer_count': [],
    #'buyer_change_count': [],
    'seller_prices': [],
    'seller_price_mean': [],
    'seller_price_median': [],
    'seller_price_std': [],
    'seller_offer_count': [],
    #'seller_change_count': [],
    'trade_prices': [],
    'trade_price_mean': [],
    'trade_price_median': [],
    'trade_price_std': [],
    'trade_count': []
}


def simulate_market(market_id, total_time):
    
    buyers, history, pre_offer, pre_trade, sellers, current_unit, bids, asks, transactions, buy_offers, sell_offers = initialize_market(
        market_id, initial_time=5
    )

    
    env = MarketEnvironment(
        buyers=buyers,
        sellers=sellers,
        model_classification=model_classification,
        model_classification_price=None,  
        model_regression=model_regression,
        current_unit=current_unit,
        bids=bids,
        asks=asks,
        transactions=transactions,
        buy_offers=buy_offers,
        sell_offers=sell_offers,
        env_history=history,
        env_pre_offer=pre_offer,
        env_pre_trade=pre_trade,
        userid=None,  
        time=5,  
        total_time=total_time
    )

    
    for t in range(6, total_time + 1):
        
        _, done, _, _ = env.step(price=0, change=0)
        if done:
            break

    
    results['market_id'].append(market_id)
    results['buyer_prices'].append(env.buy_offers.copy())
    results['buyer_price_mean'].append(np.mean(env.buy_offers))
    results['buyer_price_median'].append(np.median(env.buy_offers))
    results['buyer_price_std'].append(np.std(env.buy_offers))
    results['buyer_offer_count'].append(len(env.buy_offers))

    
    # buyer_change = sum(1 for buyer in buyers
    #                    if any(history[buyer][1][i]//10 != history[buyer][1][i - 1]//10
    #                           for i in range(1, len(history[buyer][1]))))
    # results['buyer_change_count'].append(buyer_change)

    results['seller_prices'].append(env.sell_offers.copy())
    results['seller_price_mean'].append(np.mean(env.sell_offers))
    results['seller_price_median'].append(np.median(env.sell_offers))
    results['seller_price_std'].append(np.std(env.sell_offers))
    results['seller_offer_count'].append(len(env.sell_offers))

    
    # seller_change = sum(1 for seller in sellers
    #                     if any(history[seller][1][i]//10 != history[seller][1][i - 1]//10
    #                            for i in range(1, len(history[seller][1]))))
    # results['seller_change_count'].append(seller_change)

    results['trade_prices'].append(env.transactions.copy())
    results['trade_price_mean'].append(np.mean(env.transactions) if env.transactions else 0)
    results['trade_price_median'].append(np.median(env.transactions) if env.transactions else 0)
    results['trade_price_std'].append(np.std(env.transactions) if env.transactions else 0)
    results['trade_count'].append(len(env.transactions))

for market_id in markets:
    print(f"Simulating market {market_id}")
    total_time = find_time_count("data_processing/time.jsonl", str(market_id))
    print(f"Total time steps for market {market_id}: {total_time}")
    simulate_market(market_id, total_time)

results_df = pd.DataFrame(results)
results_df.to_pickle('market_simulation_results.pkl')
results_df.to_csv('market_simulation_results.csv', index=False)

print("Simulation completed. Results saved to market_simulation_results.pkl and market_simulation_results.csv")