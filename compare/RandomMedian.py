import json
from test import load_user_profile, process_buyer_bounds, filter_buyer_data, calculate_global_mse, optimize_parameters, \
    generate_sample_cost, get_user_unit, generate_sample_cost_mean, generate_sample_cost_dl

user_profile = filter_buyer_data(
    '../data_processing/user.csv',
    '../data_processing/inference_market.txt'
)
buyer_data = process_buyer_bounds('../data_processing/bound.jsonl', user_profile)


# total_squared_error = 0.0
# total_samples = 0
# for entry in buyer_data:
#     bound = entry['unit_bounds']
#     market= entry['MarketID_period']
#     buyer= entry['userid']
#     sc = generate_sample_cost(bound)
#     if sc:  
#         unit = get_user_unit(market, buyer)
#         new_entry = entry.copy()
#         new_entry["sample_cost"] = sc[:unit]
#         new_entry["cost"] = new_entry["cost"][:unit]
#         new_entry["unit_bounds"] = entry["unit_bounds"][:unit]
#         squared_errors = [
#             (float(new_entry["sample_cost"][i]) - float(new_entry["cost"][i])) ** 2
#             for i in range(len(new_entry["sample_cost"]))
#         ]
#         total_squared_error += sum(squared_errors)
#         total_samples += len(new_entry["sample_cost"])
#         data = {
#             "MarketID_period": market,
#             "userid": buyer,
#             "sample_cost": new_entry["sample_cost"],
#             "bounds":new_entry["unit_bounds"],
#             "cost": new_entry["cost"]
#         }
#         with open("RM.jsonl", "a", encoding="utf-8") as f:
#             f.write(json.dumps(data, ensure_ascii=False) + "\n")
# global_mse = total_squared_error / total_samples
# print("Random Global MSE:", global_mse)
#
# 
# total_squared_error = 0.0
# total_samples = 0
# for entry in buyer_data:
#     bound = entry['unit_bounds']
#     market= entry['MarketID_period']
#     buyer= entry['userid']
#     sc = generate_sample_cost_mean(bound)
#     if sc:  
#         unit = get_user_unit(market, buyer)
#         new_entry = entry.copy()
#         new_entry["sample_cost"] = sc[:unit]
#         new_entry["cost"] = new_entry["cost"][:unit]
#         new_entry["unit_bounds"] = entry["unit_bounds"][:unit]
#         squared_errors = [
#             (float(new_entry["sample_cost"][i]) - float(new_entry["cost"][i])) ** 2
#             for i in range(len(new_entry["sample_cost"]))
#         ]
#         total_squared_error += sum(squared_errors)
#         total_samples += len(new_entry["sample_cost"])
#         data = {
#             "MarketID_period": market,
#             "userid": buyer,
#             "sample_cost": new_entry["sample_cost"],
#             "bounds": new_entry["unit_bounds"],
#             "cost": new_entry["cost"]
#         }
#         with open("RM.jsonl", "a", encoding="utf-8") as f:
#             f.write(json.dumps(data, ensure_ascii=False) + "\n")
# global_mse = total_squared_error / total_samples
# print("Mean Global MSE:", global_mse)



total_absolute_error = 0.0  
total_samples = 0
for entry in buyer_data:
    bound = entry['unit_bounds']
    market = entry['MarketID_period']
    buyer = entry['userid']
    sc = generate_sample_cost(bound)
    if sc:  
        unit = get_user_unit(market, buyer)
        new_entry = entry.copy()
        new_entry["sample_cost"] = sc[:unit]
        new_entry["cost"] = new_entry["cost"][:unit]
        new_entry["unit_bounds"] = entry["unit_bounds"][:unit]
        absolute_errors = [  
            abs(float(new_entry["sample_cost"][i]) - float(new_entry["cost"][i]))
            for i in range(len(new_entry["sample_cost"]))
        ]
        total_absolute_error += sum(absolute_errors)
        total_samples += len(new_entry["sample_cost"])
        data = {
            "MarketID_period": market,
            "userid": buyer,
            "sample_cost": new_entry["sample_cost"],
            "bounds": new_entry["unit_bounds"],
            "cost": new_entry["cost"]
        }
        with open("RM.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
global_mae = total_absolute_error / total_samples  
print("Random Global MAE:", global_mae)


total_absolute_error = 0.0  
total_samples = 0
for entry in buyer_data:
    bound = entry['unit_bounds']
    market = entry['MarketID_period']
    buyer = entry['userid']
    sc = generate_sample_cost_mean(bound)
    if sc:  
        unit = get_user_unit(market, buyer)
        new_entry = entry.copy()
        new_entry["sample_cost"] = sc[:unit]
        new_entry["cost"] = new_entry["cost"][:unit]
        new_entry["unit_bounds"] = entry["unit_bounds"][:unit]
        absolute_errors = [  
            abs(float(new_entry["sample_cost"][i]) - float(new_entry["cost"][i]))
            for i in range(len(new_entry["sample_cost"]))
        ]
        total_absolute_error += sum(absolute_errors)
        total_samples += len(new_entry["sample_cost"])
        data = {
            "MarketID_period": market,
            "userid": buyer,
            "sample_cost": new_entry["sample_cost"],
            "bounds": new_entry["unit_bounds"],
            "cost": new_entry["cost"]
        }
        with open("RM.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
global_mae = total_absolute_error / total_samples  
print("Mean Global MAE:", global_mae)  