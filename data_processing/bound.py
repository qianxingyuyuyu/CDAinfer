import json
import csv
from collections import defaultdict


bound_data = defaultdict(list)

with open("newnewdata_nn.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        key = (
            record["MarketID_period"],
            record["userid"],
            record["role"],
            record["unit"]
        )
        if record["behavior"] < 500:
            bound_data[key].append(record["behavior"])

with open("newnewdata_nn_5.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line.strip())
        key = (
            record["MarketID_period"],
            record["userid"],
            record["role"],
            record["unit"]
        )
        if record["behavior"] < 500:
            bound_data[key].append(record["behavior"])


cost_mapping = {}

with open("user.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if not all(row.get(key) for key in ["userid_profile", "role", "MarketID_period", "valuecost"]):
            continue
        key = (
            int(row["userid_profile"]),  
            row["role"],
            row["MarketID_period"]
        )
        
        try:
            valuecost = list(map(int, row["valuecost"].split('-')))
        except:
            valuecost =[-1]
        cost_mapping[key] = valuecost


output_records = []


for key in bound_data:
    market_id_period, userid, role, unit = key

    
    valid_values = bound_data[key]
    if role=='buyer':
        bound = max(valid_values) if valid_values else None
    else:
        bound = min(valid_values) if valid_values else None

    
    cost = None
    csv_key = (userid, role, market_id_period)
    if csv_key in cost_mapping:
        valuecost_list = cost_mapping[csv_key]
        if unit <= len(valuecost_list):
            if valuecost_list==[-1]:
                cost = None
            else:
                cost = valuecost_list[unit-1]
        else:
            print(f"warning: unit={unit} > {len(valuecost_list)} (key={csv_key})")

    output_records.append({
        "MarketID_period": market_id_period,
        "userid": userid,
        "role": role,
        "unit": unit,
        "cost": cost,
        "bound": bound
    })


with open("bound.jsonl", "w", encoding="utf-8") as f:
    json.dump(output_records, f, indent=2)
