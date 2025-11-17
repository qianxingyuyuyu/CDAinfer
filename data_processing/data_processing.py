import pandas as pd
from collections import deque
import json
import sys
import numpy as np

df=pd.read_csv('transaction.csv')
data = df.dropna()
user=pd.read_csv('user.csv')
user=user.dropna()
price=pd.read_csv('price.csv')
price=price.dropna()

row_count=0
user_count=0
price_count=0
check_count=0

market_id_period_list = []
with open('simulate_market.txt', 'r') as f:
    for line in f:
        market_id = int(line.strip())
        market_id_period_list.append(market_id)
simulate_i=0
simulate_market=market_id_period_list[simulate_i]

market_time={}
with open('time.jsonl', 'r') as f:
    for line in f:
        l=json.loads(line)
        market_time[l['MarketID_period']]=l['time_count']

market={}
market_delete={}
with open("user_unit.jsonl", "r") as f:
    for line in f:
        l = json.loads(line)
        market[l['market']]=l['user_unit']
        flag=0
        for key, value in l['user_unit'].items():
            if value>3:
                market_delete[l['market']]=1
                flag=1
                break
        if flag==0:
            market_delete[l['market']] = 0

while row_count<data.shape[0]:
    row=data.iloc[row_count]
    current_marketID_period=str(row['MarketID_period'])

    history={}
    previous_offer={}
    previous_trade = {}
    user_cost = {}
    user_row = user.iloc[user_count]
    valuecost=user_row['valuecost']
    current_unit=valuecost.count('-')+1
    while user_count<user.shape[0]:
        user_row = user.iloc[user_count]
        marketID_period = str(user_row['MarketID_period'])
        if marketID_period!=current_marketID_period:
            break
        userid=int(user_row['userid_profile'])
        role=user_row['role']
        user_role=(userid,role)
        q = deque(maxlen=20)
        unit=1
        hist={user_role:(unit,q)}
        history.update(hist)
        user_count+=1
        previous_offer[user_role]=[]
        previous_trade[user_role] = []
        user_cost[user_role] = [int(x) for x in user_row['valuecost'].split('-')]

    if (int(current_marketID_period)==simulate_market)|(int(row['eq_price'])>250):
        row_count+=market_time[current_marketID_period]
        simulate_i +=1
        if simulate_i<len(market_id_period_list):
            simulate_market = market_id_period_list[simulate_i]
        continue

    
    asks={}
    bids= {}
    trade=[]
    buyer_offer=[]
    seller_offer=[]
    price_row=price.iloc[price_count]
    while str(price_row['MarketID_period'])!=current_marketID_period:  
        price_count+=1
        price_row=price.iloc[price_count]

    price_time=int(price_row['time'])
    #time_count=0
    while str(row['MarketID_period'])==current_marketID_period:
        #time_count+=1
        userid = int(row['userid'])
        behavior=row['bidask']
        role=row['playerrole'].lower()
        time=int(row['offer_time'])
        if behavior=='MARKET ORDER':
            if row['status']=='EXPIRED':
                row_count+=1
                row = data.iloc[row_count]
                continue
            if row['transactedprice']=='--':
                behavior =18402
            else:
                behavior=int(row['transactedprice'])
        else:
            behavior=eval(behavior)

        row_count += 1
        row = data.iloc[row_count]
        mul_user = {}
        
        while int(row['offer_time'])==time:
            mul_userid = int(row['userid'])
            mul_behavior = row['bidask']
            if mul_behavior == 'MARKET ORDER':
                if row['status'] == 'EXPIRED':
                    row_count += 1
                    row = data.iloc[row_count]
                    continue
                if row['transactedprice'] == '--':
                    mul_behavior = 18402
                else:
                    mul_behavior = int(row['transactedprice'])
            else:
                mul_behavior = eval(mul_behavior)
            mul_role = row['playerrole'].lower()
            mul_user[mul_userid]=(mul_role,mul_behavior)
            row_count += 1
            row = data.iloc[row_count]


        
        if (time>price_time) & (str(price_row['MarketID_period'])==current_marketID_period):
            print(current_marketID_period)
            selleruserid = int(price_row['selleruserid'])
            print(selleruserid)
            if selleruserid in asks.keys():
                del asks[selleruserid]
            old_unit = history[(selleruserid, 'seller')][0]
            history.update({(selleruserid, 'seller'): (old_unit + 1, deque(maxlen=20))})
            buyeruserid = int(price_row['buyeruserid'])
            print(buyeruserid)
            del bids[buyeruserid]
            old_unit = history[(buyeruserid, 'buyer')][0]
            history.update({(buyeruserid, 'buyer'): (old_unit + 1, deque(maxlen=20))})
            price_count += 1
            price_row = price.iloc[price_count]
            price_time = int(price_row['time'])

        
        trade_count = len(trade)
        if trade == []:
            trade_max = -1
            trade_min = -1
            trade_avg = -1
            trade_mid = -1
        else:
            trade_max = max(trade)
            trade_min = min(trade)
            trade_avg = np.mean(trade)
            trade_mid = np.median(trade)
        buyer_offer_count = len(buyer_offer)
        if buyer_offer == []:
            buyer_offer_max = -1
            buyer_offer_min = -1
            buyer_offer_avg = -1
            buyer_offer_mid = -1
        else:
            buyer_offer_max = max(buyer_offer)
            buyer_offer_min = min(buyer_offer)
            buyer_offer_avg = np.mean(buyer_offer)
            buyer_offer_mid = np.median(buyer_offer)
        seller_offer_count = len(seller_offer)
        if seller_offer == []:
            seller_offer_max = -1
            seller_offer_min = -1
            seller_offer_avg = -1
            seller_offer_mid = -1
        else:
            seller_offer_max = max(seller_offer)
            seller_offer_min = min(seller_offer)
            seller_offer_avg = np.mean(seller_offer)
            seller_offer_mid = np.median(seller_offer)

        for user_role, unit_hist in history.items():
            if current_unit < unit_hist[0]:  
                continue
            if userid!=user_role[0]:  
                if len(unit_hist[1]) == 0:
                    behave=18401  
                else:
                    behave = unit_hist[1][-1]  
            elif user_role[0] in mul_user.keys():
                behave=mul_user[user_role[0]][1]
            else:
                behave=behavior
            if user_role[1]=='buyer':
                offer_count =buyer_offer_count
                offer_max=buyer_offer_max
                offer_min=buyer_offer_min
                offer_avg=buyer_offer_avg
                offer_mid=buyer_offer_mid
            else:
                offer_count = seller_offer_count
                offer_max=seller_offer_max
                offer_min=seller_offer_min
                offer_avg=seller_offer_avg
                offer_mid=seller_offer_mid

            
            item_info = {
                "MarketID_period": current_marketID_period,
                "time": time,
                "userid": user_role[0],
                "role": user_role[1],
                "unit": unit_hist[0],
                "history": list(unit_hist[1]),
                "previous_offer": previous_offer[user_role],
                "previous_trade":previous_trade[user_role],
                "trade_count": trade_count,
                "trade_max": trade_max,
                "trade_min": trade_min,
                "trade_avg": trade_avg,
                "trade_mid": trade_mid,
                "offer_count": offer_count,
                "offer_max": offer_max,
                "offer_min": offer_min,
                "offer_avg": offer_avg,
                "offer_mid": offer_mid,
                "bids": bids,
                "asks": asks,
                "behavior": behave,
                "cost": user_cost[user_role][unit_hist[0]-1],
            }
            
            history[user_role][1].append(behave)

            flag=0
            
            if market_delete[item_info['MarketID_period']] == 1:
                flag=1
            
            if (item_info['behavior'] != 18401) & (item_info['behavior'] != 18402):
                if item_info['behavior'] > 300:
                    flag=1
            
            if (len(item_info['history']) < 5) & (item_info['unit'] == 1):
                flag=1
            
            if item_info['unit'] > market[item_info['MarketID_period']][str(item_info['userid'])]:
                flag=1
            if item_info['history'] == []:
                if item_info['behavior'] == 18401:
                    item_info['class'] = 0
                else:
                    item_info['class'] = 1
            else:
                if item_info['behavior'] == item_info['history'][-1]:
                    item_info['class'] = 0
                else:
                    item_info['class'] = 1
            if flag==0:
                
                with open('newnewdata_nn.jsonl', mode='a+', encoding='utf-8') as fout:
                    json.dump(item_info, fout, ensure_ascii=False)
                    fout.write('\n')
                    sys.stdout.flush()
            if item_info['class'] == 1:
                if item_info['role'] == 'buyer':
                    buyer_offer.append(item_info['behavior'])
                else:
                    seller_offer.append(item_info['behavior'])

        
        if time == price_time:
            mul_buyer=[]
            mul_seller=[]
            mul_prices=[]
            while(time ==price_time):
                mul_buyer.append(int(price_row['buyeruserid']))
                mul_seller.append(int(price_row['selleruserid']))
                mul_prices.append(int(price_row['price']))
                price_count += 1
                price_row = price.iloc[price_count]
                price_time = int(price_row['time'])

            for buyer in mul_buyer:
                old_unit = history[(buyer, 'buyer')][0]
                previous_offer[(buyer, 'buyer')].append(history[(buyer, 'buyer')][1][-1])
                index = next((i for i, name in enumerate(mul_buyer) if name == buyer), None)
                previous_trade[(buyer, 'buyer')].append(mul_prices[index])
                history.update({(buyer, 'buyer'): (old_unit + 1, deque(maxlen=20))})
                if buyer in bids.keys():
                    del bids[buyer]

            for seller in mul_seller:
                old_unit = history[(seller, 'seller')][0]
                previous_offer[(seller, 'seller')].append(history[(seller, 'seller')][1][-1])
                index = next((i for i, name in enumerate(mul_seller) if name == seller), None)
                previous_trade[(seller, 'seller')].append(mul_prices[index])
                history.update({(seller, 'seller'): (old_unit + 1, deque(maxlen=20))})
                if seller in asks.keys():
                    del asks[seller]

            for trade_price in mul_prices:
                trade.append(trade_price)

            if role=='buyer':
                if userid not in mul_buyer:
                    bids[userid]=behavior
            else:
                if userid not in mul_seller:
                    asks[userid]=behavior

            for mul_userid,role_behave in mul_user.items():
                if role_behave[0]=='buyer':
                    if mul_userid not in mul_buyer:
                        bids[mul_userid]=behavior
                else:
                    if mul_userid not in mul_seller:
                        asks[mul_userid] = behavior

        else:
            if role=='buyer':
                bids[userid]=behavior
            else:
                asks[userid]=behavior



    
    # check_row = check.iloc[check_count]
    # asks_count=0
    # bids_count=0
    # while str(check_row['MarketID_period'])==current_marketID_period:
    #     if check_row['buysell']=='Sell':
    #         asks_count+=1
    #     else:
    #         bids_count+=1
    #     check_count+=1
    #     check_row = check.iloc[check_count]
    # if (asks_count!=len(list(asks)) ) | (bids_count!=len(list(bids))):
    #     print('error!!!!!!!!')
    #     print(current_marketID_period)

    # time_info = {
    #     "MarketID_period": current_marketID_period,
    #     "time_count": time_count,
    # }
    # with open('time.jsonl', mode='a+', encoding='utf-8') as fout:
    #     json.dump(time_info, fout, ensure_ascii=False)
    #     fout.write('\n')
    #     sys.stdout.flush()



