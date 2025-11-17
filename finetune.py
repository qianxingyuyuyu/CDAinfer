import numpy as np
from keras.models import load_model
from training.utils import F1_score
from utils import generate_data,initialize_market,MarketEnvironment,generate_rl_data,find_time_count,ReplayBuffer,generate_state_data,calculate_reward1,process_tensors,smoothness_loss,get_random_time
import tensorflow as tf
from keras.layers import Input, LSTM, Concatenate, Dense,BatchNormalization, Dropout,Masking
import keras
import copy
import random
import tensorflow_probability as tfp
import os
import argparse
import glob

def create_actor_regression_model():
    input_history = Input(shape=(15, 1))
    input_bids = Input(shape=(3,))
    input_asks = Input(shape=(3,))
    input_unit = Input(shape=(1,))
    input_previous_offer = Input(shape=(2,))
    input_previous_trade = Input(shape=(2,))
    input_trade_count = Input(shape=(1,))
    input_trade_max = Input(shape=(1,))
    input_trade_min = Input(shape=(1,))
    input_trade_avg = Input(shape=(1,))
    input_trade_mid = Input(shape=(1,))
    input_offer_count = Input(shape=(1,))
    input_offer_max = Input(shape=(1,))
    input_offer_min = Input(shape=(1,))
    input_offer_avg = Input(shape=(1,))
    input_offer_mid = Input(shape=(1,))
    input_role = Input(shape=(1,))
    input_value = Input(shape=(1,))
    input_time=Input(shape=(1,))
    input_value_ratio=Input(shape=(1,))
    input_risk_level=Input(shape=(1,))
    
    masking_layer = Masking(mask_value=-3)
    masked_input_history = masking_layer(input_history)
    masked_input_bids = masking_layer(input_bids)
    masked_input_asks = masking_layer(input_asks)
    masked_input_previous_offer = masking_layer(input_previous_offer)
    masked_input_previous_trade = masking_layer(input_previous_trade)
    #lstm_out1 = LSTM(4, return_sequences=True)(masked_input_history)
    lstm_out2 = LSTM(8)(masked_input_history)
    #lstm_out3 = LSTM(16)(lstm_out2)
    
    # concatenated = Concatenate()([lstm_out2, masked_input_previous_offer, masked_input_previous_trade,
    #                               input_unit, input_value, input_time, masked_input_bids, masked_input_asks,
    #                               input_trade_max, input_trade_min, input_trade_avg, input_trade_mid,
    #                               input_offer_max, input_offer_min, input_offer_avg, input_offer_mid])
    concatenated = Concatenate()([lstm_out2,input_value, masked_input_asks,input_unit,input_trade_avg,input_value_ratio,input_risk_level,input_time])
    dense1 = Dense(32, activation='relu')(concatenated)
    dense2 = Dense(16, activation='relu')(dense1)
    dense3 = Dense(16, activation='relu')(dense2)
    
    # value_sensitive_layer = Dense(8, activation='relu')(input_value)
    # dense3 = Concatenate()([dense2, value_sensitive_layer])
    
    output = Dense(30, activation='softmax')(dense3)
    model = keras.Model(
        inputs=[input_history, input_bids, input_asks, input_unit, input_previous_offer, input_previous_trade,
                input_trade_count, input_trade_max, input_trade_min, input_trade_avg, input_trade_mid,
                input_offer_count, input_offer_max, input_offer_min, input_offer_avg, input_offer_mid,
                input_role,input_value,input_time,input_value_ratio,input_risk_level],
        outputs=[output])
    print(model.summary())
    return model

def create_critic_model():
    input_history = Input(shape=(15, 1))
    input_bids = Input(shape=(3,))
    input_asks = Input(shape=(3,))
    input_unit = Input(shape=(1,))
    input_previous_offer = Input(shape=(2,))
    input_previous_trade = Input(shape=(2,))
    input_trade_count = Input(shape=(1,))
    input_trade_max = Input(shape=(1,))
    input_trade_min = Input(shape=(1,))
    input_trade_avg = Input(shape=(1,))
    input_trade_mid = Input(shape=(1,))
    input_offer_count = Input(shape=(1,))
    input_offer_max = Input(shape=(1,))
    input_offer_min = Input(shape=(1,))
    input_offer_avg = Input(shape=(1,))
    input_offer_mid = Input(shape=(1,))
    input_role = Input(shape=(1,))
    input_value = Input(shape=(1,))
    input_time=Input(shape=(1,))
    input_value_ratio=Input(shape=(1,))
    input_risk_level=Input(shape=(1,))
    masking_layer = Masking(mask_value=-3)
    masked_input_history = masking_layer(input_history)
    masked_input_bids = masking_layer(input_bids)
    masked_input_asks = masking_layer(input_asks)
    masked_input_previous_offer = masking_layer(input_previous_offer)
    masked_input_previous_trade = masking_layer(input_previous_trade)
    #lstm_out1 = LSTM(8, return_sequences=True)(masked_input_history)
    lstm_out2 = LSTM(8)(masked_input_history)
    #lstm_out3 = LSTM(16)(lstm_out2)
    
    # concatenated = Concatenate()([lstm_out2, masked_input_previous_offer, masked_input_previous_trade,
    #                               input_unit,input_value,input_time,masked_input_bids, masked_input_asks,
    #                               input_trade_max, input_trade_min, input_trade_avg, input_trade_mid,
    #                               input_offer_max, input_offer_min, input_offer_avg, input_offer_mid])
    concatenated = Concatenate()([lstm_out2,input_value, masked_input_asks,input_unit,input_trade_avg,input_value_ratio,input_risk_level,input_time])
    dense1 = Dense(32, activation='relu')(concatenated)
    dense2 = Dense(16, activation='relu')(dense1)
    dense3 = Dense(8, activation='relu')(dense2)
    
    # value_sensitive_layer = Dense(8, activation='relu')(input_value)
    # output = Concatenate()([dense2, value_sensitive_layer])
    output = Dense(1, activation='linear')(dense2)
    
    model = keras.Model(
        inputs=[input_history, input_bids, input_asks, input_unit, input_previous_offer, input_previous_trade,
                input_trade_count, input_trade_max, input_trade_min, input_trade_avg, input_trade_mid,
                input_offer_count, input_offer_max, input_offer_min, input_offer_avg, input_offer_mid,
                input_role,input_value,input_time,input_value_ratio,input_risk_level],
        outputs=[output])
    #print(model.summary())
    return model

class REINFORCE:
    def __init__(self, actor_lr, critic_lr,lmbda, epochs, eps, gamma,risk_gamma,epsilon_start=0.95, epsilon_end=0.05, epsilon_decay_steps=5000):
        self.actor3 = create_actor_regression_model()
        self.actor3 = load_model('actor_model_all_20-250_new_0.5_2.h5', compile=False)
        for layer in self.actor3.layers:
            layer.trainable = True
        self.critic = create_critic_model()
        self.actor_optimizer3 = tf.keras.optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
        self.gamma = gamma
        self.risk_gamma = risk_gamma
        self.lmbda = lmbda
        self.epochs = epochs  
        self.eps = eps  
        self.time=0
        self.epsilon = epsilon_start  
        self.epsilon_min = epsilon_end  
        self.epsilon_decay_steps = epsilon_decay_steps  


    def _epsilon_Greedy(self):
        
        self.epsilon = max(self.epsilon_min, self.epsilon-0.0002)
        return self.epsilon

    def save_model(self):
        self.actor3.save(f"actor_model_risk_gamma_{self.risk_gamma}.h5")
        return

    def take_action3(self, state):
        self._epsilon_Greedy()  
        probs = self.actor3(state,training=True)
        print(probs)
        action_dist = tfp.distributions.Categorical(probs=probs)
        action = action_dist.sample()
        print(action)
        final_action = np.random.randint(10*action.numpy().item(), 10*action.numpy().item()+10)
        return final_action

    def update(self, transition_dict):
        self.time += 1
        feature_num = 21
        states = transition_dict['states']
        states = process_tensors(states, feature_num)
        actions = transition_dict['actions']
        actions_array = np.array(actions)
        scaled_actions = actions_array // 10
        price = tf.cast(scaled_actions, dtype=tf.int32)
        price = tf.reshape(price, (-1, 1))
        #print(price)
        rewards = tf.convert_to_tensor(transition_dict['rewards'], dtype=tf.float32)
        rewards = tf.reshape(rewards, (-1, 1))

        advantage = tf.convert_to_tensor(transition_dict['advantages'], dtype=tf.float32)
        advantage = tf.reshape(advantage, (-1, 1))
        log_dir = f"logs_finetune_gamma_{self.risk_gamma}/live_multiple_lists"
        log_file = os.path.join(log_dir, f"finetune_gamma_{self.risk_gamma}.txt")

        with open(log_file, mode='a+', encoding='utf-8') as fout:
            
            price_flat = [int(x[0]) for x in price.numpy().tolist()]
            fout.write(f"price: {price_flat}\n")
           
            rewards_flat = [round(float(x[0]), 2) for x in rewards.numpy().tolist()]
            fout.write(f"rewards: {rewards_flat}\n")
            advantage_flat = [round(float(x[0]), 2) for x in advantage.numpy().tolist()]
            fout.write(f"advantage: {advantage_flat}\n")

        next_states = transition_dict['next_states']
        next_states = process_tensors(next_states, feature_num)
        dones = tf.convert_to_tensor(transition_dict['dones'], dtype=tf.float32)
        dones = tf.reshape(dones, (-1, 1))

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        action_probs = self.actor3(states)
        indices = tf.range(0, action_probs.shape[0]) * tf.shape(action_probs)[1] + price[:, 0]
        epsilon = 1e-30
        old_safe_action_probs = tf.clip_by_value(
            tf.reshape(tf.gather(tf.reshape(action_probs, [-1]), indices), (-1, 1)),
            epsilon, 1.0 - epsilon)
        old_log_probs = tf.math.log(old_safe_action_probs)

        for _ in range(self.epochs):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                action_probs = self.actor3(states, training=True)
                safe_action_probs = tf.clip_by_value(action_probs, epsilon, 1.0 - epsilon)
                log_probs = tf.math.log(safe_action_probs)
                entropy = -tf.reduce_sum(safe_action_probs * log_probs, axis=1, keepdims=True)
                entropy = tf.reduce_mean(entropy)

                safe_action_probs = tf.clip_by_value(
                    tf.reshape(tf.gather(tf.reshape(action_probs, [-1]), indices), (-1, 1)), epsilon, 1.0 - epsilon)
                log_probs = tf.math.log(safe_action_probs)
                
                advantage = advantage
                # exp(-γ * R(τ))
                weights = tf.exp(-self.risk_gamma * advantage)
                # γ * E[exp(-γ * R(τ))]
                denominator = self.risk_gamma * tf.reduce_mean(weights)
                # log πθ(a|s) * exp(-γ * R(τ))
                numerator = log_probs * weights
                #  E[numerator / denominator]
                actor_loss = tf.reduce_mean(numerator) / denominator

                entropy_coefficient=1
                smoothness_coeff = 1

                actor_loss = actor_loss - entropy_coefficient * entropy + smoothness_coeff * smoothness_loss(
                    action_probs)
                critic_loss = tf.reduce_mean(tf.square(self.critic(states, training=True) - td_target))

            actor_grads = actor_tape.gradient(actor_loss, self.actor3.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)
            self.actor_optimizer3.apply_gradients(zip(actor_grads, self.actor3.trainable_variables))
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        return entropy.numpy(), actor_loss.numpy(), critic_loss.numpy(), smoothness_loss(action_probs).numpy()

    def reset_state(self):
        self.actor3.reset_states()
        self.critic.reset_states()
        return


def train_actor_model(risk_gamma):
    
    log_dir = f"logs_finetune_gamma_{risk_gamma}/live_multiple_lists"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    summary_writer = tf.summary.create_file_writer(log_dir)
    log_file = os.path.join(log_dir, f"finetune_gamma_{risk_gamma}.txt")

    actor_lr = 0.00005
    critic_lr = 0.0005
    num_episodes = 600
    gamma = 0.95
    #gamma =0.81
    #gamma=0.2
    lmbda = 0.98
    epochs = 5
    eps = 0.2
    buffer_size = 1500
    minimal_size = 700
    batch_size = 256
    replay_buffer = ReplayBuffer(buffer_size)

    model_classification = load_model('saved_model/classification_model.h5', custom_objects={'F1_score': F1_score})
    model_classification.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score])
    model_classification_price = load_model('saved_model/classification_price_model.h5',
                                            custom_objects={'F1_score': F1_score})
    model_classification_price.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', F1_score])
    model_regression = load_model('saved_model/regression_model_new.h5')

    agent = REINFORCE(actor_lr, critic_lr, lmbda,epochs, eps, gamma,risk_gamma)

    

    
    start_episode = 0


    
    for episode in range(start_episode, num_episodes):
        # print('----------------episode-------------')
        # print(episode)
        current_marketID_period= random.choice(open('data_processing/simulate_market.txt', 'r', encoding='utf-8').readlines()).strip()
        total_time = find_time_count("data_processing/time.jsonl", current_marketID_period)
        time=get_random_time(total_time)
        buyers, whole_history, whole_previous_offer, whole_previous_trade, sellers, current_unit, bids, asks, transactions, buy_offers, sell_offers = initialize_market(
            current_marketID_period, time)
        buyer = random.choice(buyers)
        agent.reset_state()
        new_buyers = buyers.copy()
        new_buyers.remove(buyer)
        env_history = copy.deepcopy(whole_history)
        del env_history[buyer]
        env_pre_offer = copy.deepcopy(whole_previous_offer)
        del env_pre_offer[buyer]
        env_pre_trade = copy.deepcopy(whole_previous_trade)
        del env_pre_trade[buyer]

        # print(whole_history)
        # print(whole_previous_offer)
        # print(whole_previous_trade)
        with open(log_file, mode='a+', encoding='utf-8') as fout:
            fout.write(f"episode: {episode}\n")


        env = MarketEnvironment(new_buyers, sellers, model_classification, model_classification_price, model_regression,
                                current_unit, bids.copy(), asks.copy(), transactions.copy(), buy_offers.copy(),
                                sell_offers.copy(),
                                env_history.copy(), env_pre_offer.copy(), env_pre_trade.copy(), buyer, time, total_time)
        state = copy.deepcopy(env.get_state())
        done = False
        history = list(whole_history[buyer][1]).copy()
        previous_offer = whole_previous_offer[buyer].copy()
        previous_trade = whole_previous_trade[buyer].copy()
        unit = whole_history[buyer][0]
        #cost = sorted([random.randint(25, 150) for _ in range(current_unit)], reverse=True)
        cost = sorted([random.randint(20, 250) for _ in range(current_unit)], reverse=True)

        episode_return = 0
        total_return=0
        #transition_dict = {'states': [],'price':[],'next_states': [], 'rewards': [], 'dones': []}
        check_action=[]
        buffer_s,buffer_r,buffer_a,buffer_ns,buffer_d = [], [], [],[],[]
        while not done and unit<=current_unit:
            
            state_data = generate_state_data(state, cost, unit, history, previous_offer, previous_trade, time,
                                             total_time)
            states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in state_data]
            price=agent.take_action3(states)
            #print(price)

            if history!=[]:
                if price//10 == history[-1]//10:
                    change=0
                else:
                    change=1
            else:
                change=1
            history.append(price)

            
            offer=price
            check_action.append(offer)
            next_state, done, trade,price = env.step(price,change)

            # rewards shaping
            # print(state[1])
            # print(next_state[1])
            state_values = copy.deepcopy(list(state[1].values()))
            next_state_values = copy.deepcopy(list(next_state[1].values()))

            total_reward, reward = calculate_reward1(trade, offer, price, cost, unit, state_values, next_state_values)

            if trade==1:
                unit+=1
                history=[]
                previous_trade.append(price)
                previous_offer.append(offer)

            time += 1
            if unit > current_unit:
                done = True
                next_state_data = state_data
            else:
                next_state_data = generate_state_data(next_state, cost, unit, history, previous_offer, previous_trade, time, total_time)
            next_states = [tf.convert_to_tensor(s, dtype=tf.float32) for s in next_state_data]

            #replay_buffer.add(states, offer, reward, next_states, done)
            buffer_s.append(states)
            buffer_r.append(total_reward)
            buffer_a.append(offer)
            buffer_ns.append(next_states)
            buffer_d.append(done)
            state = copy.deepcopy(next_state)
            episode_return += reward
            #total_return+=total_reward
            if unit > current_unit:
                break

        if buffer_s==[]:
            continue

        #  R(τ) = Σ γ^t r_t
        discounted_cumulative_reward = 0
        for t, r in enumerate(reversed(buffer_r)):
            discounted_cumulative_reward = r + gamma * discounted_cumulative_reward
        
        advantage = [discounted_cumulative_reward] * len(buffer_s)
        buffer_d=  tf.convert_to_tensor(buffer_d, dtype=tf.float32)
        buffer_d =tf.reshape(buffer_d, (-1, 1))
        buffer_r = tf.convert_to_tensor(buffer_r, dtype=tf.float32)
        buffer_r = tf.reshape(buffer_r, (-1, 1))

        with open(log_file, mode='a+', encoding='utf-8') as fout:
            fout.write(f"buffer_s: {buffer_s}\n")
            fout.write(f"buffer_a: {buffer_a}\n")
            fout.write(f"buffer_r: {buffer_r}\n")
            fout.write(f"advantage: {advantage}\n")
            fout.write(f"buffer_ns: {buffer_ns}\n")
            fout.write(f"buffer_d: {buffer_d}\n")
        for i in range(len(buffer_s)):
            replay_buffer.add(buffer_s[i], buffer_a[i], buffer_r[i],advantage[i],buffer_ns[i], buffer_d[i])

        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ad,b_ns, b_d = replay_buffer.sample(batch_size)
            transition_dict = {'states': b_s,
                               'actions': b_a,
                               'next_states': b_ns,
                               'rewards': b_r,
                               'advantages': b_ad,
                               'dones': b_d}
            entropy,actor_loss,critic_loss,smooth_loss=agent.update(transition_dict)

            
            with summary_writer.as_default():
                tf.summary.scalar("entropy", entropy, step=episode)
                tf.summary.scalar("actor_loss", actor_loss, step=episode)
                tf.summary.scalar("critic_loss", critic_loss, step=episode)
                tf.summary.scalar("return", episode_return, step=episode)
                #tf.summary.scalar("total_return",total_return, step=episode)
                tf.summary.scalar("smoothness_loss", smooth_loss, step=episode)

            if entropy<0.1:
                break


    agent.save_model()
    return agent.actor3


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train RL agent with different risk parameters')
    parser.add_argument('--risk_gamma', type=float, required=True,
                        help='Risk sensitivity parameter (gamma_risk) for the agent')

    args = parser.parse_args()

    
    print(f"Starting training with risk_gamma: {args.risk_gamma}")

    
    train_actor_model(risk_gamma=args.risk_gamma)

# python finetune.py --risk_gamma 0.81