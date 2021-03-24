#https://www.analyticsvidhya.com/blog/2021/01/bear-run-or-bull-run-can-reinforcement-learning-help-in-automated-trading/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from datetime import datetime, timedelta
from collections import deque
import random
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
#Load data
def date_sub(date):
    day=datetime.strptime(date,'%m_%d_%Y')
    day=day-timedelta(days=1)
    return day.strftime('%m_%d_%Y')

def val_sub(val):
    if val > 0.5:
        return 1
    else:
        return -1
    
    
def read_convert_data(symbol='None'):
    prices = pd.read_csv("C:/Users/Annoy/Desktop/Spyder/Useful Code Snippets/Data/Historic BTC-USD data/BTC-USD_Megafile_01_31_2020-03_02_2020.csv")
    df=pd.read_csv("C:/Users/Annoy/Desktop/Spyder/Useful Code Snippets/Data/Historic BTC-USD data/Historic_BTC-USD_Prices_01_31_2020-03_02_2020_day_values.csv")
    #clean df data
    df['Volume'].replace(to_replace=0, method='ffill', inplace=True) 
    

# Drop all rows with NaN values
    df.dropna(how='any', axis=0, inplace=True) 
#Moving Average, calculate percentage change for columns
    df['Open'] = df['Open'].pct_change() # Create arithmetic returns column
    df['High'] = df['High'].pct_change() # Create arithmetic returns column
    df['Low'] = df['Low'].pct_change() # Create arithmetic returns column
    df['Close'] = df['Close'].pct_change() # Create arithmetic returns column
    df['Volume'] = df['Volume'].pct_change()

    df.dropna(how='any', axis=0, inplace=True) # Drop all rows with NaN values

###############################################################################
    '''Normalize price columns'''

    min_return = min(df[['Open', 'High', 'Low', 'Close']].min(axis=0))
    max_return = max(df[['Open', 'High', 'Low', 'Close']].max(axis=0))

    # Min-max normalize price columns (0-1 range)
    df['Open'] = (df['Open'] - min_return) / (max_return - min_return)
    df['High'] = (df['High'] - min_return) / (max_return - min_return)
    df['Low'] = (df['Low'] - min_return) / (max_return - min_return)
    df['Close'] = (df['Close'] - min_return) / (max_return - min_return)

    ###############################################################################
    '''Normalize volume column'''

    min_volume = df['Volume'].min(axis=0)
    max_volume = df['Volume'].max(axis=0)

    # Min-max normalize volume columns (0-1 range)
    df['Volume'] = (df['Volume'] - min_volume) / (max_volume - min_volume)
    newnames={'Low':'rel_low','High':'rel_high','Open':'rel_open','Close':'rel_close','Volume':'rel_volume'}
    df.rename(columns=newnames, inplace=True)
    df.Date=df.Date.map(date_sub)
    df.rel_low=df.rel_low.map(val_sub)
    df.rel_high=df.rel_high.map(val_sub)
    df.rel_close=df.rel_close.map(val_sub)
    df.rel_open=df.rel_open.map(val_sub)
    df.rel_volume=df.rel_volume.map(val_sub)
    
    #clean price data
    prices = prices.loc[:, ~prices.columns.str.contains('^Unnamed')]
    prices.rename(columns={'date':'Date'},inplace=True)
    #merge dataframes
    test=pd.merge(df,prices, on='Date',how='outer')
    test.fillna(value=0,inplace=True)
    #merge dataframe data
    test.to_pickle("C:/Users/Annoy/Desktop/Spyder/Useful Code Snippets/Data/Historic BTC-USD data/BTC-USD_Megafile_01_31_2020-03_02_2020.pkl")
    return

def load_data(test=False):
    #prices = pd.read_pickle('data/OILWTI_1day.pkl')
    #prices = pd.read_pickle('data/EURUSD_1day.pkl')
    #prices.rename(columns={'Value': 'close'}, inplace=True)
    prices = pd.read_pickle("C:/Users/Annoy/Desktop/Spyder/Useful Code Snippets/Data/Historic BTC-USD data/BTC-USD_Megafile_01_31_2020-03_02_2020.pkl")
    prices.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume (BTC)': 'volume'}, inplace=True)
    print(prices)
    x_train = prices.iloc[0:int(len(prices)*0.7)]
    x_test= prices.iloc[int(len(prices)*0.7):]
    if test:
        return x_test
    else:
        return x_train


df = load_data()
#indata.drop('Unnamed: 0',inplace=True, axis=1)
test_data = load_data(test=True)
#test_data.drop('Unnamed: 0',inplace=True, axis=1)
name = 'Q-learning agent'
class Agent:
    def __init__(self, state_size, window_size, trend, skip, batch_size):
        self.state_size = state_size
        self.window_size = window_size
        self.half_window = window_size // 2
        self.trend = trend
        self.skip = skip
        self.action_size = 3
        self.batch_size = batch_size
        self.memory = deque(maxlen = 1000)
        self.inventory = []
        self.gamma = 0.95
        self.epsilon = 0.5
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.X = tf.placeholder(tf.float32, [None, self.state_size])
        self.Y = tf.placeholder(tf.float32, [None, self.action_size])
        feed = tf.layers.dense(self.X, 256, activation = tf.nn.relu)
        self.logits = tf.layers.dense(feed, self.action_size)
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.GradientDescentOptimizer(1e-5).minimize(
            self.cost
        )
        self.sess.run(tf.global_variables_initializer())
    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(
            self.sess.run(self.logits, feed_dict = {self.X: state})[0]
        )
    def get_state(self, t):
        window_size = self.window_size + 1
        d = t - window_size + 1
        block = self.trend[d : t + 1] if d >= 0 else -d * [self.trend[0]] + self.trend[0 : t + 1]
        res = []
        for i in range(window_size - 1):
            res.append(block[i + 1] - block[i])
        return np.array([res])
    def replay(self, batch_size):
        mini_batch = []
        l = len(self.memory)
        for i in range(l - batch_size, l):
            mini_batch.append(self.memory[i])
        replay_size = len(mini_batch)
        X = np.empty((replay_size, self.state_size))
        Y = np.empty((replay_size, self.action_size))
        states = np.array([a[0][0] for a in mini_batch])
        new_states = np.array([a[3][0] for a in mini_batch])
        Q = self.sess.run(self.logits, feed_dict = {self.X: states})
        Q_new = self.sess.run(self.logits, feed_dict = {self.X: new_states})
        for i in range(len(mini_batch)):
            state, action, reward, next_state, done = mini_batch[i]
            target = Q[i]
            target[action] = reward
            if not done:
                target[action] += self.gamma * np.amax(Q_new[i])
            X[i] = state
            Y[i] = target
        cost, _ = self.sess.run(
            [self.cost, self.optimizer], feed_dict = {self.X: X, self.Y: Y}
        )
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return cost
    def buy(self, initial_money):
        starting_money = initial_money
        states_sell = []
        states_buy = []
        inventory = []
        state = self.get_state(0)
        for t in range(0, len(self.trend) - 1, self.skip):
            action = self.act(state)
            next_state = self.get_state(t + 1)
            if action == 1 and initial_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                inventory.append(self.trend[t])
                initial_money -= self.trend[t]
                states_buy.append(t)
                print('day %d: buy 1 unit at price %f, total balance %f'% (t, self.trend[t], initial_money))
            elif action == 2 and len(inventory):
                bought_price = inventory.pop(0)
                initial_money += self.trend[t]
                states_sell.append(t)
                try:
                    invest = ((close[t] - bought_price) / bought_price) * 100
                except:
                    invest = 0
                print(
                    'day %d, sell 1 unit at price %f, investment %f %%, total balance %f,'
                    % (t, close[t], invest, initial_money)
                )
            state = next_state
        invest = ((initial_money - starting_money) / starting_money) * 100
        total_gains = initial_money - starting_money
        return states_buy, states_sell, total_gains, invest
    def train(self, iterations, checkpoint, initial_money):
        for i in range(iterations):
            total_profit = 0
            inventory = []
            state = self.get_state(0)
            starting_money = initial_money
            for t in range(0, len(self.trend) - 1, self.skip):
                action = self.act(state)
                next_state = self.get_state(t + 1)
                if action == 1 and starting_money >= self.trend[t] and t < (len(self.trend) - self.half_window):
                    inventory.append(self.trend[t])
                    starting_money -= self.trend[t]
                elif action == 2 and len(inventory) > 0:
                    bought_price = inventory.pop(0)
                    total_profit += self.trend[t] - bought_price
                    starting_money += self.trend[t]
                invest = ((starting_money - initial_money) / initial_money)
                self.memory.append((state, action, invest, 
                                    next_state, starting_money < initial_money))
                state = next_state
                batch_size = min(self.batch_size, len(self.memory))
                cost = self.replay(batch_size)
            if (i+1) % checkpoint == 0:
                print('epoch: %d, total rewards: %f.3, cost: %f, total money: %f'%(i + 1, total_profit, cost,starting_money))
close = df.close.values.tolist()
initial_money = 10000
window_size = 30
skip = 1
batch_size = 32
agent = Agent(state_size = window_size, 
              window_size = window_size, 
              trend = close, 
              skip = skip, 
              batch_size = batch_size)
agent.train(iterations = 2000, checkpoint = 10, initial_money = initial_money)
states_buy, states_sell, total_gains, invest = agent.buy(initial_money = initial_money)
fig = plt.figure(figsize = (15,5))
plt.plot(close, color='r', lw=2.)
plt.plot(close, '^', markersize=10, color='m', label = 'buying signal', markevery = states_buy)
plt.plot(close, 'v', markersize=10, color='k', label = 'selling signal', markevery = states_sell)
plt.title('total gains %f, total investment %f%%'%(total_gains, invest))
plt.legend()
plt.savefig(name+'.png')
plt.show()
                          
