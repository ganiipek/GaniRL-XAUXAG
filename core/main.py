import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import Adam, RMSprop
from algorithms.ppo_tf2.agent import CustomAgent
from algorithms.ppo_tf2.multiprocessing_env import train_multiprocessing
from environments.ratioenv import RatioEnv
from environments.loader_xauxag import Loader
from utils import plot_learning_curve

# logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# file_writer = tf.summary.create_file_writer(logdir + "/metrics")
# file_writer.set_as_default()

if __name__ == '__main__':
    loader = Loader()
    train_df = loader.load(start_date='2019-01-01', end_date='2019-12-31')
    
    N = 20
    batch_size = 128
    n_epochs = 4
    alpha = 0.0003
    lookback_window_size = 120

    # agent = CustomAgent(n_actions=3, batch_size=batch_size,
    #               alpha=alpha, n_epochs=n_epochs,
    #               input_dims=50)
    agent = CustomAgent(lookback_window_size=lookback_window_size, lr=0.00001, epochs=5, optimizer=Adam, batch_size = batch_size, model="CNN")
    train_multiprocessing(RatioEnv, agent, train_df, num_worker = 1, training_batch_size=500, lookback_window_size=lookback_window_size, EPISODES=200000)
    
# if __name__ == '__main__':
#     # The algorithms require a vectorized environment to run
#     env = RatioEnv(verbose=False)
    
#     N = 20
#     batch_size = 128
#     n_epochs = 4
#     alpha = 0.0003
#     agent = Agent(n_actions=3, batch_size=batch_size,
#                   alpha=alpha, n_epochs=n_epochs,
#                   input_dims=env.state_shape())
#     n_games = 15
#     score_history = []

#     learn_iters = 0
#     avg_score = 0
#     n_steps = 0
#     best_score = 0

#     for i in range(n_games):
#         start_time = datetime.now()
#         observation = env.reset()
#         done = False
#         score = 0
#         result = None

#         while not done:
#             action, prob, val = agent.choose_action(np.expand_dims(observation, axis=0))
#             observation_, reward, done, info = env.step(action)
#             result = info
#             n_steps += 1
#             score += reward
#             agent.store_transition(observation, action,
#                                    prob, val, reward, done)
#             if n_steps % N == 0:
#                 agent.learn()
#                 learn_iters += 1
#             observation = observation_
            
#         score_history.append(score)
#         avg_score = np.mean(score_history[-100:])
        
#         if avg_score > 0:
#             best_score = avg_score
#             agent.save_models()

#         print(f"\nEpisode#{i} ... Learning Time: {datetime.now()-start_time}")
#         tf.summary.scalar('reward summary', data=avg_score, step=i)
#         print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
#               'time_steps', n_steps, 'learning_steps', learn_iters)
#         print(result)

        
#     filename = 'PPO_trading_view.png'
#     x = [i+1 for i in range(len(score_history))]
#     plot_learning_curve(x, score_history, filename)
