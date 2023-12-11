#================================================================
#
#   File name   : multiprocessing_env.py
#   Author      : PyLessons
#   Created date: 2021-02-08
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : functions to train/test multiple custom BTC trading environments
#
#================================================================
from collections import deque
from multiprocessing import Process, Pipe
import numpy as np
from datetime import datetime
from timeit import timeit

class Environment(Process):
    def __init__(self, env_idx, child_conn, env, training_batch_size, visualize):
        super(Environment, self).__init__()
        self.env = env
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.training_batch_size = training_batch_size
        self.visualize = visualize

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        self.child_conn.send(state)

        start_time = datetime.now()
        while True:
            reset, net_worth, episode_orders = 0, 0, 0
            action = self.child_conn.recv()
            # if self.env_idx == 0:
            #     self.env.render(self.visualize)
            start_time2 = datetime.now()
            state, reward, done, info = self.env.step(action)
            print(f"Worker#{self.env_idx} ... Step Time: {datetime.now()-start_time2}")

            if done:
                net_worth = self.env.current_equity
                episode_orders = len(self.env.transaction_history)
                state = self.env.reset()
                reset = 1

                print(f"Worker#{self.env_idx} ... Trade Time: {datetime.now()-start_time}")
                start_time = datetime.now()

            self.child_conn.send([state, reward, done, reset, net_worth, episode_orders])

def train_multiprocessing(CustomEnv, agent, train_df, num_worker=4, training_batch_size=500, lookback_window_size=50, visualize=False, EPISODES=10000):
    works, parent_conns, child_conns = [], [], []
    episode = 0
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth

    for idx in range(num_worker):
        parent_conn, child_conn = Pipe()
        env = CustomEnv(data=train_df, lookback_window_size=lookback_window_size, verbose=False)
        work = Environment(idx, child_conn, env, training_batch_size, visualize)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    agent.create_writer(env.initial_balance, 0, EPISODES) # create TensorBoard writer

    states =        [[] for _ in range(num_worker)]
    next_states =   [[] for _ in range(num_worker)]
    actions =       [[] for _ in range(num_worker)]
    rewards =       [[] for _ in range(num_worker)]
    dones =         [[] for _ in range(num_worker)]
    predictions =   [[] for _ in range(num_worker)]

    state = [0 for _ in range(num_worker)]
    for worker_id, parent_conn in enumerate(parent_conns):
        state[worker_id] = parent_conn.recv()

    while episode < EPISODES:
        predictions_list = agent.Actor.actor_predict(np.reshape(state, [num_worker]+[_ for _ in state[0].shape]))
        actions_list = [np.random.choice(agent.action_space, p=i) for i in predictions_list]

        for worker_id, parent_conn in enumerate(parent_conns):
            parent_conn.send(actions_list[worker_id])
            action_onehot = np.zeros(agent.action_space.shape[0])
            action_onehot[actions_list[worker_id]] = 1
            actions[worker_id].append(action_onehot)
            predictions[worker_id].append(predictions_list[worker_id])

        for worker_id, parent_conn in enumerate(parent_conns):
            next_state, reward, done, reset, net_worth, episode_orders = parent_conn.recv()
            states[worker_id].append(np.expand_dims(state[worker_id], axis=0))
            next_states[worker_id].append(np.expand_dims(next_state, axis=0))
            rewards[worker_id].append(reward)
            dones[worker_id].append(done)
            state[worker_id] = next_state

            if reset:
                start_time = datetime.now()
                episode += 1
                a_loss, c_loss = agent.replay(states[worker_id], actions[worker_id], rewards[worker_id], predictions[worker_id], dones[worker_id], next_states[worker_id])
                print(f"Agent replay time: {(datetime.now() - start_time)}")
                total_average.append(net_worth)
                average = np.average(total_average)

                agent.writer.add_scalar('Data/average net_worth', average, episode)
                agent.writer.add_scalar('Data/episode_orders', episode_orders, episode)
                
                print("episode: {:<5} worker: {:<1} net worth: {:<7.2f} average: {:<7.2f} orders: {}".format(episode, worker_id, net_worth, average, episode_orders))
                if episode > len(total_average):
                    if best_average < average:
                        best_average = average
                        print("Saving model")
                        agent.save(score="{:.2f}".format(best_average), args=[episode, average, episode_orders, a_loss, c_loss])
                    agent.save()
                
                states[worker_id] = []
                next_states[worker_id] = []
                actions[worker_id] = []
                rewards[worker_id] = []
                dones[worker_id] = []
                predictions[worker_id] = []

    agent.end_training_log()
    # terminating processes after while loop
    works.append(work)
    for work in works:
        work.terminate()
        print('TERMINATED:', work)
        work.join()