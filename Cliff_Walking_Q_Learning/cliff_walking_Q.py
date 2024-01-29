import gymnasium as gym
from collections import defaultdict
from tensorboardX import SummaryWriter
import os
from gymnasium.wrappers.record_video import RecordVideo 


#takes too long to play test_episodes, let the environment be sampled a lot first

LOGDIR_PATH = os.getcwd()+ '/Cliff_Walking_Q_Learning/runs'
LOGDIR_PATH_VIDEO = os.getcwd()+ '/Cliff_Walking_Q_Learning/monitor'

ENV_NAME = "CliffWalking-v0"
TEST_EPISODES = 1
GAMMA = 0.9
ALPHA = 0.2 
SAMPLE_BEFORE_TEST = 20000

class Agent():
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()[0]
        self.values = defaultdict(float)

    def get_env_sample(self):
        """
        samples the environment one step at a time
        """
        old_state = self.state
        action = self.env.action_space.sample()
        new_state , reward, is_done, is_trunc, _ = self.env.step(action)
        self.state = self.env.reset()[0] if is_done else new_state
        return old_state, action, reward, new_state
    
    def get_best_action_and_value(self,state):
        """
        takes in a state and returns the best action and value for that state
        """
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            Q_val = self.values[(state,action)]
            if best_value == None or best_value < Q_val:
                best_value = Q_val
                best_action = action
        return best_action, best_value
    
    def play_test_episode(self,env):
        total_reward = 0
        state = env.reset()[0]
        inner_iter = 0
        while True:
            action , _ = self.get_best_action_and_value(state)
            new_state, reward, is_done, is_trunc, _ = env.step(action)
            total_reward += reward
            inner_iter += 1
            if is_done or inner_iter == 500:
                return total_reward
            state = new_state

    def update_action_value(self,o_s,a,r,n_s):
        _ , max_future_Q = self.get_best_action_and_value(n_s)
        old_val = self.values[(o_s,a)]
        new_value = r + (GAMMA * max_future_Q)
        self.values[(o_s,a)] = (1 - ALPHA) * old_val + ALPHA * new_value
    

if __name__ == '__main__':
    test_env = gym.make(ENV_NAME, render_mode="rgb_array")
    #test_env = RecordVideo(test_env,LOGDIR_PATH_VIDEO)
    agent = Agent()

    writer = SummaryWriter(log_dir=LOGDIR_PATH)

    best_reward = None
    iter_no = 0
    while True:
        iter_no += 1
        old_state, action, reward, new_state = agent.get_env_sample()
        agent.update_action_value(old_state, action, reward, new_state)

        if iter_no == SAMPLE_BEFORE_TEST:
            print('starting tests')
        if iter_no > SAMPLE_BEFORE_TEST:
            test_reward = 0
            for _ in range(TEST_EPISODES):
                test_reward += agent.play_test_episode(test_env)
                #if _ == TEST_EPISODES - 1:
                    #test_env.render()
            
            test_reward /= TEST_EPISODES #mean test reward

            if best_reward == None or test_reward > best_reward:
                print(f'reward updated from {best_reward} to {test_reward}')
                best_reward = test_reward
            
            writer.add_scalar("negative reward", test_reward, iter_no)

            if best_reward > -50 or iter_no == 45000:
                print(f'solved in {iter_no} iterations')
                break

    
    writer.close()
        

        



        




            