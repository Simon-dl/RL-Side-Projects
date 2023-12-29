import gymnasium as gym
from collections import namedtuple
from gymnasium.core import Env
import numpy as np
from tensorboardX import SummaryWriter

import os
import torch
import torch.nn as nn
import torch.optim as optim

from gymnasium.wrappers.record_video import RecordVideo 

HIDDEN_SIZE = 128
PERCENTILE = 90
BATCH_SIZE = 10
LOGDIR_PATH = os.getcwd()+ '/Cliff_Walking/runs'
LOGDIR_PATH_VIDEO = os.getcwd()+ '/Cliff_Walking/monitor'



class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, act_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, act_size),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits

EpisodeSteps = namedtuple('EpisodeSteps', ['obs','act'])
Episode = namedtuple('Episode', ['reward','steps'])

def Get_Batches(env,nn):
    batch = []
    act_amount = env.action_space.n
    softmax = torch.nn.Softmax(dim=0)
    episode_steps = []
    reward = 0


    obs = env.reset()[0]

    while True:
        obs = torch.FloatTensor(obs)

        action_logits = nn(obs)
        action_vector = softmax(action_logits)
        action_np = action_vector.detach().numpy()
        action = np.random.choice(act_amount,p=action_np)

        obs, step_reward, done, trunc, _ = env.step(action)


        episode_steps.append(EpisodeSteps(obs,action))
        reward += step_reward



        if done or trunc:
            e = Episode(reward,episode_steps)
            batch.append(e)
            reward = 0
            episode_steps = []
            next_obs = env.reset()[0]
            obs = next_obs
            if len(batch) == BATCH_SIZE:
                yield batch
                batch = []
            


def Get_Elite_Episodes(batch, percentile):
    rewards_list = list(map(lambda e: e.reward,batch))
    reward_boundry = np.percentile(rewards_list,percentile)
    reward_mean = float(np.mean(rewards_list))

    training_obs = []
    training_acts = []
    for reward, steps in batch:
        if reward < reward_boundry:
            continue
        training_obs.extend(map(lambda s: s.obs, steps))
        training_acts.extend(map(lambda s: s.act, steps))
    
    training_obs = torch.FloatTensor(training_obs)
    training_acts = torch.LongTensor(training_acts)

    return training_obs, training_acts, reward_boundry, reward_mean


        
class Make_Discrete(gym.ObservationWrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        assert isinstance(env.observation_space,gym.spaces.Discrete)
        
        shape = (env.observation_space.n,)
        self.observation_space = gym.spaces.Box(0.0,1.0,shape, dtype=np.float32)
    
    def observation(self, observation):
        one_hot = np.copy(self.observation_space.low)
        one_hot[observation] = 1.0
        return one_hot



if __name__ == '__main__':
    env = Make_Discrete(gym.make("CliffWalking-v0",render_mode="rgb_array"))  
    env = RecordVideo(env,LOGDIR_PATH_VIDEO)

    

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    print(obs_size)
    print(act_size)

    net = Net(obs_size,HIDDEN_SIZE,act_size)

    optimizer = optim.Adam(net.parameters(), lr = .01)
    loss_fn = nn.CrossEntropyLoss()
    writer = SummaryWriter(LOGDIR_PATH,"Cliff Walker")

    for iter_no, batch in enumerate(Get_Batches(env,net)):
        t_obs, t_act, r_b, r_m = Get_Elite_Episodes(batch, PERCENTILE)
        optimizer.zero_grad()

        pred_acts = net(t_obs)
        loss = loss_fn(pred_acts,t_act)
        loss.backward()
        optimizer.step()

        env.render()

        print("%d: loss=%.3f, reward_mean=%.1f, rw_bound=%.1f" % (iter_no, loss.item(), r_m, r_b))
        writer.add_scalar("Loss",loss.item(),iter_no)
        writer.add_scalar("Reward Boundry", r_b, iter_no)
        writer.add_scalar("Mean Negative Reward", -r_m, iter_no)
        if -r_m < 200 or iter_no == 500:
            print('done')
            break
    
    writer.close()
        


