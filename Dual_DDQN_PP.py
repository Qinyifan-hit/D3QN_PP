import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Floss
from SumTree import sumtree_func

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net_con(nn.Module):
    def __init__(self, action_n, fc_width):
        super(Net_con, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, fc_width),
            nn.ReLU()
        )
        self.A = nn.Sequential(*[nn.Linear(fc_width, action_n), nn.Identity()])
        self.V = nn.Sequential(*[nn.Linear(fc_width, 1), nn.Identity()])

    def forward(self, obs):
        s = obs.float() / 255
        s = self.conv(s)
        Adv = self.A(s)
        V = self.V(s)
        Q_s_a = torch.add(V, Adv - torch.mean(Adv, dim=-1, keepdim=True))
        return Q_s_a


class D_DDQN_PP_train(object):
    def __init__(self, opt):
        self.q_net = Net_con(opt.action_dim, opt.fc_width).to(device)
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=opt.lr)
        self.q_target_net = copy.deepcopy(self.q_net)

        for p in self.q_target_net.parameters(): p.requires_grad = False
        self.device = device
        self.action_dim = opt.action_dim
        self.batch_size = opt.batch_size
        self.gamma = opt.gamma
        self.train_counter = 0
        self.huber_loss = opt.huber_loss
        self.Double = opt.Double
        self.Noisy = opt.Noisy
        self.target_freq = opt.target_freq
        self.exp_noise = opt.initial

        self.rank = opt.rank

    def action_selection(self, s, iseval):
        with torch.no_grad():
            state = s.unsqueeze(0).to(device)
            if iseval:
                if np.random.rand() < 0.01:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = torch.argmax(self.q_net(state), dim=1).item()
            else:
                if np.random.rand() < self.exp_noise:
                    action = np.random.randint(0, self.action_dim)
                else:
                    action = torch.argmax(self.q_net(state), dim=1).item()
        return action

    def train(self, Replay):
        s, a, r, s_, dw, ind, Is_w = Replay.sample(self.batch_size)
        with torch.no_grad():
            argmax_a = torch.argmax(self.q_net(s_), dim=-1, keepdim=True)
            Q_target_s_a = torch.gather(self.q_target_net(s_), -1, argmax_a).to(device)

            Y = r + (~dw) * self.gamma * Q_target_s_a

        Q_s_a = torch.gather(self.q_net(s), -1, a)
        TD_error = (Q_s_a - Y).squeeze(-1)

        Replay.priority_update(TD_error.detach().cpu().numpy(), ind, self.batch_size)
        q_loss = torch.mean(Is_w * (TD_error ** 2))
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.q_optimizer.step()

        self.train_counter += 1
        if self.train_counter % self.target_freq == 0:
            for p, p_target in zip(self.q_net.parameters(), self.q_target_net.parameters()):
                p_target.data.copy_(p.data)
        for p in self.q_target_net.parameters(): p.requires_grad = False

    def save(self, algo_name, env_name, steps):
        torch.save(self.q_net.state_dict(), "./model/{}_{}_{}.path".format(algo_name, env_name, steps))

    def load(self, algo_name, env_name, Index):
        self.q_net.load_state_dict(torch.load("./model/{}_{}_{}".format(algo_name, env_name, Index)))


class Replay_buffer(object):
    def __init__(self, opt):
        self.ptr = 0
        self.size = 0
        self.max_size = int(opt.buffersize)
        self.state = torch.zeros((self.max_size, 4, 84, 84), dtype=torch.uint8)
        self.state_ = torch.zeros((self.max_size, 4, 84, 84), dtype=torch.uint8)
        self.reward = torch.zeros((self.max_size, 1), dtype=torch.float32)
        self.action = torch.zeros((self.max_size, 1), dtype=torch.int32)
        self.dw = torch.zeros((self.max_size, 1), dtype=torch.bool)

        self.rank = opt.rank
        self.alpha = opt.alpha
        self.beta = opt.beta_0

        self.sumTree = sumtree_func(self.max_size)

    def add(self, s, a, r, s_, dw):
        self.state[self.ptr] = s
        self.state_[self.ptr] = s_
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.dw[self.ptr] = dw

        if self.size == 0:
            priority = 1.0
        else:
            priority = self.sumTree.p_max()
        self.sumTree.update(self.ptr, priority)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, b_size):
        ind, Is_w = self.sumTree.sample(self.size, b_size, self.beta)
        return (torch.tensor(self.state[ind], dtype=torch.float32).to(device),
                torch.tensor(self.action[ind], dtype=torch.int64).to(device),
                torch.tensor(self.reward[ind], dtype=torch.float32).to(device),
                torch.tensor(self.state_[ind], dtype=torch.float32).to(device),
                torch.tensor(self.dw[ind], dtype=torch.bool).to(device),
                ind,
                Is_w.to(device)
                )

    def priority_update(self, TD_error, Ind, b_size):
        priority = (np.abs(TD_error) + 0.0001) ** self.alpha
        for j in range(b_size):
            self.sumTree.update(Ind[j], priority[j])
