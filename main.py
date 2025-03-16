# Dueling-DDQN-PP (PRIORITIZED EXPERIENCE REPLAY)
import numpy as np
import torch
import gym
from Duel_DDQN_PP import D_DDQN_PP_train, Replay_buffer
from torch.utils.tensorboard import SummaryWriter
import os, shutil
from datetime import datetime
import argparse
import warnings
from Atari_Name import Name
from wrappers_deepmind import make_env

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
'str2bool'

def str2bool(V):
    if isinstance(V, bool):
        return V
    elif V.lower in ('yes', 'true', 't', 'y'):
        return True
    elif V.lower in ('no', 'false', 'f', 'n'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def eval_func(env_eval, model, seed, e_turns=3):
    score = 0
    for t in range(e_turns):
        s, _ = env_eval.reset(seed=seed)
        done, e_r = False, 0
        while not done:
            action = model.action_selection(s, True)
            s_, r, dw, tr, info = env_eval.step(action)
            done = (dw or tr)
            e_r += r
            s = s_
        score += e_r
    return int(score / e_turns)


class line_func(object):
    def __init__(self, step, final, initial):
        self.step = step
        self.final = final
        self.initial = initial

    def value(self, t):
        f = min(t / self.step, 1.0)
        return self.initial + f * (self.final - self.initial)
        
def show_func(env_eval, model, seed, e_turns=3):
    for t in range(e_turns):
        s, _ = env_eval.reset(seed=seed)
        done = False
        while not done:
            action = model.action_selection(s, True)
            s_, r, dw, tr, _ = env_eval.step(action)
            s = s_
            done = (dw or tr)
            env_eval.render()


'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=900, help='which model to load')

parser.add_argument('--Max_train_steps', type=int, default=int(1E6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(1E5), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--random_steps', type=int, default=int(8192),
                    help='random steps before training, 5E4 in DQN Nature')
parser.add_argument('--buffersize', type=int, default=int(8192), help='Size of the replay buffer')
parser.add_argument('--target_freq', type=int, default=int(1E3), help='frequency of target net updating')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='lenth of sliced trajectory')
parser.add_argument('--initial', type=float, default=1.0, help='Initial e-greedy noise')
parser.add_argument('--anneal_frac', type=int, default=3e5, help='annealing fraction of e-greedy noise')
parser.add_argument('--final', type=float, default=0.02, help='Final e-greedy noise')
parser.add_argument('--noop_reset', type=str2bool, default=False, help='use NoopResetEnv or not')
parser.add_argument('--huber_loss', type=str2bool, default=True, help='True: use huber_loss; False:use mse_loss')
parser.add_argument('--fc_width', type=int, default=200, help='number of units in Fully Connected layer')

parser.add_argument('--EnvIdex', type=int, default=37, help='Index of the Env; 20=Enduro; 37=Pong')
parser.add_argument('--seed', type=int, default=5, help='random seed')
parser.add_argument('--Double', type=str2bool, default=False, help="whether to use Double Q-learning")
parser.add_argument('--Duel', type=str2bool, default=False, help="whether to use Duel. Q-learning")
parser.add_argument('--Noisy', type=str2bool, default=False, help="whether to use NoisyNet")
parser.add_argument('--e_turns', type=int, default=3, help='evaluation turns')

parser.add_argument('--alpha_r', type=float, default=0.7, help='weight for PRIORITIZED (rank_based)')
parser.add_argument('--beta_r_0', type=float, default=0.5, help='weight for UPDATE (rank_based)')
parser.add_argument('--alpha_p', type=float, default=0.6, help='weight for PRIORITIZED (prioritization_based)')
parser.add_argument('--beta_p_0', type=float, default=0.4, help='weight for UPDATE (prioritization_based)')
parser.add_argument('--rank', type=str2bool, default=True, help='rank or prioritization')
parser.add_argument('--beta_gain_steps', type=int, default=int(3e5), help='steps of beta from beta_init to 1.0')
opt = parser.parse_args()
opt.EnvName = Name[opt.EnvIdex] + "NoFrameskip-v4"
opt.algo_name = "Dueling-DDQN"
opt.ExperimentName = opt.algo_name + '_' + opt.EnvName
print('\n', opt)
if opt.rank:
    opt.alpha = opt.alpha_r
    opt.beta_0 = opt.beta_r_0
else:
    opt.alpha = opt.alpha_p
    opt.beta_0 = opt.beta_p_0


def main():
    # Seed Everything
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    render_mode = 'human' if opt.render else None
    env_eval = make_env(opt.EnvName, noop_reset=opt.noop_reset, episode_life=False, clip_rewards=False,
                        render_mode=render_mode)
    opt.action_dim = env_eval.action_space.n

    print('Algorithm:', opt.algo_name, '  Env:', opt.EnvName, ' action_dim:', opt.action_dim, '  Random Seed:',
          opt.seed, '\n')

    if opt.write:
        timenow = str(datetime.now())[0:-10]
        timenow = ' ' + timenow[0:13] + '_' + timenow[-2::]
        writepath = 'runs/{}_{}'.format(opt.algo_name, opt.EnvName) + timenow
        if os.path.exists(writepath): shutil.rmtree(writepath)
        writer = SummaryWriter(log_dir=writepath)

    if not os.path.exists('model'): os.mkdir('model')
    model = D_DDQN_PP_train(opt)
    if opt.Loadmodel: model.load(opt.algo_name, opt.EnvName, opt.ModelIdex)

    if opt.render:
        score = eval_func(env_eval, model, opt.seed, 1)
        print('Env_Name:', opt.EnvName, ' seed:', opt.seed, 'initial', ' score:', score)

    replay = Replay_buffer(opt)
    env_train = make_env(opt.EnvName, noop_reset=opt.noop_reset)
    line = line_func(opt.anneal_frac, opt.final, opt.initial)
    beta_line = line_func(opt.beta_gain_steps, opt.beta_0, 1.0)
    total_steps = 0
    seed = opt.seed + 1
    while not total_steps > opt.Max_train_steps:
        s, _ = env_train.reset(seed=seed)
        seed += 1
        done = False
        while not done:
            a = model.action_selection(s, False)
            s_, r, dw, tr, _ = env_train.step(a)
            replay.add(s, a, r, s_, dw)
            done = (dw or tr)
            s = s_

            if replay.size >= opt.random_steps:
                model.train(replay)

                if total_steps % opt.eval_interval == 0:
                    score = eval_func(env_eval, model, opt.seed, )
                    if opt.write:
                        writer.add_scalar('ep_r', score, global_step=total_steps)
                        writer.add_scalar('noise', model.exp_noise, global_step=total_steps)
                    print(f"{opt.ExperimentName}, Seed:{opt.seed}, Step:{int(total_steps / 1000)}k, Score:{score}")
                    model.exp_noise = line.value(total_steps)
                    replay.beta = beta_line.value(total_steps)

                total_steps += 1
                if total_steps % opt.save_interval == 0:
                    model.save(opt.algo_name, opt.EnvName, total_steps)

    if opt.show:
        env_show = gym.make(Bench, render_mode="human" if opt.show else None)
        show_func(env_show, model, env_seed + 1, 3)
    env_train.close()
    env_eval.close()


if __name__ == '__main__':
    main()
