import gymnasium                # for gymnasium.Env
import numpy as np              # for the arrays (observations/actions)
import matplotlib.pyplot as plt # to get the plot of the rewards
import math                     # for math.pi and math.fmod
import cmath                    # for cmath.exp, and the use of complex numbers to represent the positions
import random                   # for the reset
import pygame                   # for the rendering
import torch                    # for the activations functions
import sys                      # to get sys.argv
import time                     # for the logs
import torch.nn as nn, torch    # for the custom policy
from stable_baselines3.common.policies import ActorCriticPolicy # same
thirty = 30/180*math.pi

def get_arr(*arr):
    return np.array(arr, dtype=np.float32)

class Car(gymnasium.Env):
    maxSpeed = 5
    maxi = 250
    dist = 300
    radius = 55
    refresh = 0.02
    eps = 1e+1
    epsAngle = 0.1
    limitStep = 10**4
    stLengthMax = 50
    def __init__(self, render_mode=None, id=None, every=100, actions=False):
        self.mimax = self.maxi/2
        assert render_mode is None or render_mode == 'human', f"Car env dosen't support the {render_mode} rendering"
        self.render_mode = False
        self.acts = []
        self.obs = []
        self.reset()
        self.render_mode = render_mode == 'human'
        self.acts = []
        self.obs = []
        self.W1 = self.dist+0j
        self.W2 = -self.dist+0j
        self.observation_space = gymnasium.spaces.Box(low=get_arr(-1, -1, -1, -1, -1), high=get_arr(1, 1, 1, 1, 1), shape=(5,))
        self.action_space = gymnasium.spaces.Box(low = get_arr(-1, -1), high=get_arr(1, 1), shape=(2,))
        if self.render_mode:
            self.init_rendering()
        self.datas = [[], []]
        if id is not None:
            self.logfile = f'logs/Car_{id}'
            with open(self.logfile, 'w') as f:f.write('')
        else:
            self.logfile = None
        self.every = every
        self.get_action = actions
    
    def c_pt(self, c):
        """give a pygame point from complex point"""
        return (c.real+self.misize, c.imag+self.misize)

    def project(self, pt, angle, length):
        """return complex point where the angle res-pt is equal to angle and |res-pt| = length"""
        return pt+cmath.exp(angle*1j)*length

    def draw_arrow(self, pt1, size, angle):
        pt2 = self.project(pt1, angle, size)
        pt3 = self.project(pt2, angle-math.pi+thirty, size/4)
        pt4 = self.project(pt2, angle-math.pi-thirty, size/4)
        pygame.draw.line(self.screen, (255, 0, 0), self.c_pt(pt1), self.c_pt(pt2))
        pygame.draw.line(self.screen, (255, 0, 0), self.c_pt(pt2), self.c_pt(pt3))
        pygame.draw.line(self.screen, (255, 0, 0), self.c_pt(pt2), self.c_pt(pt4))

    def init_rendering(self):
        pygame.init()
        self.size = self.maxi*2
        self.misize = self.maxi
        self.screen = pygame.display.set_mode((self.size, self.size))
        self.draw()

    def plot_acts(self):
        plt.ion()
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        for i, traj in enumerate(self.acts):
            u = (255-i)/255
            ax2.plot(range(len(traj)), [t[2] for t in self.obs[i]], c=(1-u, 1-u, 1-u))
            ax1.plot(range(len(traj)), [act[0]-act[1] for act in traj], c=(u, 0, 0))
            ax1.plot(range(len(traj)), [act[0]+act[1] for act in traj], c=(0, u, 0))
            ax2.plot(range(len(traj)), [t[0] for t in self.obs[i]], c=(0, 0, u))
            ax2.plot(range(len(traj)), [t[1] for t in self.obs[i]], c=(0, 0, u))
            ax2.plot(range(len(traj)), [t[3] for t in self.obs[i]], c=(u, 0, u))
            ax2.plot(range(len(traj)), [t[4] for t in self.obs[i]], c=(u, 0, u))
        plt.show(block=True)

    def draw(self):
        self.screen.fill('white')
        W1 = self.project(self.current[0], self.current[1], self.dist/2)
        W2 = self.project(self.current[0], self.current[1], -self.dist/2)
        pygame.draw.circle(self.screen, (0, 0, 0), (self.misize, self.misize), self.eps)
        pygame.draw.circle(self.screen, (0, 0, 255), self.c_pt(self.current[0]), 2)
        pygame.draw.line(
            self.screen,
            (0, 0, 255),
            self.c_pt(W1),
            self.c_pt(W2)
        )
        self.draw_arrow(W1, self.curVel[0]*5, math.pi/2+self.current[1])
        self.draw_arrow(W2, self.curVel[1]*5, math.pi/2+self.current[1])
        pygame.display.flip()

    def clipVel(self, vel):
        return max(min(vel, self.maxSpeed), -self.maxSpeed)

    def get_obs(self):
        #length = abs(self.current[0])
        #return get_arr(length/self.maxi, cmath.log(self.current[0]/length).imag/math.pi, self.curVel[0]/self.maxSpeed, self.curVel[1]/self.maxSpeed)
        return get_arr(self.current[0].real/self.maxi, self.current[0].imag/self.maxi, self.current[1]/math.pi-1, self.curVel[0]/self.maxSpeed, self.curVel[1]/self.maxSpeed)

    def step(self, acc):
        #acc1, acc2 = acc
        #comPenality = (acc1**2+acc2**2)*10
        #acc += np.random.randn(2)/100
        if self.get_action:
            self.acts[-1].append(acc)
        self.curVel += acc*self.refresh
        self.curVel = np.array(list(map(self.clipVel, self.curVel)), dtype=np.float64)
        if self.curVel[0] == self.curVel[1]:
            self.current[0] += cmath.exp((self.current[1]+math.pi/2)*1j)*self.curVel[0]*self.refresh
        else:
            v1, v2 = self.curVel*self.refresh#*self.radius
            w = (v1-v2)/self.dist
            r1 = v1/w
            r2 = r1-self.dist
            C = self.current[0]+(self.dist/2-r1)*cmath.exp(1j*self.current[1])
            #print(r1, C, self.current[0], abs(self.current[0]))
            Ap = C+(self.current[0]-C)*cmath.exp(1j*w)
            self.current = [Ap, math.fmod(self.current[1]+w, math.pi*2)]
        dist = abs(self.current[0])
        distA = abs(self.current[1])
        self.counter += 1
        final_obs = self.get_obs()
        if self.get_action:
            self.obs[-1].append(final_obs)
        if dist < self.eps:# and distA < self.epsAngle:
            reward, done, trunc = self.limitStep, True, False
        elif dist > self.maxi:
            reward, done, trunc = -self.limitStep, False, True
        elif self.counter > self.limitStep:
            reward, done, trunc = -2, False, True
        else:
            reward = math.exp(-2*dist**2/self.maxi**2)-1
            #r = 1-dist**2/self.maxi**2
            #r = math.exp(self.lastDist-dist)-1 if self.lastDist < dist else dist-self.lastDist
            #r = -dist**2/self.maxi**2
            if dist < self.dist:
                reward = math.exp(-(self.dist-dist)**2)
            else:
                reward = 0#-(self.dist-dist+1)
            done, trunc = False, False
            reward -= 1
        self.lastDist = dist
        self.somreward += reward
        return final_obs, reward, done, trunc, {}

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        rA = random.random()*2*math.pi
        while True:
            L = (random.random()-0.5)*2*self.stLengthMax
            if L > self.eps:break
        self.current = [self.project(0j, rA, L), random.random()*2*math.pi]
        self.curVel = np.array([0, 0], dtype=np.float64)
        self.lastDist = abs(self.current[0])
        if hasattr(self, 'somreward') and self.logfile is not None:
            with open(self.logfile, 'a') as f:
                f.write(f'{self.somreward} {self.counter} {time.time()}\n')
        self.somreward = 0
        self.counter = 0
        if self.render_mode:
            self.draw()
        self.acts.append([])
        self.obs.append([])
        return self.get_obs(), {}
    
    def render(self, render_mode='human'):
        if isinstance(render_mode, str):
            render_mode = render_mode == 'human'
        assert render_mode is self.render_mode
        if self.render_mode and self.counter%self.every == self.every-1:
            self.draw()

    def close(self):
        if self.render_mode:
            pygame.quit()

def make_env(id):
    def _init():
        return Monitor(Car(id=id))
    return _init

class Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(Policy, self).__init__(*args, **kwargs)
        self.mlp_extractor.policy_net[-1] = nn.Tanh()

def get_exact_acc(point, angle, dist):
    cp = complex(*point)
    length, ap = cmath.polar(cp)
    a = math.pi-angle+ap
    l = length/2
    m = l/math.cos(a)
    r = l/m
    d = dist/2
    acc = np.array([(m-d)*r, (m+d)*r])
    acc /= abs(acc).max()
    return acc, length, a

class Vector:
    def __init__(self, x, y=None):
        if y is None:
            c = cmath.exp(x*1j)
            self.x = c.real
            self.y = c.imag
        else:
            self.x = x
            self.y = y
    
    def scalar(self, other):
        return self.x*other.x+self.y*other.y
    
    def __sub__(self, other):
        return Vector(self.x-other.x, self.y-other.y)

    def __add__(self, other):
        return Vector(self.x+otherx, self.y+other.y)

def proved_exact(env):
    state, info = env.reset()
    A = Vector(0, 0)
    kw = -0.05
    ka = -0.01
    rlength, ra = cmath.polar(complex(*(state[:2]*env.maxi)))
    while True:
        V1, V2 = state[-2:]*env.maxSpeed
        B = Vector(state[0]*env.maxi, state[1]*env.maxi)
        angle = state[2]*math.pi
        d = (B-A).scalar(Vector(angle+math.pi/2))
        length, ap = cmath.polar(complex(*(state[:2]*env.maxi)))
        psi = math.fmod(math.pi-angle+ap, math.pi*2)-math.pi
        v1 = -(kw*psi+ka*d)
        v2 =  (kw*psi-ka*d)
        acc = get_arr(v1-V1, v2-V2)/env.refresh
        print(v1, v2, V1, V2, acc)
        state, reward, done, trunc, info = env.step(acc)
        env.render()
        if done or trunc:
            break
    return env.somreward, rlength, ra

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "exact":
        env = Car(render_mode='human', every=100)
        random.seed(42)
        somreward = 0
        n = 0
        good = 0
        reds = []
        greens = []
        for i in range(100):
            start, info = env.reset()
            acc, length, a = get_exact_acc(start[:2]*env.maxi, start[3]*math.pi, env.dist)
            stacc = acc.copy()
            #print(acc, m, env.current[1], a, env.lastDist, length, r)
            while True:
                ns, r, done, trunc, info = env.step(acc)
                env.render()
                
                assert abs(ns[3]/ns[4]-stacc[0]/stacc[1]) < 0.01, (acc, ns)
                if max(abs(ns[3:]*env.maxSpeed+acc*env.refresh)) > env.maxSpeed:
                    acc = np.array([0, 0], dtype=np.float64)
                #assert not (done or trunc)
                if done or trunc:break
            somreward += env.somreward
            n += 1
            good += env.somreward > 0
            if env.somreward > 0:
                greens.append(cmath.rect(length, a))
            else:
                reds.append(cmath.rect(length, a))
        print(f'{somreward} {round(somreward/n, 2)} -> {good}%')
        A = np.array(greens)
        B = np.array(reds)
        plt.scatter(A.real, A.imag, c='green')
        plt.scatter(B.real, B.imag, c='red')
        plt.show(block=True)
        sys.exit()
    elif len(sys.argv) > 1 and sys.argv[1] == 'exact2':
        reds = []
        greens = []
        env = Car(render_mode='human')
        somreward = 0
        n = 0
        random.seed(42)
        goods = 0
        for i in range(100):
            sr, length, a = proved_exact(env)
            goods += sr > 0
            somreward += sr
            n += 1
            if sr > 0:
                greens.append(cmath.rect(length, a))
            else:
                reds.append(cmath.rect(length, a))
        print(f'{somreward} {round(somreward/n, 2)} -> {goods}%')
        A = np.array(greens)
        B = np.array(reds)
        plt.scatter(A.real, A.imag, c='green')
        plt.scatter(B.real, B.imag, c='red')
        plt.show(block=True)
        sys.exit()
    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        random.seed(42)
        env = Car(render_mode='human', every=1)
        u = 0.1
        for command in np.array([[-u, u], [u, 0.0], [0.0, -u], [u, u*2], [u, u]]):
            env.reset()
            while True:
                ns, r, done, trunc, info = env.step(command)
                env.render()
                if trunc:break
            print(env.counter)
        sys.exit()
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    if len(sys.argv) <= 1:
        #parEnv = make_vec_env(env, n_envs=2)
        policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                        net_arch=dict(pi=[16, 16], vf=[16, 16]))
        nb_envs = 6
        vecEnv = SubprocVecEnv([make_env(id) for id in range(nb_envs)])
        model = PPO(Policy, vecEnv, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.003)
        model.learn(total_timesteps=int(2e6), progress_bar=True)
        model.save('ppo_car.zip')
        env = Car(render_mode='human', actions=True)
        plt.ion()
        dataE = []
        for i in range(nb_envs):
            with open(f'logs/Car_{i}') as f:
                data = f.readlines()
                data = [tuple(map(float, line.split())) for line in data]
            dataE.extend(data)
        dataE.sort(key=lambda l:l[2])
        plt.plot(range(len(dataE)), [i[0] for i in dataE])
        plt.plot(range(len(dataE)), [i[1] for i in dataE])
        plt.show(block=True)
    else:
        model = PPO.load(sys.argv[1])
        env = Car(render_mode="human", every=100, actions=True)
    mean_reward, std_reward = evaluate_policy(model, Monitor(env), render=env.render_mode, n_eval_episodes=100)
    if(env.get_action):
        env.plot_acts()
    print(f"{mean_reward}±{std_reward}")
    vec_env = env
    obs, _ = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(np.array(obs), deterministic=True)
        obs, rewards, done, truncated, info = vec_env.step(action)
        vec_env.render("human")
        if done or truncated:
            break
    env.close()

