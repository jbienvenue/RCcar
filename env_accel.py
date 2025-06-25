import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import random
import pygame
import torch
import sys
thirty = 30/180*math.pi

def get_arr(*arr):
    return np.array(arr, dtype=np.float32)

class Car(gymnasium.Env):
    maxSpeed = 5
    maxi = 250
    dist = 120
    radius = 50
    refresh = 0.02*7
    eps = 1e+1
    epsAngle = 0.1
    def __init__(self, render_mode=None):
        self.mimax = self.maxi/2
        assert render_mode is None or render_mode == 'human', f"Car env dosen't support the {render_mode} rendering"
        self.render_mode = False
        self.reset()
        self.render_mode = render_mode == 'human'
        self.W1 = self.dist+0j
        self.W2 = -self.dist+0j
        self.observation_space = gymnasium.spaces.Box(low=get_arr(-1, -1, -1, -1, -1), high=get_arr(1, 1, 1, 1, 1), shape=(5,))
        self.action_space = gymnasium.spaces.Box(low = get_arr(-1, -1), high=get_arr(1, 1), shape=(2,))
        self.limitStep = 10**3
        if self.render_mode:
            self.init_rendering()
        self.datas = [[], []]
    
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
        print("initialised")

    def draw(self):
        self.screen.fill('white')
        W1 = self.project(self.current[0], self.current[1], self.dist/2)
        W2 = self.project(self.current[0], self.current[1], -self.dist/2)
        pygame.draw.circle(self.screen, (0, 0, 0), (self.misize, self.misize), self.eps)
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

    def step(self, acc):
        #acc1, acc2 = acc
        #comPenality = (acc1**2+acc2**2)*10
        self.curVel += acc*self.refresh
        self.curVel = np.array(list(map(self.clipVel, self.curVel)), dtype=np.float64)
        if self.curVel[0] == self.curVel[1]:
            self.current[0] += cmath.exp((self.current[1]+math.pi/2)*1j)*self.curVel[0]*self.refresh
        else:
            v1, v2 = self.curVel*self.refresh
            w = (v1-v2)/self.dist
            r1 = v1/w
            #r2 = r1-self.dist
            C = self.current[0]+(self.dist/2-r1)*cmath.exp(1j*self.current[1])
            #print(r1, C, self.current[0], abs(self.current[0]))
            Ap = C+(self.current[0]-C)*cmath.exp(1j*w)
            self.current = [Ap, math.fmod(self.current[1]+w, math.pi*2)]
        dist = abs(self.current[0])
        distA = abs(self.current[1])
        self.counter += 1
        if dist < self.eps:# and distA < self.epsAngle:
            self.somreward += 1
            return self.get_obs(), 1000, True, False, {}
        if dist > self.maxi or self.counter > self.limitStep:
            self.somreward -= 1
            return self.get_obs(), -1, False, True, {}
        #return self.get_obs(), -(dist+distA*10/(2*math.pi)), False, False, {}
        reward = math.exp(-dist**2/self.mimax**2)#-comPenality
        self.lastDist = dist
        self.somreward += reward
        return self.get_obs(), reward, False, False, {}

    def get_obs(self):
        return get_arr(self.current[0].real/250, self.current[0].imag/250, self.current[1]/math.pi-1, self.curVel[0]/self.maxSpeed, self.curVel[1]/self.maxSpeed)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.current = [(random.random()-0.5)*self.maxi+(random.random()-0.5)*self.maxi*1j, random.random()*2*math.pi]
        self.curVel = np.array([0, 0], dtype=np.float64)
        self.lastDist = abs(self.current[0])
        if hasattr(self, 'somreward'):
            self.datas[0].append(self.somreward)
            self.datas[1].append(self.counter)
        self.somreward = 0
        self.counter = 0
        if self.render_mode:
            self.draw()
        return self.get_obs(), {}
    
    def render(self, render_mode='human'):
        if isinstance(render_mode, str):
            render_mode = render_mode == 'human'
        assert render_mode is self.render_mode
        if self.render_mode:
            self.draw()

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        env = Car(render_mode='human')
        while True:
            ns, r, done, trunc, info = env.step(np.array((0.001, 0.0)))
            assert not (done or trunc)
            env.counter %= 10
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.env_util import make_vec_env
    if len(sys.argv) <= 1:
        #parEnv = make_vec_env(env, n_envs=2)
        policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                        net_arch=dict(pi=[128, 128], vf=[128, 128]))
        envs = [Car(render_mode=None) for i in range(4)]
        vecEnv = SubprocVecEnv([lambda :Monitor(Car()) for e in envs])
        model = PPO("MlpPolicy", vecEnv, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.03)
        model.learn(total_timesteps=int(2e5), progress_bar=True)
        model.save('ppo_car.zip')
        env = Car(render_mode='human')
        plt.ion()
        for e in envs:
            plt.plot(range(len(e.datas[0])), e.datas[0])
            plt.plot(range(len(e.datas[1])), e.datas[1])
        plt.show(block=True)
    else:
        model = PPO.load(sys.argv[1])
        env = Car(render_mode='human')
        env.limitStep *= 10
    mean_reward, std_reward = evaluate_policy(model, Monitor(env), render=True, n_eval_episodes=100)
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

