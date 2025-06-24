import gymnasium
import numpy as np
import math
import cmath
import random
import pygame
import torch
thirty = 30/180*math.pi

def get_arr(*arr):
    return np.array(arr, dtype=np.float32)

class Car(gymnasium.Env):
    def __init__(self, render_mode=None):
        assert render_mode is None or render_mode == 'human', f"Car env dosen't support the {render_mode} rendering"
        self.render_mode = False
        self.maxSpeed = 20
        self.reset()
        self.render_mode = render_mode == 'human'
        self.refresh = 0.08
        self.radius = 50
        self.dist = 120
        self.W1 = self.dist+0j
        self.W2 = -self.dist+0j
        self.eps = 1e+1
        self.epsAngle = 0.1
        self.observation_space = gymnasium.spaces.Box(low=get_arr(-1, -1, -1, -1, -1), high=get_arr(1, 1, 1, 1, 1), shape=(5,))
        self.action_space = gymnasium.spaces.Box(low = get_arr(-1, -1), high=get_arr(1, 1), shape=(2,))
        self.limitStep = 10**3
        if self.render_mode:
            self.init_rendering()
    
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
        self.size = 500
        self.misize = self.size//2
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
        acc1, acc2 = acc
        comPenality = (acc1**2+acc2**2)*10
        self.curVel[0] += acc1*self.refresh
        self.curVel[1] += acc2*self.refresh
        self.curVel = list(map(self.clipVel, self.curVel))
        if self.curVel[0] == self.curVel[1]:
            self.current[0] += cmath.exp(self.current[1])*self.curVel[0]*self.refresh
        else:
            v1, v2 = self.curVel
            v1 *= self.refresh
            v2 *= self.refresh
            w = (v1-v2)/self.dist
            r1 = v1/w
            #r2 = r1-dist
            C = (self.dist/2-r1)*self.current[0]*cmath.exp(1j*self.current[1])
            Ap = C+(self.current[0]-C)*cmath.exp(1j*w)
            self.current = [Ap, math.fmod(self.current[1]+math.pi-w, math.pi*2)]
        dist = abs(self.current[0])
        distA = abs(self.current[1])
        self.counter += 1
        if dist < self.eps:# and distA < self.epsAngle:
            return self.get_obs(), 1-comPenality, True, False, {}
        if dist > 250 or any(abs(i) > 250 for i in self.curVel) or self.counter > self.limitStep:
            return self.get_obs(), -1000, False, True, {}
        self.render()
        #return self.get_obs(), -(dist+distA*10/(2*math.pi)), False, False, {}
        return self.get_obs(), -dist-comPenality, False, False, {}

    def get_obs(self):
        return get_arr(self.current[0].real/250, self.current[0].imag/250, self.current[1]/math.pi-1, self.curVel[0]/self.maxSpeed, self.curVel[1]/self.maxSpeed)

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.current = [random.random()*250-125+random.random()*250j-125j, random.random()*2*math.pi]
        self.curVel = [0, 0]
        self.counter = 0
        if self.render_mode:
            self.draw()
        return self.get_obs(), {}
    
    def render(self, render_mode='human'):
        assert (render_mode == 'human') is self.render_mode
        if self.render_mode:
            self.draw()

    def close(self):
        if self.render_mode:
            pygame.quit()

if __name__ == '__main__':
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common import vec_env

    env = Car(render_mode='human')
    check_env(env, warn=True)
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=dict(pi=[32, 32], vf=[32, 32]))
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=int(2e5), progress_bar=True)
    model.save('ppo_car.zip')
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"{mean_reward}+-{std_reward}")

    vec_env = env
    obs = vec_env.reset()
    for i in range(1000):
        action, _states = model.predict(np.array(obs), deterministic=True)
        obs, rewards, done, truncated, info = vec_env.step(action)
        vec_env.render("human")
        if done or truncated:
            break
    obs.close()
