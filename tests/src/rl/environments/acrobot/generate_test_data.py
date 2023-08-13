import gymnasium as gym
from gymnasium.envs.classic_control.acrobot import AcrobotEnv, rk4, wrap, bound
from stable_baselines3 import TD3, PPO
import numpy as np
import h5py
import time


env = gym.make('Acrobot-v1')

class AcrobotContinuousEnv(AcrobotEnv):
    def __init__(self):
        super().__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
    
    def step(self, action):
        s = self.state
        assert s is not None, "Call reset before using AcrobotEnv object."
        torque = np.clip(action[0], -1, 1)

        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(
                -self.torque_noise_max, self.torque_noise_max
            )

        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, self.dt])

        ns[0] = wrap(ns[0], -np.pi, np.pi)
        ns[1] = wrap(ns[1], -np.pi, np.pi)
        ns[2] = bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminated = self._terminal()
        reward = -1.0 if not terminated else 0.0

        if self.render_mode == "human":
            self.render()
        return (self._get_ob(), reward, terminated, False, {})


if __name__ == "__main__":
    env = AcrobotContinuousEnv()#render_mode="human")

    with h5py.File("tests/data/rl_environments_acrobot_test_data.h5", "w") as f:
        episodes_group = f.create_group("episodes")
        for episode_i in range(10):
            group = episodes_group.create_group(f"{episode_i}")
            actions = np.clip(np.random.randn(100, 1), -1, 1)

            observations = []
            states = []
            next_observations = []
            next_states = []
            rewards = []
            terminateds = []
            truncateds = []
            infos = []
            observations.append(env.reset(seed=episode_i)[0])
            states.append(env.state.copy())
            for action in actions:
                observation, reward, terminated, truncated, info = env.step(action)
                observations.append(observation)
                next_observations.append(observation)
                next_states.append(env.state.copy())
                states.append(env.state.copy())
                rewards.append(reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                infos.append(info)

            observations = np.array(observations[:-1])
            states = np.array(states[:-1])
            next_observations = np.array(next_observations)
            next_states = np.array(next_states)
            rewards = np.array(rewards)
            terminateds = np.array(terminateds)
            truncateds = np.array(truncateds)

            group.create_dataset("observations", data=observations)
            group.create_dataset("states", data=states)
            group.create_dataset("next_observations", data=next_observations)
            group.create_dataset("next_states", data=next_states)
            group.create_dataset("rewards", data=rewards)
            group.create_dataset("terminated", data=terminateds)
            group.create_dataset("truncated", data=truncateds)
            group.create_dataset("actions", data=actions)
        

# if __name__ == "__main__":
#     env = AcrobotEnv(render_mode="human")
#     obs, info = env.reset(seed=0)
#     env.state[0] = 0.3
#     env.state[1] = 0.3
#     env.render()
#     time.sleep(1000)