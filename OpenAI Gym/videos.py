import imageio
import numpy as np

from stable_baselines import PPO

model = PPO.load("D:/SoftUni/Deep Learning/Run Data/Run 1/PPO/Final Model/PPO_2275000.zip")

images = []
obs = model.env.reset()
img = model.env.render(mode='rgb_array')
for i in range(350):
    images.append(img)
    action, _ = model.predict(obs)
    obs, _, _ ,_ = model.env.step(action)
    img = model.env.render(mode='rgb_array')

imageio.mimsave('Videos/Run 1/ppo.gif', [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)
