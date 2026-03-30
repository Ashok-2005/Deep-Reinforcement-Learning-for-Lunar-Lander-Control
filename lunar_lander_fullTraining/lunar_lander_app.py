import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import io
import gradio as gr
from PIL import Image

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

def simulate(episodes=1, steps=500):
    """Simulate the Lunar Lander environment and return final frame."""
    frames = []
    for ep in range(episodes):
        obs, _ = env.reset()
        for t in range(steps):
            action = env.action_space.sample()  # Random actions
            obs, reward, done, truncated, _ = env.step(action)
            frame = env.render()
            frames.append(frame)
            if done or truncated:
                break
    env.close()
    # Return the last frame as an image
    img = Image.fromarray(frames[-1])
    return img

# Gradio interface
demo = gr.Interface(
    fn=simulate,
    inputs=[
        gr.Slider(1, 10, value=1, step=1, label="Episodes"),
        gr.Slider(100, 1000, value=500, step=50, label="Max Steps per Episode")
    ],
    outputs=gr.Image(type="pil"),
    title="🚀 Lunar Lander Simulation",
    description="Run the OpenAI Gym LunarLander-v3 environment and see how it lands!"
)

if __name__ == "__main__":
    demo.launch()
