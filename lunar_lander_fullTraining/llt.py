import gymnasium as gym
import torch
import numpy as np
from collections import deque
from PIL import Image
import gradio as gr
import time

# ------------------- CONFIG -------------------
BEST_MODEL_PATH = r"stm_dqn.pth"
SEQ_LEN = 8
PAD_CENTER_X = 0.0
PAD_WIDTH = 0.4
CENTER_TOLERANCE = PAD_WIDTH / 2
TARGET_LANDINGS = 5

# ------------------- MODEL -------------------
class STM_DQNNet(torch.nn.Module):
    def __init__(self, obs, act, h=128):
        super().__init__()
        self.lstm = torch.nn.LSTM(obs, h, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(h,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,act)
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        o,_ = self.lstm(x)
        return self.fc(o[:,-1,:])

# ------------------- ENV & MODEL -------------------
env = gym.make("LunarLander-v3", render_mode='rgb_array')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

model = STM_DQNNet(obs_size, n_actions)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
model.eval()

seq_model = True
seq = deque(maxlen=SEQ_LEN)

# ------------------- STATE -------------------
landing_records = []
successful_landings = 0
first_landing_done = False
running = False

# ------------------- AUTO LANDER -------------------
def start_auto():
    global running
    running = True
    return "Auto-Landing Started"

def stop_auto():
    global running
    running = False
    return "Auto-Landing Stopped"

def terminate_auto():
    msg = f"✅ Total Success: {successful_landings} | ❌ Total Fail: {len(landing_records)-successful_landings}"
    return msg

def auto_landing_generator():
    global running, seq, landing_records, successful_landings, first_landing_done

    while True:
        if not running:
            img = Image.new("RGB", (400, 400), color="black")
            yield img, "\n".join([f"{i+1}: {'✅' if r else '❌'}" for i,r in enumerate(landing_records)])
            time.sleep(0.1)
            continue

        # Reset environment
        s,_ = env.reset()
        done = False
        seq.clear()
        for _ in range(SEQ_LEN-1):
            seq.append(np.zeros(obs_size))
        seq.append(s)

        while not done and running:
            inp = torch.tensor(np.array(seq)[None,:,:], dtype=torch.float32)
            a = int(model(inp).argmax(1).item())
            ns, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            s = ns
            if seq_model:
                seq.append(s)

            frame = env.render()
            img = Image.fromarray(frame)
            records_text = "\n".join([f"{i+1}: {'✅' if r else '❌'}" for i,r in enumerate(landing_records)])
            yield img, records_text
            time.sleep(0.05)

        # Check landing success
        landing_success = abs(s[0]-PAD_CENTER_X) <= CENTER_TOLERANCE
        if landing_success:
            successful_landings += 1
        landing_records.append(landing_success)

        if not first_landing_done:
            first_landing_done = True
            print(f"First landing success: {landing_success}")

        if first_landing_done and (landing_success or successful_landings >= TARGET_LANDINGS):
            running = False

# ------------------- GRADIO INTERFACE -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌕 Auto Lunar Lander Simulator")
    gr.Markdown("STM-DQN auto-lands the Lunar Lander. Only Start, Stop, Terminate. Records outside the frame.")
    
    with gr.Row():
        btn_start = gr.Button("Start Auto-Landing")
        btn_stop = gr.Button("Stop Auto-Landing")
        btn_terminate = gr.Button("Terminate")

    output_image = gr.Image(type="pil")
    output_text = gr.Textbox(label="Landing Records", lines=10)

    btn_start.click(fn=start_auto, outputs=None)
    btn_stop.click(fn=stop_auto, outputs=None)
    btn_terminate.click(fn=terminate_auto, outputs=output_text)

    demo.load(fn=auto_landing_generator, outputs=[output_image, output_text])

demo.launch()
