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
GROUND_FREEZE = 2.0  # seconds to freeze after landing

# ------------------- MODEL -------------------
class STM_DQNNet(torch.nn.Module):
    def __init__(self, obs, act, h=128):
        super().__init__()
        self.lstm = torch.nn.LSTM(obs, h, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(h, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, act)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])

# ------------------- ENV & MODEL -------------------
env = gym.make("LunarLander-v3", render_mode='rgb_array')
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

model = STM_DQNNet(obs_size, n_actions)
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
model.eval()

seq = deque(maxlen=SEQ_LEN)

# ------------------- STATE -------------------
landing_records = []
successful_landings = 0
running = False
terminated = False
status_text = "Idle"

# ------------------- CONTROL FUNCTIONS -------------------
def start_auto():
    global running, terminated, status_text
    running = True
    terminated = False
    status_text = "🟢 Auto-Landing Started"
    return status_text

def stop_auto():
    global running, terminated, status_text
    running = False
    terminated = False
    status_text = "🟡 Auto-Landing Stopped"
    return status_text

def terminate_auto():
    """Show summary only when Terminate is pressed and keep visible."""
    global landing_records, successful_landings, running, terminated
    running = False
    terminated = True

    total = len(landing_records)
    fails = total - successful_landings
    msg_summary = f"\n\n---\n✅ Total Success: {successful_landings} | ❌ Total Fail: {fails}"

    text_records = "\n".join(
        [f"{i+1}: {'✅' if r else '❌'}" for i, r in enumerate(landing_records)]
    ) + msg_summary

    return text_records

# ------------------- AUTO LANDING LOOP -------------------
def auto_landing_generator():
    global running, terminated, seq, landing_records, successful_landings, status_text

    while True:
        if not running:
            # Idle display
            img = Image.new("RGB", (400, 400), color="black")
            if terminated:
                total = len(landing_records)
                fails = total - successful_landings
                msg_summary = f"\n\n---\n✅ Total Success: {successful_landings} | ❌ Total Fail: {fails}"
                text_records = "\n".join(
                    [f"{i+1}: {'✅' if r else '❌'}" for i, r in enumerate(landing_records)]
                ) + msg_summary
            else:
                text_records = "\n".join(
                    [f"{i+1}: {'✅' if r else '❌'}" for i, r in enumerate(landing_records)]
                )

            yield img, text_records, status_text
            time.sleep(0.1)
            continue

        # Reset environment
        s, _ = env.reset()
        seq.clear()
        for _ in range(SEQ_LEN - 1):
            seq.append(np.zeros(obs_size))
        seq.append(s)

        done = False
        landed_time = None
        status_text = "🟢 Landing in progress..."

        # ------------------- MAIN CONTROL LOOP -------------------
        while not done and running:
            inp = torch.tensor(np.array(seq)[None, :, :], dtype=torch.float32)
            model_action = int(model(inp).argmax(1).item())
            a = model_action

            # Optional gentle correction if way off-center
            if abs(s[0]) > 0.4 and abs(s[2]) < 0.3:
                a = 1 if s[0] > 0 else 3

            ns, _, terminated_flag, truncated, _ = env.step(a)
            done = terminated_flag or truncated
            s = ns
            seq.append(s)

            # Ground contact detection (y near 0)
            if s[1] < 0.05 and abs(s[3]) < 0.1:
                if landed_time is None:
                    landed_time = time.time()
                elif time.time() - landed_time >= GROUND_FREEZE:
                    done = True
            else:
                landed_time = None

            frame = env.render()
            img = Image.fromarray(frame)
            text_records = "\n".join(
                [f"{i+1}: {'✅' if r else '❌'}" for i, r in enumerate(landing_records)]
            )

            yield img, text_records, status_text
            time.sleep(0.05)

        # ------------------- POST LANDING EVALUATION -------------------
        legs_down = s[6] == 1.0 and s[7] == 1.0
        upright = abs(s[4]) < 0.3
        centered = abs(s[0] - PAD_CENTER_X) <= CENTER_TOLERANCE
        landing_success = legs_down and upright and centered

        if landing_success:
            successful_landings += 1
        landing_records.append(landing_success)

        # Freeze display after landing
        freeze_start = time.time()
        while time.time() - freeze_start < GROUND_FREEZE:
            frame = env.render()
            img = Image.fromarray(frame)
            text_records = "\n".join(
                [f"{i+1}: {'✅' if r else '❌'}" for i, r in enumerate(landing_records)]
            )
            yield img, text_records, "🧊 Lander frozen after landing..."
            time.sleep(0.05)

        running = False
        status_text = "🔵 Idle – ready for next run."

# ------------------- GRADIO INTERFACE -------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🌕 Auto Lunar Lander Simulator (STM-DQN)")
    gr.Markdown("DQN-based auto-lander with LSTM memory. Click start to see it land.")

    with gr.Row():
        output_image = gr.Image(type="pil", label="Lunar Lander View")
        with gr.Column():
            output_text = gr.Textbox(label="Landing Records", lines=10)
            status_box = gr.Textbox(label="Status", interactive=False)

    with gr.Row():
        btn_start = gr.Button("Start Auto-Landing")
        btn_stop = gr.Button("Stop Auto-Landing")
        btn_terminate = gr.Button("Terminate")

    btn_start.click(fn=start_auto, outputs=status_box)
    btn_stop.click(fn=stop_auto, outputs=status_box)
    btn_terminate.click(fn=terminate_auto, outputs=output_text)

    demo.load(fn=auto_landing_generator, outputs=[output_image, output_text, status_box])

demo.launch()
