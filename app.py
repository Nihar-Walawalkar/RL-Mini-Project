"""
Snake RL Comparison Lab
━━━━━━━━━━━━━━━━━━━━━━━
Features
  • Q-Learning, SARSA, Double Q-Learning, DQN (Deep RL)
  • Real-time live training charts (reward + score, updates every N eps)
  • Model save (download) & load (upload) for all agent types
  • Side-by-side greedy playback after training
  • Full performance analytics: learning curves, convergence, Q-distribution, comparison table
"""

import io
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from snake_env import SnakeEnv
from agents import QLearningAgent, SARSAAgent, DoubleQLearningAgent
from train import train_agent, smooth, run_episode_visual

# ── Optional DQN (requires torch) ─────────────────────────────────────────────
try:
    from dqn_agent import DQNAgent, TORCH_AVAILABLE
    if TORCH_AVAILABLE:
        import torch
except Exception:
    TORCH_AVAILABLE = False
    DQNAgent = None

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Snake RL Lab",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS  — loaded from style.css so Streamlit's sanitizer doesn't mangle it
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_css_path = _os.path.join(_os.path.dirname(__file__), "style.css")
with open(_css_path) as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)
del _os, _css_path, _f

# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "Q-Learning":        "#6366f1",
    "SARSA":             "#10b981",
    "Double Q-Learning": "#f59e0b",
    "DQN":               "#ec4899",
}

MPL_BG     = "#080b14"
MPL_AX_BG  = "#0d1120"
MPL_GRID   = "#1e2540"
MPL_LABEL  = "#64748b"
MPL_TICK   = "#475569"
MPL_SPINE  = "#1e2540"

# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def hex_to_rgb(h: str):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255 for i in (0, 2, 4)]


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(MPL_AX_BG)
    ax.set_title(title, color=MPL_TEXT, fontsize=10, fontweight="600", pad=10)
    ax.set_xlabel(xlabel, color=MPL_LABEL, fontsize=8)
    ax.set_ylabel(ylabel, color=MPL_LABEL, fontsize=8)
    ax.tick_params(colors=MPL_TICK, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color(MPL_SPINE)
    ax.grid(alpha=0.12, color=MPL_GRID, linewidth=0.7)


def _legend(ax):
    ax.legend(facecolor=MPL_AX_BG, labelcolor=MPL_TEXT,
              framealpha=0.9, fontsize=8, edgecolor=MPL_SPINE)


def render_grid(frame, grid_size, title, body_color, game_over=False):
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor(MPL_BG)
    ax.set_facecolor(MPL_AX_BG)
    
    empty_color = [0.04, 0.06, 0.12]
    img = np.zeros((*frame.shape, 3))
    cmap = {0: empty_color, 1: hex_to_rgb(body_color),
            2: [1.0, 1.0, 1.0],   # Head color
            3: [1.0, 0.82, 0.0]}  # Food color
    for val, rgb in cmap.items():
        img[frame == val] = rgb
    
    if game_over:
        img *= 0.28
    
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color=MPL_BG, linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    
    for spine in ax.spines.values():
        spine.set_color(MPL_SPINE)
        
    score = int(np.sum(frame == 1) + np.sum(frame == 2) - 1)
    if game_over:
        ax.text(grid_size / 2, grid_size / 2, "GAME OVER",
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#ef4444",
                bbox=dict(boxstyle="round,pad=0.5", facecolor=MPL_BG,
                          alpha=0.95, edgecolor="#ef4444", linewidth=1.5))
        ax.set_title(f"{title}  ·  {score} pts", color="#ef4444", fontsize=9, pad=6)
    else:
        ax.set_title(f"{title}  ·  {score} pts", color=MPL_LABEL, fontsize=9, pad=6)
    plt.tight_layout(pad=0.4)
    return fig


def draw_live_chart(live_data: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
    fig.patch.set_facecolor(MPL_BG)
    has_data = False
    for name, data in live_data.items():
        eps = len(data["rewards"])
        if eps == 0: continue
        has_data = True
        color = ALGO_COLORS.get(name, "#888")
        x_axis = list(range(1, eps + 1))
        for ax, key in zip(axes, ("rewards", "scores")):
            raw = data[key]
            ax.plot(x_axis, raw, alpha=0.08, color=color, linewidth=0.7)
            sm = smooth(raw, min(20, max(1, eps // 10)))
            ax.plot(np.linspace(1, eps, len(sm)), sm, color=color, linewidth=2.0, label=name)
            
    style_ax(axes[0], "Live — Reward per Episode", "Episode", "Total Reward")
    style_ax(axes[1], "Live — Score per Episode",  "Episode", "Score")
    if has_data:
        _legend(axes[0])
        _legend(axes[1])
    plt.tight_layout(pad=1.2)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:0.5rem 0 1rem;'>
        <div style='font-size:2rem;'>🐍</div>
        <div style='font-size:1.1rem; font-weight:800;
             background:linear-gradient(135deg,#6366f1,#34d399);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;
             background-clip:text;'>Snake RL Lab</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**⚙️ Training Config**")
    episodes  = st.slider("Episodes", 200, 3000, 800, 100)
    grid_size = st.select_slider("Grid Size", [8, 10, 12], value=10)

    st.markdown("<br>**🎛️ Tabular RL**", unsafe_allow_html=True)
    alpha         = st.slider("Learning Rate (α)", 0.01, 0.50, 0.10, 0.01)
    gamma         = st.slider("Discount Factor (γ)", 0.80, 0.999, 0.95, 0.005)
    epsilon_decay = st.slider("Epsilon Decay", 0.990, 0.999, 0.995, 0.001)

    st.markdown("<br>**🤖 Algorithms**", unsafe_allow_html=True)
    use_qlearn = st.checkbox("Q-Learning",        value=True)
    use_sarsa = st.checkbox("SARSA", value=True)
    use_double = st.checkbox("Double Q-Learning", value=True)

    if TORCH_AVAILABLE:
        use_dqn = st.checkbox("DQN  🔮 Deep RL", value=False)
    else:
        st.checkbox("DQN (install torch)", value=False, disabled=True)
        use_dqn = False

    if use_dqn and TORCH_AVAILABLE:
        with st.expander("⚙️ DQN Config"):
            dqn_lr      = st.slider("Adam LR", 0.0001, 0.005, 0.001, 0.0001, format="%.4f")
            dqn_batch   = st.select_slider("Batch Size", [32, 64, 128], value=64)
            dqn_freq    = st.slider("Train Freq (steps)", 1, 10, 4)
            dqn_target  = st.slider("Target Sync", 50, 500, 200, 50)
            dqn_buffer  = st.select_slider("Buffer Capacity", [5000, 10000, 20000], value=10000)
    else:
        dqn_lr, dqn_batch, dqn_freq, dqn_target, dqn_buffer = 0.001, 64, 4, 200, 10000

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS Injection
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_css_path = _os.path.join(_os.path.dirname(__file__), "style.css")
with open(_css_path) as _f:
    _css_content = _f.read()
st.markdown(f"<style>{_css_content}</style>", unsafe_allow_html=True)
del _os, _css_path, _f

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "Q-Learning":        "#6366f1",
    "SARSA":             "#10b981",
    "Double Q-Learning": "#f59e0b",
    "DQN":               "#ec4899",
}

# Dark theme colors for Matplotlib
MPL_BG     = "#080b14"
MPL_AX_BG  = "#0d1120"
MPL_GRID   = "#1e2540"
MPL_LABEL  = "#64748b"
MPL_TICK   = "#475569"
MPL_SPINE  = "#1e2540"
MPL_TEXT   = "#e2e8f0"


# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-wrap'>
    <div class='hero-badge'>🧠 Reinforcement Learning · Tabular + Deep Methods</div>
    <div class='hero-title'>Snake RL Comparison Lab</div>
    <div class='hero-sub'>
        Train, compare, and watch four RL algorithms compete on a custom Snake environment.
        Live charts, model save/load, and deep learning (DQN) all built in.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Algorithm info cards
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)

_cards = [
    ("Q-Learning",        "🟣", "#6366f1", "Off-Policy",
     "Greedy Bellman backup. Learns the optimal policy fast, but the max operator causes Q-value overestimation."),
    ("SARSA",             "🟢", "#10b981", "On-Policy",
     "Updates with the action actually taken (ε-greedy). More conservative near danger; slower but more stable."),
    ("Double Q-Learning", "🟡", "#f59e0b", "Bias Correction",
     "Two Q-tables — one selects, one evaluates. Eliminates the overestimation bias present in standard Q-Learning."),
    ("DQN",               "🔮", "#ec4899", "Deep RL",
     "MLP approximates Q(s,a). Experience replay + target network produce stable, scalable learning." +
     ("" if TORCH_AVAILABLE else " <span style='color:#f59e0b;font-size:0.75rem;'>(requires torch)</span>")),
]

for col, (name, icon, color, tag, desc) in zip([c1, c2, c3, c4], _cards):
    r, g, b = [int(x * 255) for x in hex_to_rgb(color)]
    with col:
        st.markdown(f"""
        <div class='algo-card'>
            <div class='algo-icon'>{icon}</div>
            <div class='algo-name'>{name}</div>
            <span class='algo-tag' style='background:rgba({r},{g},{b},0.18);
                  color:{color}; border:1px solid {color}40;'>{tag}</span>
            <div class='algo-desc'>{desc}</div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Train button
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    st.markdown("""
    <div style='text-align:center; color:#475569; font-size:0.85rem; margin-bottom:0.6rem;'>
        Configure in the sidebar, then hit Train
    </div>""", unsafe_allow_html=True)
    train_btn = st.button("🚀  Train Agents", use_container_width=True, type="primary")

# ─────────────────────────────────────────────────────────────────────────────
# Training loop  (with LIVE CHART)
# ─────────────────────────────────────────────────────────────────────────────
if train_btn:
    selected = []
    if use_qlearn:                  selected.append(("Q-Learning",        QLearningAgent))
    if use_sarsa:                   selected.append(("SARSA",              SARSAAgent))
    if use_double:                  selected.append(("Double Q-Learning",  DoubleQLearningAgent))
    if use_dqn and TORCH_AVAILABLE: selected.append(("DQN",                DQNAgent))

    if not selected:
        st.warning("Select at least one algorithm from the sidebar.")
        st.stop()

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🏋️ Training Progress</div>", unsafe_allow_html=True)

    # ── Live chart placeholder ────────────────────────────────────────────────
    live_chart_ph = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)

    # Initialise live data storage
    live_data = {n: {"rewards": [], "scores": []} for n, _ in selected}
    UPDATE_EVERY = max(1, episodes // 40)   # refresh chart ~40 times per agent

    results = {}

    for name, AgentClass in selected:
        # ── Instantiate agent ─────────────────────────────────────────────────
        if AgentClass is DQNAgent:
            agent = DQNAgent(
                alpha=dqn_lr, gamma=gamma,
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=epsilon_decay,
                buffer_size=dqn_buffer, batch_size=dqn_batch,
                target_update=dqn_target,
                train_freq=dqn_freq,
            )
        else:
            agent = AgentClass(
                alpha=alpha, gamma=gamma,
                epsilon=1.0, epsilon_min=0.01,
                epsilon_decay=epsilon_decay,
            )

        color = ALGO_COLORS[name]
        st.markdown(
            f"<div style='color:#94a3b8; font-size:0.85rem; font-weight:600; margin-bottom:4px;'>"
            f"Training <span style='color:{color};'>{name}</span>…</div>",
            unsafe_allow_html=True,
        )
        prog = st.progress(0)
        info = st.empty()

        r_list = live_data[name]["rewards"]
        s_list = live_data[name]["scores"]

        def make_cb(bar, txt, total, n=name, col=color,
                    r=r_list, s=s_list, ld=live_data, ph=live_chart_ph):
            def cb(ep, reward, score):
                r.append(reward)
                s.append(score)
                bar.progress(ep / total)
                txt.markdown(
                    f"<div style='color:#475569; font-size:0.78rem; font-family:monospace;'>"
                    f"ep <b style='color:#e2e8f0;'>{ep:>4}/{total}</b>"
                    f"  &nbsp;·&nbsp;  reward <b style='color:{col};'>{reward:+.1f}</b>"
                    f"  &nbsp;·&nbsp;  score <b style='color:{col};'>{score}</b>"
                    f"  &nbsp;·&nbsp;  ε={r[-1] if r else 1.0:.4f}</div>",
                    unsafe_allow_html=True,
                )
                # Live chart refresh
                if ep % UPDATE_EVERY == 0 or ep == total:
                    fig = draw_live_chart(ld)
                    ph.pyplot(fig, use_container_width=True)
                    plt.close(fig)
            return cb

        res = train_agent(
            agent, episodes=episodes, grid_size=grid_size,
            callback=make_cb(prog, info, episodes),
        )
        results[name] = {**res, "color": color}
        prog.progress(1.0)
        info.markdown(
            f"<div style='color:#10b981; font-size:0.82rem; font-weight:600;'>"
            f"✅ {name} complete &nbsp;·&nbsp; "
            f"Memory / Q-states: {res['q_table_size']:,}</div>",
            unsafe_allow_html=True,
        )

    # Final chart after all agents done
    fig = draw_live_chart(live_data)
    live_chart_ph.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.session_state["results"]   = results
    st.session_state["grid_size"] = grid_size
    st.session_state["episodes"]  = episodes

# ─────────────────────────────────────────────────────────────────────────────
# Results section
# ─────────────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    results   = st.session_state["results"]
    grid_size = st.session_state["grid_size"]
    episodes  = st.session_state["episodes"]

    # ── Watch agents ──────────────────────────────────────────────────────────
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False

    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-wrap' style='padding:1.2rem 0 0.8rem;'>
        <div class='hero-badge'>🎮 Live Gameplay</div>
        <div class='hero-title' style='font-size:2rem;'>Watch Trained Agents Play</div>
        <div class='hero-sub' style='font-size:0.9rem;'>
            All trained agents run simultaneously in greedy mode (ε = 0).
        </div>
    </div>""", unsafe_allow_html=True)

    ctrl_col1, ctrl_col2, _ = st.columns([1, 1, 2])
    with ctrl_col1:
        play_speed = st.slider("Frame delay (ms)", 50, 400, 120, 10)
    
    with ctrl_col2:
        st.write("<div style='height:28px;'></div>", unsafe_allow_html=True)
        if not st.session_state.is_playing:
            if st.button("▶️  Play All Agents", use_container_width=True, type="primary"):
                st.session_state.is_playing = True
                st.rerun()
        else:
            if st.button("⏹️  Stop Simulation", use_container_width=True):
                st.session_state.is_playing = False
                st.rerun()

    # Placeholders for gameplay
    game_container = st.container()
    
    if st.session_state.is_playing:
        all_frames, all_scores, all_deaths = {}, {}, {}
        for name, res in results.items():
            f, s, d = run_episode_visual(res["agent"], grid_size)
            all_frames[name] = f
            all_scores[name] = s
            all_deaths[name] = d

        max_f   = max(len(v) for v in all_frames.values())
        names   = list(all_frames.keys())
        done_fl = {n: False for n in names}
        death_icons = {"wall": "🧱 Wall", "self": "🐍 Self-collision", "timeout": "⏱️ Timeout"}

        score_ph, game_ph = {}, {}
        with game_container:
            game_area, rules_panel = st.columns([3, 1])

            with rules_panel:
                st.markdown("""
                <div class='rules-card'>
                    <div style='font-size:0.92rem; font-weight:700; color:#e2e8f0; margin-bottom:0.8rem;'>
                        📋 Reward Structure
                    </div>
                    🍎 Eat food<br>
                    <span style='color:#10b981; font-weight:600; font-size:0.8rem;'>+10 reward</span>
                    <hr class='rules-divider'>
                    🧱 Hit wall<br>
                    <span style='color:#ef4444; font-weight:600; font-size:0.8rem;'>−10 · ends episode</span>
                    <hr class='rules-divider'>
                    🐍 Hit itself<br>
                    <span style='color:#ef4444; font-weight:600; font-size:0.8rem;'>−10 · ends episode</span>
                    <hr class='rules-divider'>
                    ➡️ Move toward food<br>
                    <span style='color:#94a3b8; font-weight:600; font-size:0.8rem;'>+0.1 shaping</span>
                    <hr class='rules-divider'>
                    <div style='font-size:0.78rem; color:#334155;'>
                        ⬜ White = Head<br>
                        🟩 Colored = Body<br>
                        🟨 Gold = Food
                    </div>
                </div>""", unsafe_allow_html=True)

            with game_area:
                s_row = st.columns(len(names))
                for i, n in enumerate(names):
                    with s_row[i]: score_ph[n] = st.empty()
                g_row = st.columns(len(names))
                for i, n in enumerate(names):
                    with g_row[i]: game_ph[n] = st.empty()

        for fi in range(max_f):
            # Crucial check: if user clicked stop, session_state.is_playing will be false in NEXT run,
            # but we can't easily detect the button click INSIDE this loop without st.rerun or fragments.
            # HOWEVER, Streamlit buttons in loops are tricky.
            # The best way in current version is to use a placeholder for the stop button that actually works.
            # But the Stop button already calls st.rerun(). 
            # If the user clicks "Stop", the script restarts, and is_playing is False, so the loop doesn't run.
            
            for name in names:
                frames  = all_frames[name]
                is_done = fi >= len(frames) - 1
                frame   = frames[min(fi, len(frames) - 1)]
                color   = results[name]["color"]
                if is_done and not done_fl[name]:
                    done_fl[name] = True
                fig = render_grid(frame, grid_size, name, color,
                                  game_over=done_fl[name])
                game_ph[name].pyplot(fig, use_container_width=True)
                plt.close(fig)
                cur = int(np.sum(frame == 1) + np.sum(frame == 2) - 1)
                
                if done_fl[name]:
                    dlabel = death_icons.get(all_deaths[name], "💀")
                    score_ph[name].markdown(
                        f"<div style='background:rgba(13,17,32,0.85); border:1px solid rgba(255,255,255,0.07);"
                        f"border-radius:12px; padding:0.8rem; text-align:center; margin-bottom:0.5rem;'>"
                        f"<div style='color:{color}; font-weight:700; font-size:0.82rem;'>{name}</div>"
                        f"<div style='font-size:1.8rem; font-weight:800; color:#ef4444;'>{cur}</div>"
                        f"<div style='color:#ef4444; font-size:0.75rem; font-weight:600; margin-top:4px;'>{dlabel}</div>"
                        f"</div>", unsafe_allow_html=True)
                else:
                    score_ph[name].markdown(
                        f"<div style='background:rgba(13,17,32,0.85); border:1px solid rgba(255,255,255,0.07);"
                        f"border-radius:12px; padding:0.8rem; text-align:center; margin-bottom:0.5rem;'>"
                        f"<div style='color:{color}; font-weight:700; font-size:0.82rem;'>{name}</div>"
                        f"<div style='font-size:1.8rem; font-weight:800; color:{color};'>{cur}</div>"
                        f"<div style='color:#10b981; font-size:0.75rem; font-weight:600; margin-top:4px;'>▶ Playing</div>"
                        f"</div>", unsafe_allow_html=True)
            
            time.sleep(play_speed / 1000)
            
        # End of loop
        st.session_state.is_playing = False
        st.rerun()

    # ── Performance summary ───────────────────────────────────────────────────
    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-wrap' style='padding:1rem 0 0.5rem;'>
        <div class='hero-badge'>📊 Analytics</div>
        <div class='hero-title' style='font-size:2rem;'>Performance Analysis</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🏆 Summary Metrics</div>", unsafe_allow_html=True)
    cols = st.columns(len(results))
    for i, (name, res) in enumerate(results.items()):
        with cols[i]:
            avg_sc = np.mean(res["scores"][-100:])
            mx_sc  = max(res["scores"])
            avg_rw = np.mean(res["rewards"][-100:])
            color  = res["color"]
            mem_label = "Buffer Size" if name == "DQN" else "Q-States"
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:{color}; font-size:0.85rem; font-weight:700; margin-bottom:4px;'>{name}</div>
                <div class='metric-value' style='color:{color};'>{avg_sc:.1f}</div>
                <div class='metric-label'>Avg Score · Last 100 eps</div>
                <hr style='border-color:var(--border-color); margin:0.8rem 0;'>
                <div class='metric-sub'>
                    <div>
                        <div class='metric-sub-val' style='color:{color};'>{mx_sc}</div>
                        <div class='metric-sub-label'>Best</div>
                    </div>
                    <div>
                        <div class='metric-sub-val'>{avg_rw:.1f}</div>
                        <div class='metric-sub-label'>Avg Reward</div>
                    </div>
                    <div>
                        <div class='metric-sub-val'>{res["q_table_size"]:,}</div>
                        <div class='metric-sub-label'>{mem_label}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Learning curves ───────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📈 Learning Curves</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["📦  Reward", "🍎  Score", "👣  Steps", "🎲  Epsilon"])

    def make_plot(key, ylabel, title, w=30):
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor(MPL_BG)
        for n, res in results.items():
            raw = res[key]
            sm  = smooth(raw, w)
            ax.plot(raw, alpha=0.08, color=res["color"], linewidth=0.8)
            ax.plot(np.linspace(0, len(raw), len(sm)), sm,
                    color=res["color"], linewidth=2.2, label=n)
        style_ax(ax, title, "Episode", ylabel)
        _legend(ax)
        plt.tight_layout()
        return fig

    with tab1: st.pyplot(make_plot("rewards", "Total Reward", "Total Reward per Episode"), use_container_width=True)
    with tab2: st.pyplot(make_plot("scores",  "Score",        "Score per Episode"), use_container_width=True)
    with tab3: st.pyplot(make_plot("steps",   "Steps",        "Steps per Episode"), use_container_width=True)
    with tab4:
        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor(MPL_BG)
        for n, res in results.items():
            ax.plot(res["epsilons"], color=res["color"], linewidth=2.2, label=n)
        style_ax(ax, "Exploration Rate (ε) Decay", "Episode", "Epsilon")
        _legend(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    # ── Convergence ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🔬 Convergence Analysis</div>", unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    fig.patch.set_facecolor(MPL_BG)
    w  = max(20, episodes // 50)
    for n, res in results.items():
        sm = smooth(res["scores"], w)
        axes[0].plot(np.linspace(0, episodes, len(sm)), sm,
                     color=res["color"], linewidth=2.4, label=n)
    style_ax(axes[0], "Smoothed Score Over Training", "Episode", "Score")
    _legend(axes[0])
    
    axes[1].set_facecolor(MPL_AX_BG)
    names_  = list(results.keys())
    avgs_   = [np.mean(results[n]["scores"][-100:]) for n in names_]
    colors_ = [results[n]["color"] for n in names_]
    bars    = axes[1].bar(names_, avgs_, color=colors_, width=0.5,
                          edgecolor=MPL_BG, linewidth=2, zorder=3)
    for bar, val in zip(bars, avgs_):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f"{val:.2f}", ha="center", va="bottom",
                     color=MPL_TEXT, fontsize=10, fontweight="bold")
    style_ax(axes[1], "Avg Score — Last 100 Episodes", ylabel="Score")
    axes[1].grid(axis="y", alpha=0.12, color=MPL_GRID)
    plt.tight_layout(pad=1.5)
    st.pyplot(fig, use_container_width=True)

    # ── Q-Value distribution ──────────────────────────────────────────────────
    st.markdown("<div class='section-title'>🧠 Q-Value Distribution</div>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor(MPL_BG)
    for n, res in results.items():
        agent = res["agent"]
        if n == "DQN":
            q_vals = agent.sample_q_values(500)
            label  = "DQN (network outputs)"
        else:
            q_vals = np.concatenate([v for v in agent.Q.values()]) if agent.Q else np.array([])
            label  = n
        if len(q_vals):
            ax.hist(q_vals, bins=60, alpha=0.55, color=res["color"],
                    label=label, density=True)
    style_ax(ax,
             "Q-Value Distribution · Q-Learning = shifted right (overestimation) · DQN/Double = centred",
             "Q-Value", "Density")
    _legend(ax)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("<div class='section-title'>📋 Comparison Table</div>", unsafe_allow_html=True)
    rows = []
    for n, res in results.items():
        rows.append({
            "Algorithm":             n,
            "Avg Score (last 100)":  f"{np.mean(res['scores'][-100:]):.2f}",
            "Best Score":            max(res["scores"]),
            "Avg Reward (last 100)": f"{np.mean(res['rewards'][-100:]):.2f}",
            "Memory / Q-States":     f"{res['q_table_size']:,}",
            "Policy Type":           "On-policy" if n == "SARSA" else "Off-policy",
            "Bias Correction":       "✅" if n in ("Double Q-Learning", "DQN") else "❌",
            "Deep Learning":         "✅" if n == "DQN" else "❌",
        })
    st.dataframe(pd.DataFrame(rows).set_index("Algorithm"), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Model  SAVE / LOAD
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-wrap' style='padding:1rem 0 0.8rem;'>
        <div class='hero-badge'>💾 Persistence</div>
        <div class='hero-title' style='font-size:2rem;'>Save &amp; Load Models</div>
        <div class='hero-sub' style='font-size:0.9rem;'>
            Download trained agents and reload them later — no need to retrain.
        </div>
    </div>""", unsafe_allow_html=True)

    save_col, load_col = st.columns([1, 1], gap="large")

    # ── Download ──────────────────────────────────────────────────────────────
    with save_col:
        st.markdown("<div class='io-card'><div class='io-card-title'>📥 Download Trained Models</div>",
                    unsafe_allow_html=True)
        for name, res in results.items():
            agent = res["agent"]
            buf   = io.BytesIO()
            agent.save(buf)
            buf.seek(0)
            ext     = ".pt"  if name == "DQN" else ".pkl"
            label   = f"dqn"  if name == "DQN" else name.lower().replace(" ", "_")
            mime    = "application/octet-stream"
            color   = res["color"]
            st.download_button(
                label=f"⬇️  {name}{ext}",
                data=buf.getvalue(),
                file_name=f"snake_rl_{label}{ext}",
                mime=mime,
                use_container_width=True,
                key=f"dl_{name}",
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Upload / Load ─────────────────────────────────────────────────────────
    with load_col:
        st.markdown("<div class='io-card'><div class='io-card-title'>📤 Load a Saved Model</div>",
                    unsafe_allow_html=True)

        algo_opts = ["Q-Learning", "SARSA", "Double Q-Learning"]
        if TORCH_AVAILABLE:
            algo_opts.append("DQN")

        load_name = st.selectbox("Algorithm slot to load into", algo_opts,
                                 key="load_algo_sel")
        upload    = st.file_uploader("Upload .pkl (tabular) or .pt (DQN)",
                                     type=["pkl", "pt"], key="model_upload")

        if upload is not None:
            load_btn = st.button("🔄  Load Model", use_container_width=True,
                                 type="primary", key="load_btn")
            if load_btn:
                try:
                    buf = io.BytesIO(upload.read())
                    if load_name == "Q-Learning":
                        a = QLearningAgent(alpha=alpha, gamma=gamma,
                                           epsilon=0.01, epsilon_min=0.01,
                                           epsilon_decay=epsilon_decay)
                        a.load(buf)
                    elif load_name == "SARSA":
                        a = SARSAAgent(alpha=alpha, gamma=gamma,
                                       epsilon=0.01, epsilon_min=0.01,
                                       epsilon_decay=epsilon_decay)
                        a.load(buf)
                    elif load_name == "Double Q-Learning":
                        a = DoubleQLearningAgent(alpha=alpha, gamma=gamma,
                                                 epsilon=0.01, epsilon_min=0.01,
                                                 epsilon_decay=epsilon_decay)
                        a.load(buf)
                    elif load_name == "DQN" and TORCH_AVAILABLE:
                        a = DQNAgent()
                        a.load(buf)
                    else:
                        st.error("DQN requires PyTorch.")
                        a = None

                    if a is not None:
                        # Merge loaded agent into current results
                        loaded_res = {
                            "rewards": [], "scores": [], "steps": [],
                            "epsilons": [], "q_table_size": a.get_q_table_size(),
                            "agent": a, "color": ALGO_COLORS.get(load_name, "#888"),
                        }
                        st.session_state["results"][load_name] = loaded_res
                        st.success(f"✅ {load_name} loaded successfully! "
                                   f"Click ▶️ Play above to watch it.")
                except Exception as exc:
                    st.error(f"Failed to load model: {exc}")

        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Theory
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='hero-wrap' style='padding:1rem 0 0.5rem;'>
        <div class='hero-badge'>📚 Theory</div>
        <div class='hero-title' style='font-size:2rem;'>Key Concepts</div>
    </div>""", unsafe_allow_html=True)

    th1, th2 = st.columns(2)
    with th1:
        st.markdown("""
        <div class='theory-card'>
        <div style='font-weight:700; color:#e2e8f0; margin-bottom:0.6rem;'>Update Rules</div>
        <code>Q-Learning (off-policy):
  Q(s,a) ← Q(s,a) + α·[r + γ·max Q(s',·) − Q(s,a)]

SARSA (on-policy):
  Q(s,a) ← Q(s,a) + α·[r + γ·Q(s', a') − Q(s,a)]
            a' = action actually taken next

Double Q-Learning:
  Q₁(s,a) ← Q₁ + α·[r + γ·Q₂(s', argmax Q₁(s',·)) − Q₁]

DQN (neural network):
  θ ← θ − α·∇MSE(Q_θ(s,a),  r + γ·max Q_θ̄(s',·))
            Q_θ̄  = frozen target network</code>
        <div style='font-weight:700; color:#e2e8f0; margin:1rem 0 0.5rem;'>State (8-dim)</div>
        <code>[danger_straight, danger_right, danger_left,   ← 3 bits
 direction ∈ {UP, DOWN, LEFT, RIGHT},          ← 1 value
 food_up, food_down, food_left, food_right]    ← 4 bits
→ ~2 048 reachable states (tabular RL viable)</code>
        </div>""", unsafe_allow_html=True)

    with th2:
        st.markdown("""
        <div class='theory-card'>
        <div style='font-weight:700; color:#6366f1; margin-bottom:0.4rem;'>Q-Learning — Off-Policy</div>
        <div style='color:#64748b; font-size:0.84rem; line-height:1.7; margin-bottom:0.9rem;'>
        Uses max over next state independent of the agent's actual next action.
        Converges fast to the <em>optimal</em> policy but the max operator causes
        <strong style='color:#e2e8f0;'>Q-value overestimation</strong>, especially with noisy rewards.
        </div>
        <div style='font-weight:700; color:#10b981; margin-bottom:0.4rem;'>SARSA — On-Policy</div>
        <div style='color:#64748b; font-size:0.84rem; line-height:1.7; margin-bottom:0.9rem;'>
        Updates using the action <em>actually</em> taken. Feels the cost of exploration —
        naturally more <strong style='color:#e2e8f0;'>conservative near walls</strong> and slower to converge,
        but more stable throughout training.
        </div>
        <div style='font-weight:700; color:#f59e0b; margin-bottom:0.4rem;'>Double Q-Learning — Bias Fix</div>
        <div style='color:#64748b; font-size:0.84rem; line-height:1.7; margin-bottom:0.9rem;'>
        Two Q-tables — one selects the action, the other evaluates it.
        Decoupling prevents a single table from inflating its own choices, giving
        <strong style='color:#e2e8f0;'>tighter Q-values</strong> late in training.
        </div>
        <div style='font-weight:700; color:#ec4899; margin-bottom:0.4rem;'>DQN — Deep RL</div>
        <div style='color:#64748b; font-size:0.84rem; line-height:1.7;'>
        A neural network replaces the Q-table. <strong style='color:#e2e8f0;'>Experience replay</strong>
        breaks temporal correlations; a <strong style='color:#e2e8f0;'>frozen target network</strong>
        stabilises the Bellman target. Gradient clipping prevents catastrophic updates.
        </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>🔑 What to Look for in the Charts</div>",
                unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    _insights = [
        ("📈 Reward Curve",
         "Should trend upward. Flat or noisy = still exploring. Q-Learning rises fastest; "
         "SARSA more smoothly; DQN slower to warm up but strongest long-term."),
        ("🧠 Q-Value Distribution",
         "Q-Learning's histogram shifts right — classic overestimation. "
         "Double Q and DQN produce tighter, more centred distributions."),
        ("👣 Steps per Episode",
         "More steps = longer survival = better policy. Early episodes are short. "
         "Tracks how quickly each agent learns to avoid walls."),
        ("🎲 Epsilon Decay",
         "All tabular agents share the same decay schedule. DQN may need fewer "
         "episodes to converge because the neural net generalises across similar states."),
    ]
    for col, (t, b) in zip([k1, k2, k3, k4], _insights):
        with col:
            st.markdown(f"""
            <div class='insight-card'>
                <div class='insight-title'>{t}</div>
                <div class='insight-body'>{b}</div>
            </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<hr class='glow-divider'>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; padding:0.8rem 0 0.3rem; color:#1e293b; font-size:0.75rem;'>
    Snake RL Lab · Q-Learning · SARSA · Double Q-Learning · DQN · Streamlit · PyTorch
</div>
""", unsafe_allow_html=True)
