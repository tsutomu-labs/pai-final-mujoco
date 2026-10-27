import os
import json
import numpy as np
import mujoco
import imageio
from PIL import Image, ImageDraw, ImageFont
from ollama import chat

# ===== 設定 =====
MODEL_NAME = "qwen2.5:3b"          # Ollamaで使うモデル
COMMANDS = [                        # ロボットへの日本語の指示(ここを書き換えると動きが変わる)
    "赤い箱を緑のトレイに置いて",
    "青い箱をオレンジのトレイへ運んで",
    "黄色い箱も緑のトレイに入れて",
    "青い箱を緑のトレイに移して",
    "赤い箱をオレンジのトレイに移してください",
]
PANDA_DIR = os.path.expanduser("~/mujoco_menagerie/franka_emika_panda")
OUT_PATH = os.path.expanduser("~/pick_place_llm.mp4")

BOXES = {   # 名前: (位置x, 位置y, 色)
    "赤": (0.40, -0.25, "0.9 0.1 0.1 1"),
    "青": (0.50, -0.25, "0.1 0.3 0.9 1"),
    "黄": (0.60, -0.25, "0.95 0.85 0.1 1"),
}
TRAYS = {   # 名前: (位置x, 位置y, 色)
    "緑":       (0.40, 0.28, "0.2 0.8 0.3 1"),
    "オレンジ": (0.62, 0.28, "1.0 0.35 0.0 1"),
}
LABEL = {"赤": "赤い箱", "青": "青い箱", "黄": "黄色い箱"}   # 字幕用の呼び名
SLOTS = [(-0.05, -0.04), (0.05, -0.04), (0.0, 0.04)]   # トレイ内の置き場所(3か所)

# ===== シーン作成 =====
body_xml, key_qpos = "", "0 0 0 -1.57079 0 1.57079 -0.7853 0.04 0.04"
for name, (x, y, rgba) in BOXES.items():
    body_xml += f"""
    <body name="box_{name}" pos="{x} {y} 0.02"><freejoint/>
      <geom type="box" size="0.02 0.02 0.02" rgba="{rgba}" friction="1.5 0.05 0.01" mass="0.05"/>
    </body>"""
    key_qpos += f" {x} {y} 0.02 1 0 0 0"
for name, (x, y, rgba) in TRAYS.items():
    body_xml += f"""
    <geom name="tray_{name}" type="box" pos="{x} {y} 0.005" size="0.09 0.09 0.005" rgba="{rgba}"/>"""

SCENE_XML = f"""
<mujoco model="pick and place llm">
  <include file="scene.xml"/>
  <worldbody>{body_xml}
  </worldbody>
  <keyframe>
    <key name="start" qpos="{key_qpos}" ctrl="0 0 0 -1.57079 0 1.57079 -0.7853 255"/>
  </keyframe>
</mujoco>"""

xml_path = os.path.join(PANDA_DIR, "pick_place_llm_scene.xml")
with open(xml_path, "w") as f:
    f.write(SCENE_XML)
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, model.key("start").id)
mujoco.mj_forward(model, data)

# ===== AI: 日本語の指示を「どの箱を・どのトレイへ」に変換する =====
SCHEMA = {
    "type": "object",
    "properties": {
        "box":  {"type": "string", "enum": list(BOXES.keys())},
        "tray": {"type": "string", "enum": list(TRAYS.keys())},
    },
    "required": ["box", "tray"],
}
SYSTEM_PROMPT = (
    "あなたはロボットアームの指示を解釈するAIです。"
    "ユーザーの日本語の指示から、動かす箱の色(赤・青・黄)と、置き先のトレイの色(緑・オレンジ)を"
    "JSONで答えてください。例: {\"box\": \"赤\", \"tray\": \"緑\"}"
)

def parse_instruction(text):
    res = chat(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": text}],
        format=SCHEMA,
        options={"temperature": 0},
    )
    out = json.loads(res["message"]["content"])
    return out["box"], out["tray"]

# ===== ロボットの制御(逆運動学) =====
hand_id = model.body("hand").id
TCP_OFFSET = np.array([0, 0, 0.10])
R_target = data.xmat[hand_id].reshape(3, 3).copy()
kin = mujoco.MjData(model)

def tcp_of(d):
    return d.xpos[hand_id] + d.xmat[hand_id].reshape(3, 3) @ TCP_OFFSET

def ik_step(q, target_pos, damping=1e-4):
    kin.qpos[:7] = q
    mujoco.mj_forward(model, kin)
    p = tcp_of(kin)
    jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
    mujoco.mj_jac(model, kin, jacp, jacr, p, hand_id)
    J = np.vstack([jacp[:, :7], jacr[:, :7]])
    R = kin.xmat[hand_id].reshape(3, 3)
    rot_err = 0.5 * sum(np.cross(R[:, i], R_target[:, i]) for i in range(3))
    err = np.concatenate([target_pos - p, rot_err])
    dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
    return q + dq

# ===== 動画の字幕 =====
FONT = None
for p in ["/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
          "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
          "/System/Library/Fonts/Hiragino Sans GB.ttc",
          "/Library/Fonts/Arial Unicode.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    if os.path.exists(p):
        FONT = ImageFont.truetype(p, 22)
        break
if FONT is None:
    print("日本語フォントが見つからないため、字幕なしで作ります")

caption = []
def add_caption(frame):
    if FONT is None or not caption:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, frame.shape[1], 12 + 32 * len(caption)], fill=(0, 0, 0, 150))
    for i, line in enumerate(caption):
        d.text((12, 8 + 32 * i), line, font=FONT, fill=(255, 255, 255, 255))
    return np.array(img)

# ===== 実行 =====
renderer = mujoco.Renderer(model, height=480, width=640)
cam = mujoco.MjvCamera()
cam.lookat[:] = [0.5, 0.0, 0.1]; cam.distance = 1.5
cam.azimuth = 150; cam.elevation = -25

FPS = 30
frames = []
q_des = data.qpos[:7].copy()
dt = model.opt.timestep
slots = {t: [None] * len(SLOTS) for t in TRAYS}   # トレイごとの置き場所の使用状況

def run(plan):
    global q_des
    for target, grip, dur in plan:
        kin.qpos[:7] = q_des; mujoco.mj_forward(model, kin)
        start = tcp_of(kin).copy()
        steps = int(dur / dt)
        for i in range(steps):
            s = (i + 1) / steps
            s = s * s * (3 - 2 * s)
            q_des = ik_step(q_des, start + (target - start) * s)
            data.ctrl[:7] = q_des
            data.ctrl[7] = grip
            mujoco.mj_step(model, data)
            if len(frames) < data.time * FPS:
                renderer.update_scene(data, camera=cam)
                frames.append(add_caption(renderer.render()))

HOVER = 0.25
home = tcp_of(data)
caption[:] = ["ロボットへの指示を待っています"]
run([(home, 255, 1.5)])

for text in COMMANDS:
    caption[:] = [f"指示: {text}", "AIが解釈中..."]
    box, tray = parse_instruction(text)
    print(f"指示「{text}」 -> 箱={box}, トレイ={tray}")
    caption[:] = [f"指示: {text}", f"AIの解釈: {LABEL[box]} → {tray}のトレイ"]

    # 箱が今あるトレイの場所を空ける / 置き先の空き場所を決める
    for t in slots:
        slots[t] = [None if b == box else b for b in slots[t]]
    free = slots[tray].index(None)
    slots[tray][free] = box

    bp = data.body(f"box_{box}").xpos.copy()                 # 箱の今の位置
    tx, ty, _ = TRAYS[tray]
    dx, dy = SLOTS[free]
    place = np.array([tx + dx, ty + dy, 0.0])
    run([
        (home, 255, 1.0),
        (np.array([bp[0], bp[1], HOVER]), 255, 2.0),
        (np.array([bp[0], bp[1], bp[2] + 0.005]), 255, 1.5),
        (np.array([bp[0], bp[1], bp[2] + 0.005]), 0, 1.0),      # つかむ
        (np.array([bp[0], bp[1], HOVER]), 0, 1.5),               # 持ち上げる
        (np.array([place[0], place[1], HOVER]), 0, 2.0),
        (np.array([place[0], place[1], 0.065]), 0, 1.2),
        (np.array([place[0], place[1], 0.065]), 255, 1.0),      # はなす
        (np.array([place[0], place[1], HOVER]), 255, 1.5),
    ])
    home = np.array([place[0], place[1], HOVER])

caption[:] = ["すべての指示が完了しました"]
run([(home, 255, 2.0)])

imageio.mimwrite(OUT_PATH, frames, fps=FPS)
print("動画の長さ: 約%d秒" % (len(frames) / FPS), "->", OUT_PATH)
for n in BOXES:
    print(n, np.round(data.body(f"box_{n}").xpos, 3))
