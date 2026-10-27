# pai-final-mujoco
# 日本語の指示で動くピック&プレース(MuJoCo + ローカルLLM)

Physical AI応用1講座の最終課題です。日本語の指示をローカルLLM(Ollama)で解釈し、
ロボットアーム(Franka Panda)が3色の箱を2つのトレイへ運びます。

## 仕組み
1. 日本語の指示(例: 「青い箱をオレンジのトレイへ運んで」)をLLMに渡す
2. LLMが「どの箱を・どのトレイへ」をJSONで返す(選択肢を限定した構造化出力)
3. 逆運動学(ヤコビアン法)でアームを動かし、箱をつかんで置く
4. 指示とAIの解釈を字幕にして、MP4に書き出す

## 動かし方(Mac)
    pip install mujoco imageio imageio-ffmpeg numpy pillow ollama
    git clone https://github.com/google-deepmind/mujoco_menagerie.git ~/mujoco_menagerie
    ollama pull qwen2.5:3b
    python pick_place_llm.py

Ollamaのアプリを起動した状態で実行してください。動画は `~/pick_place_llm.mp4` に出力されます。
