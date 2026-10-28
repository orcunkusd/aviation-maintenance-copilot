# Streamlit page: deploy steps
1. Copy this folder's contents into your repo under `app/` (or the repo root).
2. Make sure `.gitignore` contains `.streamlit/secrets.toml`.
3. Push to GitHub.
4. share.streamlit.io -> New app -> pick repo, branch, main file `app/app.py` (or `app.py`).
5. Advanced settings -> Python 3.12 (or 3.13) and paste the secrets from `.streamlit/secrets.toml.example`.
6. Test: open the URL, move the mouse, press "Explain the project", ask a question.
Swap the mascot: replace `webapp/robot.glb` with any other .glb that has Neck/Head bones.
Mascot model: RobotExpressive by Tomas Laulhe (CC0, via three.js examples). three.js is MIT.
