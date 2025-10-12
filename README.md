# Reproducing Current Research Session

The backend services for this research run inside a tmux session to keep the OpenHands research extension alive while automation proceeds. To restart the environment the following command was used successfully:

```
tmux send-keys -t uagent-backend 'source /Users/wuy/Desktop/code/UAgent/.venv/bin/activate && cd /Users/wuy/Desktop/code/UAgent && ./start_openhands_research.sh' C-m
```

After the backend is up, open the OpenHands UI (served on `http://localhost:2999`) and start a new conversation with the detailed research goal to trigger the automated workflow.
