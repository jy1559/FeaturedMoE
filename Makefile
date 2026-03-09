PYTHON ?= env PYTHONDONTWRITEBYTECODE=1 python3
CHAT_MGR := .codex/scripts/chatlog_manager.py

TITLE ?= general

GPUS ?= 0,1
SEED ?= 42
DATASETS ?= movielens1m,retail_rocket

.PHONY: session_start session_end
.PHONY: fmoe-드라이런 fmoe-실행 fmoe-수집 fmoe-다음계획

session_start:
	$(PYTHON) $(CHAT_MGR) open --title "$(TITLE)" --topic "fmoe" --mode "chat"

session_end:
	$(PYTHON) $(CHAT_MGR) close
	$(PYTHON) $(CHAT_MGR) summary-refresh

fmoe-드라이런:
	bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets "$(DATASETS)" --gpus "$(GPUS)" --seed-base "$(SEED)" --dry-run

fmoe-실행:
	bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets "$(DATASETS)" --gpus "$(GPUS)" --seed-base "$(SEED)"

fmoe-수집:
	$(PYTHON) .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets "$(DATASETS)" --metric mrr@20

fmoe-다음계획:
	$(PYTHON) .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/hyperopt_results/summary.csv --mode fmoe-first --topn 3

