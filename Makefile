PYTHON ?= env PYTHONDONTWRITEBYTECODE=1 python3
CHAT_MGR := .codex/scripts/chatlog_manager.py

TITLE ?=
TOPIC ?= other
MODE ?= chat
CHAT ?=
DATE ?=
WEEK ?=

USER_INTENT ?= 요청 미기재
REQUEST_TYPE ?= 분석요청
ISSUE_TYPE ?= 없음
OUTPUT_TYPE ?= 요약
PRIORITY ?= 중간
USER_REQUIREMENT ?= 요구 미기재
ASSISTANT_ACTION ?= 수행 미기재
STATUS ?= ok

AUTO_USER_MSG ?= 자동 분류용 사용자 메시지 미기재
AUTO_ACTION ?= 자동 분류용 수행 내용 미기재
AUTO_PRIORITY ?= 중간
AUTO_STATUS ?= ok
AUTO_USER_REQUIREMENT ?=
AUTO_REQUEST_TYPE ?=
AUTO_ISSUE_TYPE ?=
AUTO_OUTPUT_TYPE ?=

GOAL ?= -
WHAT_RAN ?= -
RESULTS ?= -
DECISIONS ?= -
NEXT ?= -

GPUS ?= 0,1
SEED ?= 42
DATASETS ?= movielens1m,retail_rocket

.PHONY: 세션-시작 세션-종료 세션-목록 세션-요약 턴-추가 턴-자동 종합-갱신 종합-보기 fmoe-세션-보장 fmoe-턴-자동
.PHONY: chat-open chat-close chat-list chat-summary chat-turn chat-auto-turn summary-refresh summary-view chat-ensure
.PHONY: fmoe-드라이런 fmoe-실행 fmoe-수집 fmoe-다음계획

세션-시작:
	$(PYTHON) $(CHAT_MGR) open --title "$(TITLE)" --topic "$(TOPIC)" --mode "$(MODE)"

chat-open: 세션-시작

fmoe-세션-보장:
	$(PYTHON) $(CHAT_MGR) ensure-open --title "fmoe-auto-session" --topic "fmoe" --mode "chat"

chat-ensure: fmoe-세션-보장

턴-추가:
	$(PYTHON) $(CHAT_MGR) turn --chat "$(CHAT)" \
		--user-intent "$(USER_INTENT)" \
		--request-type "$(REQUEST_TYPE)" \
		--issue-type "$(ISSUE_TYPE)" \
		--output-type "$(OUTPUT_TYPE)" \
		--priority "$(PRIORITY)" \
		--user-requirement "$(USER_REQUIREMENT)" \
		--assistant-action "$(ASSISTANT_ACTION)" \
		--status "$(STATUS)"

chat-turn: 턴-추가

턴-자동:
	$(PYTHON) $(CHAT_MGR) auto-turn --chat "$(CHAT)" \
		--user-message "$(AUTO_USER_MSG)" \
		--assistant-action "$(AUTO_ACTION)" \
		--priority "$(AUTO_PRIORITY)" \
		--status "$(AUTO_STATUS)" \
		--user-requirement "$(AUTO_USER_REQUIREMENT)" \
		--request-type "$(AUTO_REQUEST_TYPE)" \
		--issue-type "$(AUTO_ISSUE_TYPE)" \
		--output-type "$(AUTO_OUTPUT_TYPE)"

chat-auto-turn: 턴-자동

fmoe-턴-자동: fmoe-세션-보장
	$(PYTHON) $(CHAT_MGR) auto-turn --chat "$(CHAT)" \
		--user-message "$(AUTO_USER_MSG)" \
		--assistant-action "$(AUTO_ACTION)" \
		--priority "$(AUTO_PRIORITY)" \
		--status "$(AUTO_STATUS)" \
		--user-requirement "$(AUTO_USER_REQUIREMENT)" \
		--request-type "$(AUTO_REQUEST_TYPE)" \
		--issue-type "$(AUTO_ISSUE_TYPE)" \
		--output-type "$(AUTO_OUTPUT_TYPE)"

세션-종료:
	$(PYTHON) $(CHAT_MGR) close --chat "$(CHAT)" \
		--goal "$(GOAL)" \
		--what-ran "$(WHAT_RAN)" \
		--results "$(RESULTS)" \
		--decisions "$(DECISIONS)" \
		--next "$(NEXT)"

chat-close: 세션-종료

세션-목록:
	$(PYTHON) $(CHAT_MGR) list

chat-list: 세션-목록

세션-요약:
	$(PYTHON) $(CHAT_MGR) chat-summary --chat "$(CHAT)"

chat-summary: 세션-요약

종합-갱신:
	$(PYTHON) $(CHAT_MGR) summary-refresh

summary-refresh: 종합-갱신

종합-보기:
	@if [ -n "$(DATE)" ]; then \
		$(PYTHON) $(CHAT_MGR) summary-view --date "$(DATE)"; \
	elif [ -n "$(WEEK)" ]; then \
		$(PYTHON) $(CHAT_MGR) summary-view --week "$(WEEK)"; \
	elif [ -n "$(TOPIC)" ]; then \
		$(PYTHON) $(CHAT_MGR) summary-view --topic "$(TOPIC)"; \
	else \
		echo "DATE=YYMMDD 또는 WEEK=YYWW 또는 TOPIC=<topic> 중 하나를 지정하세요."; \
		exit 1; \
	fi

summary-view: 종합-보기

fmoe-드라이런: fmoe-세션-보장
	bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets "$(DATASETS)" --gpus "$(GPUS)" --seed-base "$(SEED)" --dry-run
	$(PYTHON) $(CHAT_MGR) auto-turn --user-message "FMoE 드라이런 실행 요청" --assistant-action "make fmoe-드라이런 실행 완료 (datasets=$(DATASETS), gpus=$(GPUS), seed=$(SEED))" --request-type "실행요청" --issue-type "없음" --output-type "실험결과" --priority "중간" --status "ok"

fmoe-실행: fmoe-세션-보장
	bash .codex/skills/fmoe/scripts/launch_track.sh --track fmoe-main --datasets "$(DATASETS)" --gpus "$(GPUS)" --seed-base "$(SEED)"
	$(PYTHON) $(CHAT_MGR) auto-turn --user-message "FMoE 메인 실행 요청" --assistant-action "make fmoe-실행 완료 (datasets=$(DATASETS), gpus=$(GPUS), seed=$(SEED))" --request-type "실행요청" --issue-type "없음" --output-type "실험결과" --priority "높음" --status "ok"

fmoe-수집: fmoe-세션-보장
	$(PYTHON) .codex/skills/fmoe/scripts/collect_results.py --repo-root /workspace/jy1559/FMoE --datasets "$(DATASETS)" --metric mrr@20
	$(PYTHON) $(CHAT_MGR) auto-turn --user-message "FMoE 결과 수집 요청" --assistant-action "make fmoe-수집 완료 (datasets=$(DATASETS), metric=mrr@20)" --request-type "실행요청" --issue-type "없음" --output-type "실험결과" --priority "중간" --status "ok"

fmoe-다음계획: fmoe-세션-보장
	$(PYTHON) .codex/skills/fmoe/scripts/recommend_next.py --summary /workspace/jy1559/FMoE/experiments/run/hyperopt_results/summary.csv --mode fmoe-first --topn 3
	$(PYTHON) $(CHAT_MGR) auto-turn --user-message "FMoE 다음 계획 생성 요청" --assistant-action "make fmoe-다음계획 완료 (mode=fmoe-first, topn=3)" --request-type "의사결정요청" --issue-type "없음" --output-type "계획" --priority "중간" --status "ok"
