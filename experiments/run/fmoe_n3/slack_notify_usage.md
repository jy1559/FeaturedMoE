# Slack Webhook Notification Usage

## 1) Local secret file (recommended)
Create `experiments/run/fmoe_n3/.env.slack`:

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

`experiments/run/fmoe_n3/.env.slack` is ignored by git.

## 2) Quick connection test
```bash
bash experiments/run/fmoe_n3/slack_webhook_test.sh
```

## 3) Run any command with optional Slack notification
Default is OFF:

```bash
bash experiments/run/fmoe_n3/run_with_slack_notify.sh python your_script.py
```

Enable notification:

```bash
bash experiments/run/fmoe_n3/run_with_slack_notify.sh --on python your_script.py
bash experiments/run/fmoe_n3/run_with_slack_notify.sh --on sh experiments/run/fmoe_n3/phase_8_router_wrapper_diag.sh
```

Disable notification explicitly:

```bash
bash experiments/run/fmoe_n3/run_with_slack_notify.sh --off python your_script.py
```

Environment toggle also works:

```bash
SLACK_NOTIFY=1 bash experiments/run/fmoe_n3/run_with_slack_notify.sh python your_script.py
```

When notification is ON, Slack gets:
- `STARTED` message right before command execution (includes `started_kst`)
- `SUCCESS` or `FAILED` message when command exits (includes `started_kst`, `finished_kst`, `duration`)
- If interrupted by `Ctrl+C`/`SIGTERM`, `INTERRUPTED` end message is sent (`interrupted_by=INT|TERM`)
