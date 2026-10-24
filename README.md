# Sunrise Community Clinic (晨光社區診所) — Customer Service Agent

A Gradio chat agent for a small family clinic in Kaohsiung. It answers clinic questions, looks up the doctor schedule, calculates visit fees, and hands complaints or unknown questions to the front desk. Chinese and English are both supported.


## What it does

| Question | Tool |
|---|---|
| Opening hours, address, registration, cancelling, payment, parking, flu shots | `get_clinic_info` (table `shop_info`, 12 topics) |
| Which doctor / which day / is a session full | `search_doctors` (table `doctors`, 20 sessions) |
| Total fee for a visit (registration fee + co-payment per patient) | `calculate_visit_fee` (computed in code, not by the model) |
| Complaints, "speak to a manager", anything not in the data | `transfer_to_human` |

Medical advice and off-topic questions are refused by the system prompt. The right-hand panel shows every tool call, its arguments and its result.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in LLM_API_URL, MODEL_NAME and API_KEY
python app.py
```

The tables are rebuilt from `shop_info.csv` and `doctors.csv` each time the app starts, so the `.db` file is not committed.

## Test the tool choice

```bash
python test_tool_choice.py
```

15 questions, each with the tool(s) we accept. Result: 14/15 at first, 15/15 after fixing the tool descriptions and rule 6 of the system prompt.

## Files

- `app.py` — database, tools, system prompt, agent loop, Gradio page
- `shop_info.csv`, `doctors.csv` — the data
- `test_tool_choice.py` — tool-choice test
- `.env.example`, `requirements.txt`, `.gitignore`

## Limits

The agent cannot book or change appointments; it tells the patient to register at the front desk or by phone.
