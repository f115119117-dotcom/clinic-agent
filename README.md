# Sunrise Community Clinic (晨光社區診所) — Customer Service Agent (Tool Calling + RAG)

A Gradio chat agent for a small family clinic in Kaohsiung. It answers clinic questions, looks up the doctor schedule, calculates visit fees, **searches the clinic's visit rules with RAG**, and hands complaints or unknown questions to the front desk. Chinese and English are both supported. All clinic data is made up for practice.

![screenshot](screenshots/chat.png)  <!-- TODO: add a screenshot that shows the tool trace -->

## What it does

| Question | Tool |
|---|---|
| Opening hours, address, registration, cancelling, payment, parking, flu shots | `get_clinic_info` (table `shop_info`, 12 topics) |
| Which doctor / which day / is a session full | `search_doctors` (table `doctors`, 20 sessions) |
| Total fee for a visit (registration fee + co-payment per patient) | `calculate_visit_fee` (computed in code, not by the model) |
| Visit rules with conditions and exceptions (late arrival, fever or cough, medical records, referral, wheelchair access, typhoon closure…) | `search_clinic_rules` (**RAG** over `clinic_rules.txt`, 10 chunks) |
| Complaints, "speak to a manager", anything not in the data | `transfer_to_human` |

Medical advice and off-topic questions are refused by the system prompt. The right-hand panel shows every tool call, its arguments and its result; for RAG it shows the chunks that were found and their scores.

## How the RAG part works

1. `clinic_rules.txt` is cut into chunks: one `##` section = one chunk (10 chunks).
2. When the app starts, every chunk is turned into a vector with an embedding model.
3. For each rule question, the model calls `search_clinic_rules`. The question is embedded, compared with every chunk by cosine similarity, and the best 2 chunks are returned (only if the score is at least `MIN_SCORE`, 0.45).
4. The chat model writes the answer only from those chunks and names the source, like `(source: Late arrival)`.

Facts that fit in a table (hours, prices, schedule) stay in the database. Each fact lives in one place only.

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in LLM_API_URL, MODEL_NAME, API_KEY and the EMBEDDING_* settings
python app.py
```

The tables are rebuilt from `shop_info.csv` and `doctors.csv` each time the app starts, and the rule chunks are embedded at start-up, so the `.db` file is not committed.

## Test the tool choice

```bash
python test_tool_choice.py
```

21 questions, each with the tool(s) we accept. Result: TODO (write your score here after you run it).

## Files

- `app.py` — database, RAG search, tools, system prompt, agent loop, Gradio page
- `shop_info.csv`, `doctors.csv` — the table data
- `clinic_rules.txt` — the long visit rules used by RAG
- `test_tool_choice.py` — tool-choice test
- `.env.example`, `requirements.txt`, `.gitignore`

## Limits

The agent cannot book or change appointments; it tells the patient to register at the front desk or by phone. The rule search needs the embedding model; if it is not reachable, the other tools still work.