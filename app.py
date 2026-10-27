"""A customer service agent for Sunrise Community Clinic (Gradio app).

Every question goes to the agent first. The agent decides which tool to use:
clinic information and the doctor schedule come from a SQLite database, the code works out
the visit fee, the clinic's visit rules are searched by meaning (RAG), and anything the tools
cannot answer is handed to a human.
The panel on the right shows which tools were called and what they returned.

Run it with:  python app.py
"""

import csv
import json
import os
import re
from pathlib import Path
import sqlite3

import gradio as gr
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# 1. API setup
# ---------------------------------------------------------------------------
# Settings come from the .env file in this folder (copy .env.example to .env first).
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")
API_KEY = os.getenv("API_KEY")
API_ENDPOINT = os.getenv("LLM_API_URL")
CHAT_MODEL = os.getenv("MODEL_NAME")

client = OpenAI(api_key=API_KEY, base_url=API_ENDPOINT, timeout=60.0)

# RAG needs a second model, the embedding model. If its settings are empty,
# we fall back to the chat server's URL and key (both models on the same server).
EMBED_API_URL = os.getenv("EMBEDDING_API_URL") or API_ENDPOINT
EMBED_API_KEY = os.getenv("EMBEDDING_API_KEY") or API_KEY
EMBED_MODEL = os.getenv("EMBEDDING_MODEL_NAME")
embed_client = OpenAI(api_key=EMBED_API_KEY, base_url=EMBED_API_URL, timeout=60.0)


# ---------------------------------------------------------------------------
# 2. The clinic database: two tables, shop_info and doctors (SQLite)
# ---------------------------------------------------------------------------
# The program rebuilds its tables from the CSV files every time it starts,
# so the .db file is never submitted (see .gitignore).
DB_FILE = BASE_DIR / "clinic_app.db"

# check_same_thread=False because Gradio answers requests on worker threads.
# timeout=10: wait up to 10 seconds if another program is using the file.
connection = sqlite3.connect(DB_FILE, check_same_thread=False, timeout=10)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute("DROP TABLE IF EXISTS shop_info")
cursor.execute("""
    CREATE TABLE shop_info (
        topic   TEXT PRIMARY KEY,
        content TEXT NOT NULL
    )
""")
cursor.execute("DROP TABLE IF EXISTS doctors")
cursor.execute("""
    CREATE TABLE doctors (
        slot_id          INTEGER PRIMARY KEY,
        name             TEXT NOT NULL,
        name_zh          TEXT NOT NULL,   -- Chinese name, for questions in Chinese
        department       TEXT NOT NULL,
        department_zh    TEXT NOT NULL,
        weekday          TEXT NOT NULL,   -- Mon, Tue, Wed, Thu, Fri, Sat
        session          TEXT NOT NULL,   -- morning, afternoon, evening
        registration_fee INTEGER NOT NULL,  -- New Taiwan dollars
        available        INTEGER NOT NULL   -- 1 = open for registration, 0 = full
    )
""")
with open(BASE_DIR / "shop_info.csv", newline="", encoding="utf-8") as f:
    cursor.executemany(
        "INSERT INTO shop_info VALUES (?, ?)",
        [(r["topic"], r["content"]) for r in csv.DictReader(f)],
    )
with open(BASE_DIR / "doctors.csv", newline="", encoding="utf-8") as f:
    cursor.executemany(
        "INSERT INTO doctors VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [(int(r["slot_id"]), r["name"], r["name_zh"], r["department"], r["department_zh"],
          r["weekday"], r["session"], int(r["registration_fee"]), int(r["available"]))
         for r in csv.DictReader(f)],
    )
connection.commit()

CLINIC_TOPICS = [r["topic"] for r in cursor.execute("SELECT topic FROM shop_info ORDER BY rowid")]
DEPARTMENTS = [r["department"] for r in cursor.execute("SELECT DISTINCT department FROM doctors ORDER BY slot_id")]
SLOT_COUNT = cursor.execute("SELECT COUNT(*) AS n FROM doctors").fetchone()["n"]
print(f"Ready: {len(CLINIC_TOPICS)} clinic topics, {SLOT_COUNT} doctor sessions.")


# ---------------------------------------------------------------------------
# 2b. RAG: the clinic's visit rules (clinic_rules.txt), searched by meaning
# ---------------------------------------------------------------------------
# A rule has conditions and exceptions, so it does not fit in table columns.
# Step 1: cut the text into chunks (one "##" section = one chunk).
# Step 2: turn every chunk into a vector (an embedding), once, when the program starts.
# Step 3: for each question, find the chunks whose vectors are closest to the question's vector.
def embed(texts):
    """Turn a list of texts into vectors, one row per text."""
    response = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([item.embedding for item in response.data], dtype="float32")


def load_chunks(path):
    """Cut a text file into chunks: one '## heading' section = one chunk."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    chunks = []
    for part in text.split("\n## ")[1:]:          # the text before the first '## ' heading is skipped
        title, _, body = part.partition("\n")
        chunks.append({"title": title.strip(), "text": body.strip()})
    return chunks


RULE_CHUNKS = load_chunks(BASE_DIR / "clinic_rules.txt")
MIN_SCORE = 0.45      # below this score the rules probably do not cover the question; tune it with your own test questions
try:
    CHUNK_VECTORS = embed([f"{c['title']}\n{c['text']}" for c in RULE_CHUNKS])     # phase A: once
    print(f"Ready: {len(RULE_CHUNKS)} rule chunks embedded ({CHUNK_VECTORS.shape[1]} numbers each).")
except Exception as error:                  # the app still works without the rule search
    CHUNK_VECTORS = None
    print(f"WARNING: could not embed the rule chunks, search_clinic_rules is off: {error}")


def search_clinic_rules(question, k=2):
    """Find the chunks of the clinic's visit rules closest in meaning to the question (phase B)."""
    if CHUNK_VECTORS is None:
        return {"found": False, "note": "The rule search is not available now."}
    q = embed([question])[0]
    scores = CHUNK_VECTORS @ q / (np.linalg.norm(CHUNK_VECTORS, axis=1) * np.linalg.norm(q))
    best = [int(i) for i in np.argsort(scores)[::-1][:k] if scores[i] >= MIN_SCORE]
    if not best:
        return {"found": False, "note": "The clinic's visit rules do not cover this question."}
    return {"found": True,
            "chunks": [{"title": RULE_CHUNKS[i]["title"], "text": RULE_CHUNKS[i]["text"],
                        "score": round(float(scores[i]), 3)} for i in best]}


# ---------------------------------------------------------------------------
# 3. The tools, as plain functions
# ---------------------------------------------------------------------------
def get_clinic_info(topic):
    """Return one piece of clinic information, such as the opening hours or the cancel policy."""
    row = cursor.execute(
        "SELECT topic, content FROM shop_info WHERE topic = ? COLLATE NOCASE",
        (topic.strip(),),
    ).fetchone()
    if row is None:
        return {"found": False, "note": f"There is no clinic information about '{topic}'.",
                "topics_available": CLINIC_TOPICS}
    return {"found": True, "topic": row["topic"], "content": row["content"]}


SESSION_TIMES = {"morning": "09:00-12:00", "afternoon": "14:00-17:30", "evening": "18:00-21:00"}

# What patients say -> words that appear in the department names in the database.
DEPARTMENT_WORDS = {
    "pediatrician": "pediatrics", "child": "pediatrics", "children": "pediatrics", "kid": "pediatrics",
    "kids": "pediatrics", "baby": "pediatrics",
    "skin": "dermatology", "dermatologist": "dermatology",
    "family doctor": "family medicine", "gp": "family medicine",
    "internist": "internal medicine",
    "ear nose throat": "ent",
}


def matching_slots(keyword):
    """Doctor sessions that match a doctor name or department, in English or Chinese.

    No keyword = every session. The matching is done here, in Python: the model never writes SQL.
    """
    key = (keyword or "").strip().lower()
    key = re.sub(r"^dr(\.\s*|\s+)", "", key)               # "Dr. Lin" -> "lin"
    key = key.replace("醫師", "").replace("醫生", "").strip()  # "林醫師" -> "林"
    key = DEPARTMENT_WORDS.get(key, key)
    rows = cursor.execute("SELECT * FROM doctors ORDER BY slot_id").fetchall()
    if not key:
        return rows
    return [r for r in rows
            if key in " ".join([r["name"], r["name_zh"], r["department"], r["department_zh"]]).lower()]


def find_doctors(text):
    """The distinct doctors (one row each) that match the text."""
    doctors = {}
    for row in matching_slots(text):
        doctors.setdefault(row["name"], row)
    return list(doctors.values())


def search_doctors(keyword="", weekday="", session=""):
    """Find doctor sessions by doctor or department, optionally for one weekday or one session."""
    rows = matching_slots(keyword)
    if weekday:
        rows = [r for r in rows if r["weekday"].lower() == weekday.strip().lower()[:3]]
    if session:
        rows = [r for r in rows if r["session"] == session.strip().lower()]
    if not rows:
        return {"found": 0, "slots": [], "note": "No doctor session matches this search.",
                "departments_available": DEPARTMENTS}
    return {"found": len(rows),
            "slots": [{"doctor": r["name"], "name_zh": r["name_zh"], "department": r["department"],
                       "department_zh": r["department_zh"], "weekday": r["weekday"],
                       "session": r["session"], "time": SESSION_TIMES[r["session"]],
                       "registration_fee": r["registration_fee"], "available": bool(r["available"])}
                      for r in rows],
            "note": "available=false means the session is full: no more registrations."}


# The co-payment per patient type. These numbers are the same as the 'fees' topic in shop_info.csv.
COPAY = {"adult": 150, "child": 50, "senior": 50, "catastrophic": 0}


def patient_type_of(entry):
    """The patient type of one person. The program works it out from the age when the age is given."""
    kind = str(entry.get("patient_type") or "").strip().lower()
    if kind == "catastrophic":                 # only the customer can say this: it is not about age
        return kind
    if entry.get("age") is not None:
        age = int(entry["age"])
        if age < 12:
            return "child"
        if age >= 65:
            return "senior"
        return "adult"
    return kind if kind in COPAY else None


def calculate_visit_fee(doctor, patients):
    """Work out the fee of a visit: registration fee of the doctor + co-payment of each patient.

    The fees come from the database and the maths happens here.
    No total unless the doctor and every patient are clear: a partial total is a wrong bill.
    """
    matches = find_doctors(doctor)
    if not matches:
        return {"total": None, "note": f"No doctor or department matches '{doctor}'.",
                "departments_available": DEPARTMENTS}
    if len(matches) > 1:
        return {"total": None, "note": f"'{doctor}' matches several doctors. Ask the customer which one.",
                "candidates": [f"{r['name']} ({r['name_zh']}, {r['department']})" for r in matches]}
    chosen = matches[0]

    lines, problems, total = [], [], 0
    for number, entry in enumerate(patients or [], start=1):
        kind = patient_type_of(entry)
        if kind is None:
            problems.append({"patient": number, "note": "The age or patient type is missing."})
            continue
        subtotal = chosen["registration_fee"] + COPAY[kind]
        lines.append({"patient": number, "patient_type": kind,
                      "registration_fee": chosen["registration_fee"],
                      "copayment": COPAY[kind], "subtotal": subtotal})
        total += subtotal
    if not patients:
        problems.append({"patient": 0, "note": "No patient was given."})
    if problems:
        return {"total": None, "problems": problems,
                "note": "No total yet. Ask the customer for the age or patient type, then call calculate_visit_fee again."}
    return {"doctor": chosen["name"], "name_zh": chosen["name_zh"], "department": chosen["department"],
            "lines": lines, "total": total,
            "note": "Medicine and test fees are not included. They are decided by the doctor."}


def transfer_to_human():
    """Hand the conversation to a person."""
    return {"message": "Please call our front desk at 07-555-0123. The phone line is open Monday to Saturday from 08:30 to 20:30.",
            "note": ("The hand-over is done. First tell the patient in one short sentence why: "
                     "you do not have that information, or the request needs a person. "
                     "Then give this message in the patient's language. Do not say that you will connect them "
                     "or pass them on: you cannot. Do not call this tool again.")}


TOOL_FUNCTIONS = {
    "get_clinic_info": get_clinic_info,
    "search_doctors": search_doctors,
    "calculate_visit_fee": calculate_visit_fee,
    "search_clinic_rules": search_clinic_rules,
    "transfer_to_human": transfer_to_human,
}

# What the model sees: names, purpose, arguments. Most "it did not use my tool"
# problems are really "my description did not say when to use it".
TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "get_clinic_info",
        "description": ("Get one piece of clinic information: opening hours (門診時間), address (地址), phone (電話), "
                        "how to register (掛號), first visit (初診), cancel or change an appointment (取消改期), "
                        "payment and NHI card (付款、健保卡), the rules of fees and co-payment (費用、部分負擔), "
                        "parking (停車), flu shots and vaccines (疫苗), emergency (急診), "
                        "or why the assistant gives no medical advice (醫療建議). "
                        "Use it only for these topics. If the customer asks about a service, test or fact "
                        "that is not in this list (for example a COVID test or an X-ray), "
                        "do not use this tool: use transfer_to_human."),
        "parameters": {"type": "object",
                       "properties": {"topic": {"type": "string", "enum": CLINIC_TOPICS}},
                       "required": ["topic"]}}},
    {"type": "function", "function": {
        "name": "search_doctors",
        "description": ("Find doctor sessions. Use it when the customer asks which doctors or departments "
                        "we have, who sees patients on a day, when a doctor or department is open, "
                        "or whether a session is full. Departments: family medicine (家醫科), pediatrics (小兒科), "
                        "internal medicine (內科), dermatology (皮膚科), ENT (耳鼻喉科). "
                        "The clinic is closed on Sundays."),
        "parameters": {"type": "object",
                       "properties": {
                           "keyword": {"type": "string",
                                       "description": "A doctor name or a department, in the customer's own words, for example 'Dr. Lin' or '小兒科'. Leave empty for all doctors."},
                           "weekday": {"type": "string", "enum": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]},
                           "session": {"type": "string", "enum": ["morning", "afternoon", "evening"]}},
                       "required": []}}},
    {"type": "function", "function": {
        "name": "calculate_visit_fee",
        "description": ("Work out the fee of a visit to ONE doctor: the doctor's registration fee plus the co-payment "
                        "of each patient. Always use this for any fee total. Never add fees yourself. "
                        "If the customer has not said the age or the patient type, ask first. Do not guess."),
        "parameters": {"type": "object",
                       "properties": {
                           "doctor": {"type": "string",
                                      "description": "The doctor or the department, in the customer's own words. Do not translate it."},
                           "patients": {"type": "array",
                                        "description": "One entry for each person who will see the doctor.",
                                        "items": {"type": "object",
                                                  "properties": {
                                                      "age": {"type": "integer",
                                                              "description": "The age, if the customer said it."},
                                                      "patient_type": {"type": "string",
                                                                       "enum": ["adult", "child", "senior", "catastrophic"],
                                                                       "description": "Use 'catastrophic' only if the customer holds a catastrophic illness card (重大傷病卡). Otherwise give the age."}},
                                                  "required": []}}},
                       "required": ["doctor", "patients"]}}},
    {"type": "function", "function": {
        "name": "search_clinic_rules",
        "description": ("Search the clinic's visit rules by meaning. Use it for rules with conditions or exceptions: "
                        "arriving late (遲到), fever or cough symptoms at the clinic (發燒咳嗽的就診流程), "
                        "pediatric visits and the child health booklet (小兒科、兒童健康手冊), wheelchair and mobility access (輪椅、無障礙), "
                        "copies of medical records and diagnosis certificates (病歷影本、診斷證明書), referral letters (轉診單), "
                        "continuous prescriptions for chronic conditions (慢性病連續處方箋), lost and found (失物), "
                        "privacy and photos (隱私、拍照), typhoon or bad weather closures (颱風停診). "
                        "Do not use it for opening hours, fees, doctor schedules, or cancelling: those have their own tools. "
                        "If it finds nothing, use transfer_to_human."),
        "parameters": {"type": "object",
                       "properties": {"question": {"type": "string",
                                                   "description": "The customer's question, in the customer's own words."}},
                       "required": ["question"]}}},
    {"type": "function", "function": {
        "name": "transfer_to_human",
                "description": ("Use this when the customer asks to speak to a person or a manager, makes a complaint, "
                                "or asks about a service, test or fact of the clinic that no other tool can answer, "
                                "for example a COVID test or an X-ray."),
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]

SYSTEM_PROMPT = """
You are the online customer service agent of Sunrise Community Clinic (晨光社區診所), a small family clinic in Kaohsiung.
All fees are in New Taiwan dollars. Write them like NT$150.

RULES
1. Answer only from tool results. Never answer from memory, and never invent anything: no doctors, times, fees or rules.
2. Clinic questions (opening hours, address, phone, how to register, first visit, cancelling, payment, fee rules, parking, flu shots) -> get_clinic_info.
3. Doctor and schedule questions (which doctor, which day, which department, is a session full) -> search_doctors.
4. Visit rules with conditions or exceptions (arriving late, fever or cough at the clinic, pediatric visit, wheelchair access, medical records, referral letter, continuous prescription, lost items, privacy, typhoon closure) -> search_clinic_rules.
   Use only the chunks it returns, and end the answer with the chunk title, like this: (source: Late arrival). If it finds nothing, use transfer_to_human.
5. Use calculate_visit_fee for every fee total. Never add fees yourself. If you do not know the age or the patient type, ask before you call it.
6. If a tool reports that nothing was found, say so plainly. Do not guess.
7. Complaints, a request to speak to a person, or clinic questions no tool can answer (for example a service or a test that is not in the data) -> transfer_to_human.
8. You cannot make or change appointments. Tell the patient to register at the front desk or by phone.
9. Never give a diagnosis, medicine advice or any other medical advice. Use get_clinic_info with the topic medical_advice, and suggest seeing a doctor.
   If the patient describes an emergency, use get_clinic_info with the topic emergency and tell them to call 119.
10. Only talk about this clinic. Politely refuse other topics, such as the weather or homework.
11. Answer in the customer's language: English, or Traditional Chinese if the customer writes in Chinese.
   Write doctor names in the language of the answer: English names in English, Chinese names (name_zh) in Chinese.
   Be polite, and use three sentences or less.
"""

MAX_STEPS = 6          # a confused model must not loop forever


def language_note(text):
    """The program, not the model, decides the reply language: Chinese characters -> Chinese, else English.

    Models do not follow a language rule reliably: one answers in English after reading English
    tool results, another answers in Chinese because it was trained mostly on Chinese.
    So we do not leave this choice to the model.
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return [{"role": "system", "content": "The customer wrote in Chinese. Reply in Traditional Chinese."}]
    return [{"role": "system", "content": "The customer wrote in English. Reply in English."}]


def run_agent(chat_history):
    """Answer the last question in chat_history. Returns (answer, trace)."""
    messages = ([{"role": "system", "content": SYSTEM_PROMPT}] + chat_history[:-1]
                + language_note(chat_history[-1]["content"]) + chat_history[-1:])
    trace = []

    for step in range(MAX_STEPS):
        response = client.chat.completions.create(
            model=CHAT_MODEL, messages=messages, tools=TOOL_SPECS, temperature=0.2,
        )
        message = response.choices[0].message

        if not message.tool_calls:                     # plain text: the turn is finished
            return (message.content or "").strip(), trace

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.function.name,
                                         "arguments": c.function.arguments}}
                           for c in message.tool_calls],
        })

        for call in message.tool_calls:
            name = call.function.name
            arguments = json.loads(call.function.arguments or "{}")
            function = TOOL_FUNCTIONS.get(name)        # only our own names are allowed

            if function is None:
                result = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = function(**arguments)
                except Exception as error:
                    result = {"error": f"{type(error).__name__}: {error}"}

            trace.append({"step": step + 1, "tool": name,
                          "arguments": arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": json.dumps(result, ensure_ascii=False)})

    return "Sorry, I could not finish that. Please call our front desk at 07-555-0123.", trace


# ---------------------------------------------------------------------------
# 4. The Gradio page
# ---------------------------------------------------------------------------
GREETING = ("Hello! I am the online agent of Sunrise Community Clinic. "
            "Ask me about our opening hours, the doctor schedule, the cost of a visit, or our visit rules.\n\n"
            "您好！我是晨光社區診所的線上客服，用中文或英文問我都可以。")

EMPTY_TRACE = "### Tools used\n\n(nothing yet)"

# Sample questions, grouped like the customer questions you will test.
SAMPLE_GROUPS = {
    "Clinic information (get_clinic_info)": [
        "What are your opening hours?",
        "How do I register?",
        "Can I cancel or change my appointment?",
        "Do you give flu shots?",
        "Is there parking near the clinic?",
    ],
    "Doctor schedule (search_doctors)": [
        "Is there a pediatrician on Monday afternoon?",
        "When can I see a dermatologist?",
        "Is Dr. Lin available on Saturday morning?",   # on the schedule, but full
        "Which doctors work on Wednesday evening?",
    ],
    "Visit fee (calculate_visit_fee)": [
        "I am 70 years old. How much is it to see Dr. Wang?",
        "My 5-year-old needs the pediatrician. How much is it?",
        "How much is it for a 30-year-old and an 8-year-old to see Dr. Lin?",
        "How much is it to see the dermatologist?",   # no age: the agent should ask
    ],
    "Visit rules (search_clinic_rules)": [
        "I am 15 minutes late for my appointment. Can I still be seen?",
        "I have a fever and a cough. Do I need to do anything when I arrive?",
        "How do I get a copy of my medical records?",
        "Can I use a wheelchair in the clinic?",
        "What happens if there is a typhoon day off?",
    ],
    "Complaint, not in the data, not our job": [
        "I waited an hour and the nurse was rude. I want to speak to the manager.",
        "Do you have a dentist?",                      # not on the schedule
        "Do you do COVID tests?",                      # not in the data
        "I have a headache and a fever. What medicine should I take?",
        "What is the weather in Kaohsiung tomorrow?",
        "Please book a taxi for me.",
    ],
    "A conversation: click these in order, without Clear": [
        "Is there a pediatrician on Monday afternoon?",
        "How much is it for my 5-year-old?",
        "OK, what time does the afternoon session start?",
    ],
}

SAMPLE_GROUPS_ZH = {
    "診所資訊（get_clinic_info）": [
        "請問你們的門診時間是什麼時候？",
        "要怎麼掛號？",
        "可以取消或改約嗎？",
        "有打流感疫苗嗎？",
        "附近有停車位嗎？",
    ],
    "醫師診次（search_doctors）": [
        "星期一下午有小兒科嗎？",
        "皮膚科什麼時候有門診？",
        "林醫師星期六早上有診嗎？",                 # on the schedule, but full
        "星期三晚上有哪些醫師看診？",
    ],
    "掛號費用（calculate_visit_fee）": [
        "我 70 歲，看王醫師要多少錢？",
        "我 5 歲的小孩要看小兒科，總共多少錢？",
        "我 30 歲，帶 8 歲的小孩一起看林醫師，總共多少錢？",
        "看皮膚科要多少錢？",                         # no age: the agent should ask
    ],
    "看診規定（search_clinic_rules）": [
        "我預約的時間會晚到 15 分鐘，還可以看診嗎？",
        "我發燒又咳嗽，到診所要注意什麼？",
        "要怎麼申請病歷影本？",
        "診所可以進輪椅嗎？",
        "颱風放假的時候診所有開嗎？",
    ],
    "客訴、資料沒有、不是客服的事": [
        "我等了一個小時，護理師態度很差，我要找主任。",
        "你們有牙科嗎？",                             # not on the schedule
        "可以做新冠肺炎篩檢嗎？",                     # not in the data
        "我頭痛又發燒，應該吃什麼藥？",
        "高雄明天天氣如何？",
        "幫我叫一台計程車。",
    ],
    "多輪對話：依序點選，中間不要按 Clear": [
        "星期一下午有小兒科嗎？",
        "我 5 歲的小孩看的話，多少錢？",
        "好，那下午診幾點開始？",
    ],
}


def format_trace(trace):
    """Show the tool calls of the last answer, so the choice is visible."""
    if not trace:
        return "### Tools used\n\n(no tool was needed for this answer)"
    lines = ["### Tools used", ""]
    for entry in trace:
        lines.append(f"**{entry['step']}. {entry['tool']}**")
        lines.append("")
        lines.append(f"arguments: `{json.dumps(entry['arguments'], ensure_ascii=False)[:400]}`")
        lines.append("")
        result = entry["result"]
        if entry["tool"] == "search_clinic_rules" and result.get("found"):
            for chunk in result["chunks"]:            # show which chunks were found, and their scores
                lines.append(f"- chunk **{chunk['title']}** (score {chunk['score']}): {chunk['text'][:160]}...")
        else:
            lines.append(f"result: `{json.dumps(result, ensure_ascii=False)[:700]}`")
        lines.append("")
    return "\n".join(lines)


def start_chat():
    """The first state of the chat box: the greeting only.

    Used at start-up and when the visitor clears the chat, so the agent speaks first.
    """
    return [{"role": "assistant", "content": GREETING}]


def add_user_message(user_text, chat_history):
    """Show the question at once and empty the input box."""
    user_text = user_text.strip()
    if not user_text:
        return "", chat_history
    return "", chat_history + [{"role": "user", "content": user_text}]


def answer_question(chat_history):
    """Run the agent on the current conversation."""
    if not chat_history or chat_history[-1]["role"] != "user":
        return chat_history, EMPTY_TRACE

    try:
        answer, trace = run_agent(chat_history)
        panel = format_trace(trace)
    except Exception as error:
        answer = f"(system message) API error: {error}"
        panel = "### Tools used\n\n(the call failed)"

    return chat_history + [{"role": "assistant", "content": answer}], panel


with gr.Blocks(title="Sunrise Clinic Agent") as demo:
    gr.Markdown(
        "## Sunrise Community Clinic: customer service agent\n"
        "Every question goes to the agent. Clinic information and the doctor schedule come from a SQLite database, "
        "the visit rules are searched by meaning (RAG), the code works out the visit fee, and anything else goes to a human. "
        "The panel on the right shows which tool was used."
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(type="messages", value=start_chat(),
                                 height=430, label="Agent")
            with gr.Row():
                user_box = gr.Textbox(placeholder="Ask about the clinic, a doctor or a fee...",
                                      show_label=False, scale=8)
                send_button = gr.Button("Send", variant="primary", scale=1)
                clear_button = gr.Button("Clear", scale=1)

            gr.Markdown("**Sample questions** (click one, then press Send)")
            with gr.Tabs():
                with gr.Tab("English"):
                    for group, questions in SAMPLE_GROUPS.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)
                with gr.Tab("中文"):
                    for group, questions in SAMPLE_GROUPS_ZH.items():
                        gr.Examples(examples=questions, inputs=user_box, label=group)

        with gr.Column(scale=2):
            trace_panel = gr.Markdown(EMPTY_TRACE)

    # One turn: show the question, then let the agent work.
    send_button.click(
        add_user_message, [user_box, chatbot], [user_box, chatbot]
    ).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )

    user_box.submit(
        add_user_message, [user_box, chatbot], [user_box, chatbot]
    ).then(
        answer_question, chatbot, [chatbot, trace_panel]
    )

    # Clearing brings the greeting back and empties the trace panel.
    clear_button.click(start_chat, outputs=chatbot).then(
        lambda: ("", EMPTY_TRACE), outputs=[user_box, trace_panel]
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True)