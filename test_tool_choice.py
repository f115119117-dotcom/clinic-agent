"""Tool-choice test for the Sunrise Clinic agent (G-3 notebook, Step 11).

For each question we write down the tool(s) we would accept, run the agent,
and check that at least one accepted tool was used.
"(none)" means: no tool should be used (the agent should politely refuse).

Run it with:  python test_tool_choice.py
It calls the real model, so it needs the .env file, and it waits between questions
to stay under the API rate limit (429 errors).
"""

import time

from app import run_agent

NONE = "(none)"

TEST_SET = [
    # clinic information
    ("What are your opening hours?",                                          {"get_clinic_info"}),
    ("請問診所在哪裡？",                                                        {"get_clinic_info"}),
    ("Can I cancel my appointment?",                                          {"get_clinic_info"}),
    ("附近有停車位嗎？",                                                        {"get_clinic_info"}),
    # doctor schedule
    ("Is there a pediatrician on Monday afternoon?",                          {"search_doctors"}),
    ("Is Dr. Lin available on Saturday morning?",                             {"search_doctors"}),
    ("星期三晚上有哪些醫師看診？",                                               {"search_doctors"}),
    ("皮膚科什麼時候有門診？",                                                   {"search_doctors"}),
    # visit fee (the program calculates)
    ("I am 70 years old. How much is it to see Dr. Wang?",                    {"calculate_visit_fee"}),
    ("我 5 歲的小孩要看小兒科，總共多少錢？",                                      {"calculate_visit_fee"}),
    ("How much is it for a 30-year-old and an 8-year-old to see Dr. Lin?",    {"calculate_visit_fee"}),
    # hand over to a person, not in the data
    ("I waited an hour and the nurse was rude. I want to speak to the manager.", {"transfer_to_human"}),
    ("Do you do COVID tests?",                                                {"transfer_to_human"}),
    # not the clinic's job
    ("I have a headache and a fever. What medicine should I take?",           {"get_clinic_info"}),
    ("What is the weather in Kaohsiung tomorrow?",                            {NONE}),
]

correct = 0
for question, accepted in TEST_SET:
    try:
        _, trace = run_agent([{"role": "user", "content": question}])
        used = [step["tool"] for step in trace]
    except Exception as error:                       # for example a 429 error
        used = [f"ERROR: {error}"]
    ok = (not used and NONE in accepted) or bool(accepted & set(used))
    correct += ok
    print(f"{'OK  ' if ok else 'MISS'} {question[:44]:44s} accepted {'/'.join(sorted(accepted)):22s} used {used}")
    time.sleep(4)                                    # a question with a tool needs at least 2 API calls

print(f"\ntool choice = {correct}/{len(TEST_SET)} = {correct / len(TEST_SET):.0%}")