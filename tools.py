"""
PromptAutopsy — LangChain Tools
=================================
Wraps each pipeline function in a @tool decorator
so the LangChain ReAct agent can call them.
One tool per pipeline step.
"""

from langchain.tools import tool
from retrieve import retrieve
from anthropic import Anthropic
import json
import os
from dotenv import load_dotenv

load_dotenv()

TEMPERATURE_DETERMINISTIC = 0
TEMPERATURE_GENERATIVE = 0.1

client = Anthropic()

@tool
def classify_input(query: str) -> str:
    """
    Classifies the user input into one of 11 types.
    Routes to appropriate handler before any RAG runs.
    Returns a classification string.
    """

    word_count = len(query.split())
    if word_count > 5000:
        return "too_long"
    
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=50,
        temperature=TEMPERATURE_DETERMINISTIC,
        system="""
        You are a prompt classifier. Classify the given input into exactly one 
        of these types and return ONLY the classification string — no explanation, 
        no markdown, no punctuation:

        valid_prompt, keyword_dump, meta_question, not_a_prompt, empty, 
        too_short, harmful, non_english, contains_code, too_long, has_placeholders

        CLASSIFICATION RULES — reason about intent, do not pattern match:

        valid_prompt
        → Has a clear task intent that an LLM could act on
        → Test: would a competent person know roughly what to produce?
        → If yes → valid_prompt, even if vague or missing details

        too_short
        → Has NO actionable task and NO topic
        → Task verb without topic = too_short
        → Topic without task verb = keyword_dump (not too_short)
        → Test: is there anything here an LLM could act on? If no → too_short

        keyword_dump
        → Has topic signals but no task verb or sentence structure
        → Looks like a search query, not an instruction

        meta_question
        → The question is ABOUT prompting, AI, or LLMs themselves
        → Not a task for the LLM to perform — a question about how to prompt

        not_a_prompt
        → Conversational input not intended as an LLM task
        → Greetings, reactions, statements directed at the AI personally

        empty
        → Blank, whitespace only, or meaningless characters

        harmful
        → Requests information or content that could cause real-world harm
        → Weapons, violence, illegal activity, exploitation

        non_english
        → Primary language is not English
        → Mixed English/other = non_english if the core task is in another language

        contains_code
        → Input contains a code snippet, function, or technical syntax
        → Even if surrounded by natural language instructions

        too_long
        → Exceeds 500 words — too long to diagnose meaningfully

        has_placeholders
        → Contains unfilled placeholders like [TOPIC], [AUDIENCE], {name}, <variable>
        → Prompt is a template, not a complete instruction

        PRIORITY ORDER when multiple rules could apply:
        1. harmful
        2. empty  
        3. too_long
        4. non_english
        5. contains_code
        6. has_placeholders
        7. meta_question
        8. too_short
        9. keyword_dump
        10. not_a_prompt
        11. valid_prompt
        """,
        messages=[
            {"role": "user", "content": query}
        ]
    )
    classification = response.content[0].text.strip()
    return classification

@tool
def diagnose_prompt(query: str, intent: str = "unknown") -> dict:
    """
    calls the LLM with user query and intent
    And outputs in the below format
    {
    "vague_instruction":        {"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"},
    "missing_context":          {"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"},
    "wrong_format":             {"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"},
    "conflicting_instructions": {"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"},
    "missing_examples":         {"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"},
    "overall_health":           "broken/fixable/functional/healthy",
    "issues_found":             0-5
    }
    """
    prompt=f"""
        You are a prompt engineering expert trained on 
        Anthropic's prompting guidelines, OpenAI's prompt 
        engineering guide, and peer-reviewed research on 
        LLM behaviour.

        Your job is to diagnose the health of the following 
        user prompt by scoring it against exactly 5 failure 
        modes.

        User prompt: {query}

        User intent: {intent}

        Score each failure mode as true or false.
        Only flag a failure mode if it is GENUINELY missing 
        given the user's intent — not just technically absent.

        Failure modes to check:

        1. VAGUE_INSTRUCTION
        Is the task unclear? Does the prompt fail to define 
        what a good output looks like?
        
        2. MISSING_CONTEXT  
        Is role, audience, or purpose absent in a way that 
        would cause output variance?

        3. WRONG_FORMAT
        Is structured output requested (JSON, table, list) 
        without a schema or example provided?

        4. CONFLICTING_INSTRUCTIONS
        Do two or more instructions contradict each other 
        in a way the LLM cannot resolve?

        5. MISSING_EXAMPLES
        Is this a complex or pattern-sensitive task where 
        a few-shot example would significantly reduce 
        output variance?

        For contains_code inputs:
        - Focus diagnosis on the INSTRUCTIONS surrounding the code
        - vague_instruction: are requirements clear? (sort order, edge cases, return type)
        - missing_context: is language/version/constraints specified?
        - missing_examples: are input/output examples provided?
        - Do NOT flag wrong_format for code inputs

        Return ONLY valid JSON. No explanation, no markdown, 
        no code fences. Follow this exact schema:

        {{
        "vague_instruction":        {{"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"}},
        "missing_context":          {{"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"}},
        "wrong_format":             {{"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"}},
        "conflicting_instructions": {{"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"}},
        "missing_examples":         {{"detected": true/false, "severity": "critical/high/medium/low", "reason": "one sentence"}},
        "overall_health":           "broken/fixable/functional/healthy",
        "issues_found":             0-5
        }}
        """
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=prompt,
        temperature=TEMPERATURE_DETERMINISTIC,
        messages=[{"role": "user", "content": query}]
    )
    clean = response.content[0].text.strip()
    clean = clean.replace("```json", "").replace("```","")
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {
            "vague_instruction"       : {"detected": True,  "severity": "critical", "reason": "Could not parse diagnosis — treating as broken"},
            "missing_context"         : {"detected": True,  "severity": "critical", "reason": "Could not parse diagnosis — treating as broken"},
            "wrong_format"            : {"detected": False, "severity": "low",      "reason": ""},
            "conflicting_instructions": {"detected": False, "severity": "low",      "reason": ""},
            "missing_examples"        : {"detected": False, "severity": "low",      "reason": ""},
            "overall_health"          : "broken",
            "issues_found"            : 2
        }

@tool
def retrieve_best_practices(failure_mode: str, k:int=3) -> list:
    """
    Queries the RAG index for best practices
    related to a specific failure mode.
    Returns top-k chunks with source and text.
    """
    query = f"how to fix {failure_mode.replace('_',' ')} in a prompt"
    nodes = retrieve(query, k)

    return [{
        "source": node.metadata["source"],
        "text": node.text,
        "score": round(node.score, 3)
    }
    for node in nodes
    ]

@tool
def build_fix_plan(input_text: str) -> dict:
    """
    Takes diagnosis and evidence as a combined string.
    Converts retrieved guidelines into executable 
    transformation actions.
    Returns a structured fix plan dict.
    """

    prompt = """
    You are prompt fix plan executor, you recieve diagnosis and evidence.
    Your only job is to convert the evidence into an executable transformational actions.
    Do not give any advice - just provide instructions that rewrite_prompt will execute.
    Output should in the below json schema format:
    {
    "fix_plan": [
            {
                "issue"      : "vague_instruction",
                "action"     : "ADD_ROLE",
                "instruction": "exact instruction for rewrite tool",
                "format"     : "template to follow",
                "source"     : "citation from evidence"
            }
        ]
    }
    """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=prompt,
        temperature=TEMPERATURE_DETERMINISTIC,
        messages=[{"role": "user", "content": input_text}]
    )

    plan = response.content[0].text.strip()
    plan = plan.replace("```json","").replace("```","")
    try:
        return json.loads(plan)
    except json.JSONDecodeError:
        # If JSON is truncated — return a minimal valid plan
        return {
            "fix_plan": [{
                "issue"      : "parsing_error",
                "action"     : "RESUBMIT",
                "instruction": "Input was too long. Please simplify your prompt.",
                "format"     : "",
                "source"     : ""
            }]
        }

@tool
def rewrite_prompt(input_text: str) -> str:
    """
    Takes the original prompt and fix plan as a combined string.
    Executes every action in the fix plan exactly.
    Returns rewritten prompt with every change annotated
    with its source citation. 
    """

    prompt = """
        ROLE
        You are a prompt rewriter. Your only job is to execute
        the given fix plan and produce an enhanced version of
        the original prompt.

        ----------------------------------------
        CORE BEHAVIOR
        - Execute EVERY action in the fix plan exactly as specified
        - Do NOT add anything not in the fix plan
        - Do NOT change the original intent
        - Do NOT add unrequested sections or commentary

        When intent is ambiguous (keyword_dump or ambiguous input):
        - Generate 2-3 variants, one per likely intent
        - Label each: === INTENT: <intent_name> ===
        - Keep structure parallel across variants

        When intent is clear (valid_prompt):
        - Generate exactly ONE rewritten prompt
        - No variants needed

        ----------------------------------------
        HARD EXECUTION RULES
        - Execute every fix plan action — skip NONE
        - Annotate every change inline immediately after it
        - Always include <examples> block if fix plan contains ADD_EXAMPLES
        - Never truncate — complete every section fully

        ----------------------------------------
        FILLER FORMAT RULES
        For mandatory fields use:
        [REQUIRED: field_name]

        For fields with common options, add on the NEXT line:
        Options: option1 | option2 | option3 | option4

        NEVER mix placeholder and options on the same line.
        NEVER fill in [REQUIRED] fields with assumptions.
        NEVER use generic text like [your topic] — always use
        [REQUIRED: specific_field_name]

        Example:
        Tone: [REQUIRED: tone]
        Options: formal | conversational | persuasive | analytical

        ----------------------------------------
        CITATION FORMAT
        Cite every change inline immediately after it:
        <!-- ACTION_NAME: source_name -->

        Example:
        <role>You are an expert copywriter.</role>
        <!-- ADD_ROLE: anthropic_prompting_docs -->

        NEVER move citations to the end.
        NEVER skip citations.

        ----------------------------------------
        EXAMPLE FORMAT
        When fix plan contains ADD_EXAMPLES:
        - Include a COMPLETE, realistic, filled-in example
        - NOT a template — a real working example
        - Wrap in <examples><example>...</example></examples>
        - Never use placeholder text inside examples
        - Never truncate examples

        Good example:
        <examples>
        <example>
        Subject: Cut reporting time by 40%
        Hi Sarah,
        Your team spends 8 hours weekly on manual reports...
        </example>
        </examples>

        Bad example:
        <examples>
        <example>[example subject line]</example>
        </examples>

        ----------------------------------------
        CLASSIFICATION HANDLING
        The classification and routing has already been handled 
        upstream. You will only receive valid inputs to rewrite.
        Focus entirely on executing the fix plan.

        ----------------------------------------

        REQUIRED FIELD RULES:
        - Every [REQUIRED: field_name] you generate MUST have options
        - Options must be on the next line immediately after the field
        - Format: Options: option1 | option2 | option3 | option4
        - Minimum 3 options, maximum 6 options per field
        - Never leave a [REQUIRED] field without options
        - This applies to ALL fields including role, audience, and purpose

        Example:
        You are a [REQUIRED: role]
        Options: science communicator | journalist | policy analyst | educator

        writing for [REQUIRED: audience]
        Options: general public | high school students | policy makers | professionals

        with the goal of [REQUIRED: purpose]
        Options: inform | persuade | raise awareness | drive action

        ----------------------------------------
        MODE HANDLING
        You will receive a mode instruction in the input.

        If mode is FILL:
        - Fill ALL [REQUIRED] fields with best-guess values
        - Base guesses on context from the original prompt
        - No [REQUIRED] placeholders in output
        - No options lists
        - Output should be ready to paste into any LLM immediately
        - Citations still required

        If mode is TEMPLATE:
        - Use [REQUIRED: field_name] placeholders
        - Add Default: best_guess on next line
        - Add Other options: opt1 | opt2 | opt3 on line after that
        - Maximum 3 other options per field

        -----------------------------------------
        JSON OUTPUT RULES:
        When rewriting prompts that request JSON output:
        - Instruct the LLM to output pretty-printed JSON
        - Use 2-space indentation
        - Never instruct to output minified JSON
        - Add: "Format the JSON with 2-space indentation for readability"

        ----------------------------------------
        VERBOSITY RULES — follow based on delta:

        delta 2-4 (LIGHT):
        - Make minimal targeted changes only
        - Do NOT add XML structure
        - Do NOT add examples unless ADD_EXAMPLES in fix plan
        - Feel like a light edit of the original
        - Maximum 3-4 lines added to original

        delta 5-8 (MODERATE):
        - Add role and context block
        - Add constraints as numbered list
        - No examples unless ADD_EXAMPLES in fix plan
        - Structured but not overwhelming

        delta 9+ (FULL):
        - Full compiler output — current behaviour
        - role + context + instructions + examples
        - Complete reconstruction

        -----------------------------------------

        CONFIDENCE ROUTING — based on confidence score:
        confidence ≥ 0.8  → single rewrite (intent is clear)
        confidence 0.5-0.8 → 2 intent variants
        confidence < 0.5  → 3 intent variants (maximum ambiguity)

        For keyword_dump classification — always use confidence 
        routing regardless of other rules.

        -------------------------------------------

        NEVER RULES
        - NEVER add sections not in the fix plan
        - NEVER change the user's original intent
        - NEVER fill [REQUIRED] fields with assumptions
        - NEVER use partial or placeholder examples
        - NEVER truncate output mid-section
        - NEVER skip a fix plan action
        - NEVER move citations to the end
        - NEVER hallucinate sources not in the evidence
        """


    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        temperature=TEMPERATURE_GENERATIVE,
        system=prompt,
        messages=[{
            "role": "user",
            "content": input_text
        }]
    )
    rewritten_prompt = response.content[0].text.strip()
    return rewritten_prompt

@tool
def score_improvement(input_text: str) -> dict:
    """
    Takes the original prompt and rewritten prompt as a combined input and calls the LLM
    Calculates the delta change between original and rewritten prompts
    Returns True only if delta >= 2 else returns False
    """

    prompt = """
            You are a prompt judge. Your only job is to score
            the original and rewritten prompts.

            HARD RULES:
            - Score each dimension on a scale of 1-5 ONLY
            - 1 = very poor, 5 = excellent
            - Never use scores outside 1-5
            - overall_delta = sum of all three deltas
            - Maximum possible overall_delta is 12
            - show_rewrite = true ONLY if overall_delta >= 2
            - Judge only based on the given prompts
            - Do not assume anything not in the prompts

            Output ONLY valid JSON. No explanation, no markdown,
            no code fences. Follow this exact schema:
            {
            "clarity":      {"before": 1-5, "after": 1-5, "delta": 0-4, "before_reason": "one sentence", "after_reason": "one sentence"},
            "specificity":  {"before": 1-5, "after": 1-5, "delta": 0-4, "before_reason": "one sentence", "after_reason": "one sentence"},
            "completeness": {"before": 1-5, "after": 1-5, "delta": 0-4, "before_reason": "one sentence", "after_reason": "one sentence"},
            "overall_delta": 0-12,
            "show_rewrite":  true/false
            }
            """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        temperature=TEMPERATURE_DETERMINISTIC,
        max_tokens=1024,
        system=prompt,
        messages=[{
            "role":"user",
            "content":input_text
        }]
    )
    output = response.content[0].text.strip()
    output = output.replace("```json","").replace("```","")
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {
            "clarity"     : {"before": 1, "after": 3, "delta": 2, "before_reason": "Could not score", "after_reason": "Could not score"},
            "specificity" : {"before": 1, "after": 3, "delta": 2, "before_reason": "Could not score", "after_reason": "Could not score"},
            "completeness": {"before": 1, "after": 3, "delta": 2, "before_reason": "Could not score", "after_reason": "Could not score"},
            "overall_delta": 6,
            "show_rewrite" : True
        }
    
@tool
def infer_intent(query: str) -> dict:
    """
    Takes the query and calls the LLM to infer the intent of the user query.
    Later this intent will be used in rewriting the prompt.
    """

    prompt = """
        You are an intent inferrer. Your ONLY job is to take the given user query and return the following fields:

        - **intent**: what could be the intent of the given query
        - **confidence**: confidence rating in the range of 0.0 to 1.0 (0.0 = no confidence, 1.0 = fully certain the inferred intent is correct)
        - **alternatives**: two alternative intents
        - **topic**: what subject matter the query relates to
        - **format**: the ideal output format for the given query
        - **audience**: the target audience for the given query

        Return your output as pretty-printed JSON with 2-space indentation:
        ```json
        {
        "intent": "primary intent string",
        "confidence": 0.0-1.0,
        "alternatives": ["alt1", "alt2"],
        "topic": "subject matter",
        "format": "output format",
        "audience": "target audience"
        }
        ```

        **HARD RULES:**
        - Do not assume facts about the user's identity or any unstated personal context (e.g., name, location, profession, prior history).

        - Inference about likely audience, response format, topic category, and alternative intents IS expected and required — these fields must be inferred from the language, complexity, and phrasing of the query itself.

        - Your output must be grounded in what is present in the query; do not fabricate unstated facts.

        ---

        <examples>
        <example>
        EXAMPLE INPUT:
        "What is the fastest sorting algorithm?"

        EXAMPLE OUTPUT:
        ```json
        {
        "intent": "Learn which sorting algorithm has the best time complexity",
        "confidence": 0.85,
        "alternatives": [
            "Seeking a sorting algorithm suited to a specific use case or dataset",
            "Comparing sorting algorithms in preparation for a technical interview"
        ],
        "topic": "Computer Science / Algorithms",
        "format": "Explanatory answer with a comparison table of time complexities",
        "audience": "Intermediate developer or computer science student"
        }
        ```
        </example>
        </examples>
    """

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        temperature=TEMPERATURE_DETERMINISTIC,
        system=prompt,
        messages=[{"role":"user", "content": query}]
    )
    intent = response.content[0].text.strip()
    intent = intent.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(intent)
    except json.JSONDecodeError:
        return {
            "intent": "unknown",
            "confidence": 0.5,
            "alternatives": [],
            "topic": "unknown",
            "format": "unknown",
            "audience": "general"
        }

if __name__ == "__main__":
    test_prompt = "python learning matplotli"
    
    print("\n🔬 Testing tools individually...\n")
    
    # Test 1 — classify_input
    print("1️⃣  classify_input")
    classification = classify_input.invoke(test_prompt)
    print(f"   Result: {classification}\n")
    
    # Test 2 — diagnose_prompt
    print("2️⃣  diagnose_prompt")
    diagnosis = diagnose_prompt.invoke(test_prompt)
    print(f"   Result: {json.dumps(diagnosis, indent=2)}\n")
    
    # Test 3 — retrieve_best_practices
    print("3️⃣  retrieve_best_practices")
    evidence = retrieve_best_practices.invoke("vague_instruction")
    print(f"   Retrieved {len(evidence)} chunks")
    print(f"   Top source: {evidence[0]['source']}\n")
    
    # Test 4 — build_fix_plan
    print("4️⃣  build_fix_plan")
    fix_plan = build_fix_plan.invoke(
        f"diagnosis: {str(diagnosis)}\nevidence: {str(evidence)}"
    )
    print(f"   Actions: {len(fix_plan['fix_plan'])}\n")
    
    # Test 5 — rewrite_prompt
    print("5️⃣  rewrite_prompt")
    rewritten = rewrite_prompt.invoke(
        f"original prompt: {test_prompt}\n\nfix plan: {str(fix_plan)}"
    )
    print(f"   Rewritten: {rewritten}\n")
    
    # Test 6 — score_improvement
    print("6️⃣  score_improvement")
    scores = score_improvement.invoke(
        f"original prompt: {test_prompt}\n\nrewritten prompt: {rewritten}"
    )   
    print(f"   Overall delta: {scores['overall_delta']}")
    print(f"   Show rewrite:  {scores['show_rewrite']}\n")
    
    print("✅ All tools tested successfully")