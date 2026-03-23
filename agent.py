"""
Creates a tool schema based on the tools present and calls LLM with the tool schema and user query
Based on the LLM output, calls respective tool and pass the output to LLM
Repeats the process until there is not tool to call
Returns the rewritten prompt and the before-after changes
"""

from langchain_anthropic import ChatAnthropic
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from langchain.prompts import PromptTemplate
from tools import (
    classify_input,
    diagnose_prompt,
    retrieve_best_practices,
    build_fix_plan,
    rewrite_prompt,
    score_improvement,
    infer_intent
)
import os
from dotenv import load_dotenv
from anthropic import Anthropic


load_dotenv()

client = ChatAnthropic(
    model_name="claude-sonnet-4-6",
    temperature=0
)
anthropic_client = Anthropic()

tools = [classify_input,
    diagnose_prompt,
    retrieve_best_practices,
    build_fix_plan,
    rewrite_prompt,
    score_improvement]

# prompt = hub.pull("hwchase17/react")
# agent = create_react_agent(prompt=prompt, tools=tools, llm=client)
# executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def estimate_delta(diagnosis: dict) -> int:
    severity_map = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    total = 0
    for mode, details in diagnosis.items():
        if isinstance(details, dict) and details.get("detected"):
            total += severity_map.get(details.get('severity', 'low'), 1)

    return total

def run_pipeline(query: str) -> dict:
    
    # Track tool outputs directly
    results = {
        "original"    : query,
        "classification": None,
        "diagnosis"   : None,
        "rewritten"   : None,
        "scores"      : None,
        "stopped"     : None
    }
    
    # Step 1 — classify
    classification = classify_input.invoke(query)
    results["classification"] = classification
    
    if classification in ["too_short", "not_a_prompt", 
                          "empty", "harmful", "meta_question", "non_english"]:
        results["stopped"] = classification
        if classification == "too_short":
            prompt = """You are a hint provider based on the given query. Your only job is to infer the hint based on the query
            Output should be in the below format:
            "translate to french" → "Try: Translate to French: [paste your text here]"
            "summarize"           → "Try: Summarize this: [paste your content here]"
            "write"               → "Try: Write a [type] about [topic] for [audience]"

            HARD RULES:
            Do not assume the incomplete query, you should only infer the hint purely based on the query given
            """
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                temperature=0,
                max_tokens= 100,
                system=prompt,
                messages=[{
                    "role": "user",
                    "content": query
                }]
            )
            hint = response.content[0].text.strip()
            results["hint"] = hint
        return results
    
    # step 1.1 - infer intent

    intent = infer_intent.invoke(query)
    results["intent"] = intent
    
    # Step 2 — diagnose
    diagnosis = diagnose_prompt.invoke(
        f"query: {query}\nintent: {intent.get('intent', 'unknown')}"
    )
    results["diagnosis"] = diagnosis

    estimated_delta = estimate_delta(diagnosis)
    
    # Step 3 — retrieve for each detected issue
    evidence = []
    for mode, details in diagnosis.items():
        if isinstance(details, dict) and details.get("detected"):
            chunks = retrieve_best_practices.invoke(mode)
            evidence.extend(chunks)
    
    # Step 4 — build fix plan
    fix_plan = build_fix_plan.invoke(
        f"diagnosis: {str(diagnosis)}\nevidence: {str(evidence)}"
    )
    
    # Step 5 — rewrite
    # Version A — filled
    filled = rewrite_prompt.invoke(
        f"classification: {classification}\n"
        f"original prompt: {query}\n"
        f"fix plan: {str(fix_plan)}\n"
        f"delta: {estimated_delta}\n"
        f"confidence: {intent.get('confidence', 1.0)}\n"
        f"intent: {intent.get('intent', 'unknown')}\n"
        f"mode: FILL — fill all [REQUIRED] fields with "
        f"best-guess values based on the original prompt. "
        f"No [REQUIRED] placeholders in output."
    )

    # Version B — template (current)
    template = rewrite_prompt.invoke(
        f"classification: {classification}\n"
        f"original prompt: {query}\n"
        f"fix plan: {str(fix_plan)}\n"
        f"delta: {estimated_delta}\n"
        f"confidence: {intent.get('confidence', 1.0)}\n"
        f"intent: {intent.get('intent', 'unknown')}\n"
        f"mode: TEMPLATE — use [REQUIRED] fields with "
        f"Default + Other options format."
    )

    results["rewritten_filled"]    = filled
    results["rewritten_template"]  = template\
    
   # Step 6 — score
    scores = score_improvement.invoke(
        f"original prompt: {query}\nrewritten prompt: {results['rewritten_filled']}"
    )
    results["scores"] = scores
    
    return results


if __name__ == "__main__":
    test_inputs = [
        # Should be LIGHT — specific, mostly complete
        "Write a 500-word blog post about AI for software engineers in a technical tone",
        # Should be FULL — vague, missing everything
        "Write something about climate change",
    ]
    
    for test in test_inputs:
        print(f"\n{'='*60}")
        print(f"📥 ORIGINAL: {test}")
        print('='*60)
        result = run_pipeline(test)
        
        if result["stopped"]:
            print(f"🛑 STOPPED: {result['stopped']}")
        else:
            print(f"📊 Classification : {result['classification']}")
            print(f"🩺 Issues found   : {result['diagnosis'].get('issues_found')}")
            print(f"📈 Score delta    : {result['scores'].get('overall_delta')}")
            print(f"\n✏️  REWRITTEN:\n{result['rewritten']}")