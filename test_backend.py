import os
import json

from app import generate_json, evaluate_prompt_output, EXAMPLE_TRANSCRIPT

# Make sure key exists
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise RuntimeError("No GEMINI_API_KEY / GOOGLE_API_KEY set in environment!")

schema_keys = ["summary", "customer_intent", "sentiment", "friction_points", "next_best_action"]

print("Calling Gemini once for Prompt A-style test...")

prompt_text = """
You are a conversation analyst at Spring Venture Group. Given the call transcript below, extract a single JSON object
with exactly these keys: summary, customer_intent, sentiment, friction_points, next_best_action.

Output STRICT JSON only.

Transcript:
{{TRANSCRIPT}}
"""

raw, parsed, err = generate_json(prompt_text, EXAMPLE_TRANSCRIPT, schema_keys)

print("\nRaw output:\n", raw[:500], "...\n")

if err:
    print("ERROR:", err)
else:
    print("Parsed JSON:\n", json.dumps(parsed, indent=2))

    metrics = evaluate_prompt_output(
        parsed=parsed,
        raw_1=raw,
        raw_2=None,
        required_keys=schema_keys,
        transcript_text=EXAMPLE_TRANSCRIPT,
    )
    print("\nMetrics:\n", json.dumps(metrics, indent=2))
