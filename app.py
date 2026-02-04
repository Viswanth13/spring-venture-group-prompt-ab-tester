
import os
import json
import math
from typing import List, Tuple, Optional, Dict, Any

import streamlit as st
from google import genai

COMPANY_NAME = "Spring Venture Group"
LOGO_PATH = "assets/company_logo.png"

EXAMPLE_TRANSCRIPT = """Agent: Hi, thanks for calling Spring Venture Group. How can I help you today?
Customer: I'm trying to understand my options for health insurance. My employer plan is getting too expensive.
Agent: Got it. I'll ask a few questions about your needs and budget, then we can compare some options.
Customer: Sure, that sounds good.
Agent: Great. First, are you primarily concerned about monthly premium, out-of-pocket costs, or keeping your current doctors?
Customer: Mostly monthly premium, but I don't want surprise bills either.
Agent: Understood. Based on what you're telling me, I can walk you through a couple of plans and highlight trade-offs.
Customer: Okay, let's do that.
"""

# -----------------------
# Gemini Client & JSON Helper
# -----------------------

def _get_gemini_api_key() -> Optional[str]:
    """
    Resolve Gemini API key in the order:
    1) st.secrets["GEMINI_API_KEY"]
    2) env GEMINI_API_KEY
    3) env GOOGLE_API_KEY
    """
    key = None
    # Streamlit secrets (for Streamlit Cloud)
    try:
        if "GEMINI_API_KEY" in st.secrets:
            key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        # st.secrets may not be available in some contexts
        pass

    # Fallback to env vars (local / Colab)
    key = key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key


def get_gemini_client() -> Tuple[Optional[genai.Client], Optional[str]]:
    """
    Create a Gemini client or return an error string.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        return None, (
            "Gemini API key not found. Set st.secrets['GEMINI_API_KEY'] "
            "or the GEMINI_API_KEY / GOOGLE_API_KEY environment variable."
        )
    try:
        client = genai.Client(api_key=api_key)
        return client, None
    except Exception as e:
        return None, f"Failed to initialize Gemini client: {e}"


def _clean_json_text(raw_text: str) -> str:
    """
    Clean common formatting around JSON (e.g., ```json ... ```),
    removing both leading and trailing fences and optional 'json' tag.
    """
    text = raw_text.strip()

    # If model wrapped output in ``` fences
    if text.startswith("```"):
        lines = text.splitlines()

        # Drop first line if it's ``` or ```json or similar
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        # Drop last line if it's closing ```
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # If it still starts with 'json' or 'json:' strip that token
    lower = text.lower()
    if lower.startswith("json"):
        # Remove 'json' and any following colon/whitespace/newline
        text = text[4:].lstrip(" :\n\t")

    return text



def generate_json(
    prompt_text: str,
    transcript_text: str,
    schema_keys: List[str],
) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    """
    Call Gemini with the given prompt + transcript and try to parse STRICT JSON.

    Returns:
        raw_text: raw model output as string
        parsed_json_or_none: dict if JSON parsed successfully, else None
        error_or_none: error message if something failed, else None
    """
    client, err = get_gemini_client()
    if err:
        return "", None, err

    # Ensure schema keys are clean (for future use if needed)
    schema_keys = [k.strip() for k in schema_keys if k.strip()]

    # Replace placeholder {{TRANSCRIPT}} if present; otherwise, append transcript
    if "{{TRANSCRIPT}}" in prompt_text:
        filled_prompt = prompt_text.replace("{{TRANSCRIPT}}", transcript_text)
    else:
        filled_prompt = f"{prompt_text.strip()}\n\nTranscript:\n{transcript_text}"

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=filled_prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        raw_text = response.text or ""
    except Exception as e:
        return "", None, f"Gemini API error: {e}"

    if not raw_text.strip():
        return raw_text, None, "Gemini returned empty output."

    # Attempt JSON parsing (with cleaning pass)
    cleaned = _clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned)
        return raw_text, parsed, None
    except Exception as e:
        return raw_text, None, f"JSON parse error: {e}"


# -----------------------
# Evaluation / Scoring Helpers
# -----------------------

def compute_compliance(parsed: Optional[Dict[str, Any]], required_keys: List[str]) -> int:
    """
    compliance: 1 if valid JSON dict and contains all required keys, else 0.
    """
    if not isinstance(parsed, dict):
        return 0
    for key in required_keys:
        if key not in parsed:
            return 0
    return 1


def compute_coverage(parsed: Optional[Dict[str, Any]], required_keys: List[str]) -> float:
    """
    coverage: keys_present / required_keys
    """
    if not isinstance(parsed, dict) or not required_keys:
        return 0.0
    present = sum(1 for k in required_keys if k in parsed)
    return present / len(required_keys)


def _tokenize_for_similarity(text: str) -> set:
    """
    Very simple tokenization for similarity: lowercase word set.
    """
    return set(text.lower().split())


def compute_consistency(raw_1: str, raw_2: Optional[str]) -> float:
    """
    consistency: similarity between two runs (A1 vs A2 or B1 vs B2).

    If raw_2 is None or empty, we treat consistency as 1.0 (no second run yet).
    """
    if not raw_2:
        return 1.0  # single run only; we assume full self-consistency by definition
    t1 = _tokenize_for_similarity(raw_1)
    t2 = _tokenize_for_similarity(raw_2)
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return inter / union if union > 0 else 0.0


def _extract_text_items_for_risk(parsed: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """
    Collect strings from parsed JSON that we will try to match against the transcript
    to estimate hallucination risk.
    """
    items: List[str] = []
    if not isinstance(parsed, dict):
        return items

    for key in required_keys:
        if key not in parsed:
            continue
        value = parsed[key]
        if isinstance(value, str):
            if value.strip():
                items.append(value.strip())
        elif isinstance(value, list):
            for v in value:
                if isinstance(v, str) and v.strip():
                    items.append(v.strip())
        # You could extend to handle nested dicts here if needed

    return items


def compute_hallucination_risk(
    parsed: Optional[Dict[str, Any]],
    transcript_text: str,
    required_keys: List[str],
) -> Tuple[int, int, float]:
    """
    Hallucination-risk heuristic:
    - Extract text items from parsed JSON (strings & list items for required keys).
    - For each item, check if it appears as a substring in the transcript (case-insensitive).
    - risk_count = number of items with no evidence substring.
    - total_items = number of items examined.
    - normalized_risk = risk_count / max(total_items, 1).

    Returns:
        (risk_count, total_items, normalized_risk)
    """
    if not isinstance(parsed, dict):
        return 0, 0, 0.0

    transcript_lower = transcript_text.lower()
    items = _extract_text_items_for_risk(parsed, required_keys)
    if not items:
        return 0, 0, 0.0

    risk_count = 0
    for item in items:
        if item.lower() not in transcript_lower:
            risk_count += 1

    total_items = len(items)
    normalized_risk = risk_count / max(total_items, 1)
    return risk_count, total_items, normalized_risk


def compute_total_score(
    compliance: int,
    coverage: float,
    consistency: float,
    normalized_risk: float,
) -> float:
    """
    total = 0.4*compliance + 0.2*coverage + 0.2*consistency + 0.2*(1 - normalized_risk)
    """
    return (
        0.4 * float(compliance)
        + 0.2 * float(coverage)
        + 0.2 * float(consistency)
        + 0.2 * float(1.0 - max(0.0, min(1.0, normalized_risk)))
    )


def evaluate_prompt_output(
    parsed: Optional[Dict[str, Any]],
    raw_1: str,
    raw_2: Optional[str],
    required_keys: List[str],
    transcript_text: str,
) -> Dict[str, Any]:
    """
    Compute all metrics for a prompt (A or B):
    - compliance
    - coverage
    - consistency
    - risk_count, total_items, normalized_risk
    - total_score
    """
    compliance = compute_compliance(parsed, required_keys)
    coverage = compute_coverage(parsed, required_keys)
    consistency = compute_consistency(raw_1, raw_2)
    risk_count, total_items, normalized_risk = compute_hallucination_risk(
        parsed, transcript_text, required_keys
    )
    total_score = compute_total_score(compliance, coverage, consistency, normalized_risk)
    return {
        "compliance": compliance,
        "coverage": coverage,
        "consistency": consistency,
        "risk_count": risk_count,
        "total_items": total_items,
        "normalized_risk": normalized_risk,
        "total_score": total_score,
    }


{st.set_page_config(
    page_title=f"{COMPANY_NAME} Prompt A/B Tester",
    page_icon="🧪",
    layout="wide",
)
}
# --- Header with logo ---
col_logo, col_title = st.columns([1, 3])

with col_logo:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_column_width=True)
    else:
        st.markdown(
            f"**{COMPANY_NAME}**\n\n_(Logo missing at `{LOGO_PATH}` — please add it to the assets folder.)_"
        )

with col_title:
    st.title("Spring Venture Group Prompt A/B Tester — Trustworthy Conversational Metrics")
    st.markdown(
        """
        This prototype Streamlit app is built **specifically for Spring Venture Group** to explore how different
        LLM prompts extract structured, trustworthy metrics from unstructured sales and health insurance conversations.
        """
    )

st.markdown("---")

# --- Start with WHY ---
st.subheader("Why this app?")
st.markdown(
    """
    Modern call transcripts are rich but unstructured. To make them useful, we define prompts that transform raw
    conversations into **structured signals** (e.g., intent, sentiment, friction, next-best-action).  
    This app is designed to:
    - Compare two prompt definitions side-by-side (Prompt A vs Prompt B)
    - Evaluate how trustworthy and consistent their outputs are
    - Provide lightweight, explainable metrics that align with a **scientific, experiment-driven mindset**
    """
)

st.markdown("---")

# -----------------------
# 1. Conversation Transcript
# -----------------------
st.header("1. Conversation Transcript")

st.markdown(
    """
    Paste a **single call transcript** here, or upload a `.txt` file.  
    You can also load a small example transcript for quick demo purposes.
    """
)

# Session state for transcript text
if "transcript_text" not in st.session_state:
    st.session_state["transcript_text"] = ""

col_input, col_side = st.columns([3, 1])

with col_input:
    transcript_text = st.text_area(
        "Paste call transcript",
        value=st.session_state["transcript_text"],
        height=260,
        placeholder="Paste the full conversation transcript here...",
    )
    # keep state in sync with manual edits
    st.session_state["transcript_text"] = transcript_text

with col_side:
    uploaded_file = st.file_uploader(
        "Or upload transcript (.txt)",
        type=["txt"],
        help="Upload a plain-text file with the call transcript.",
    )
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode("utf-8", errors="ignore")
            st.session_state["transcript_text"] = content
            st.success("Transcript loaded from file. You can review/edit it on the left.")
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")

    if st.button("Load example transcript"):
        st.session_state["transcript_text"] = EXAMPLE_TRANSCRIPT
        st.info("Example transcript loaded. You can edit it in the text area.")

st.markdown(
    """
    _This transcript will be used as the shared input when we later compare **Prompt A** vs **Prompt B** using Gemini._
    """
)

st.info(
    "Next steps (to be implemented): Prompt A/B inputs, expected JSON schema keys, Gemini-backed extraction, "
    "and evaluation metrics (compliance, coverage, consistency, risk)."
)

# -----------------------
# 2. Prompt Definitions & Expected JSON Schema
# -----------------------
st.markdown("---")
st.header("2. Prompt Definitions & Expected JSON Schema")

st.markdown(
    """
    Here we define **Prompt A** and **Prompt B**, along with the expected JSON schema keys.  
    Both prompts should instruct Gemini to output **strict JSON** with exactly these keys and no extra text.
    """
)

DEFAULT_SCHEMA_KEYS = "summary, customer_intent, sentiment, friction_points, next_best_action"

schema_keys_str = st.text_input(
    "Expected JSON Schema Keys (comma-separated)",
    value=DEFAULT_SCHEMA_KEYS,
    help="These keys will be used to validate JSON compliance and coverage. Example: summary, customer_intent, sentiment, friction_points, next_best_action",
)

col_pa, col_pb = st.columns(2)

prompt_a_default = f"""
You are a conversation analyst at Spring Venture Group. Given the call transcript below, extract a single JSON object
with exactly these keys: {schema_keys_str}.

Definitions:
- summary: 2-3 sentence summary of the call.
- customer_intent: what the customer is trying to achieve (in their own words).
- sentiment: overall sentiment of the customer (e.g., positive, neutral, negative).
- friction_points: list of concrete obstacles, confusions, or objections raised.
- next_best_action: one recommended next step for the agent or sales team.

Rules:
- Base your analysis only on the transcript. Do not invent details.
- Output STRICT JSON with double-quoted keys and values.
- Do NOT include any explanation outside the JSON.

Transcript:
{{TRANSCRIPT}}
"""

prompt_b_default = f"""
You are a quality analyst at Spring Venture Group focusing on trustworthy metrics for A/B testing of sales conversations.

Given the call transcript, return a STRICT JSON object with exactly these keys: {schema_keys_str}.

Guidelines:
- summary: concise summary focusing on customer needs and decision stage.
- customer_intent: main goal of the customer, in natural language.
- sentiment: one of ["very_negative", "negative", "neutral", "positive", "very_positive"].
- friction_points: array of short bullet-like strings describing pain points or objections.
- next_best_action: one concrete, actionable recommendation for the next outreach.

Rules:
- Do NOT hallucinate. If information is missing, use null or an empty list.
- Output ONLY valid JSON. No markdown, no commentary, no backticks.

Transcript:
{{TRANSCRIPT}}
"""

with col_pa:
    st.subheader("Prompt A")
    prompt_a_text = st.text_area(
        "Prompt A definition",
        value=prompt_a_default.strip(),
        height=260,
        help="This prompt will be combined with the transcript and expected schema keys when calling Gemini.",
    )

with col_pb:
    st.subheader("Prompt B")
    prompt_b_text = st.text_area(
        "Prompt B definition",
        value=prompt_b_default.strip(),
        height=260,
        help="Use a slightly different strategy or wording so we can compare Prompt A vs Prompt B.",
    )

with st.expander("View built-in prompt templates for sales/health calls"):
    st.markdown(
        """
        **Template 1 – Baseline extraction (intent + sentiment + next-best-action)**  
        ```text
        You are a conversation analyst at Spring Venture Group. Given the call transcript below, extract a single JSON object
        with exactly these keys: summary, customer_intent, sentiment, friction_points, next_best_action.
        ...
        ```
        
        **Template 2 – QA-focused with discrete sentiment labels**  
        ```text
        You are a quality analyst at Spring Venture Group focusing on trustworthy metrics for A/B testing of sales conversations.
        ...
        sentiment must be one of ["very_negative", "negative", "neutral", "positive", "very_positive"].
        ...
        ```

        **Template 3 – Minimal debugging template**  
        ```text
        You are a JSON-only bot. Given the transcript, return a JSON object with exactly the requested keys.
        If you are unsure, set the value to null. Never include explanations or extra keys.
        ```
        """
    )

st.caption(
    "Later, these prompts and schema keys will be passed to Gemini to generate JSON outputs for Prompt A vs Prompt B."
)

# -----------------------
# 3. Run Prompt A/B with Gemini
# -----------------------
st.markdown("---")
st.header("3. Run Prompt A/B with Gemini")

st.markdown(
    """
    Click the button below to run **Prompt A** and **Prompt B** on the same transcript using Gemini.  
    Optionally, run each prompt twice (A1/A2 and B1/B2) to estimate consistency.
    """
)

run_twice = st.checkbox(
    "Run each prompt twice to estimate consistency (A1/A2 and B1/B2)",
    value=True,
    help="If checked, each prompt is executed twice with the same settings. We compare the two outputs for consistency.",
)

if st.button("Run A/B JSON extraction and evaluation"):
    transcript = st.session_state.get("transcript_text", "").strip()
    if not transcript:
        st.error("Transcript is empty. Please paste or load a transcript in section 1.")
    else:
        required_keys = [k.strip() for k in schema_keys_str.split(",") if k.strip()]

        # --- Prompt A runs ---
        with st.spinner("Calling Gemini for Prompt A (run 1)..."):
            raw_a1, json_a1, err_a1 = generate_json(
                prompt_text=prompt_a_text,
                transcript_text=transcript,
                schema_keys=required_keys,
            )

        raw_a2 = None
        json_a2 = None
        err_a2 = None
        if run_twice:
            with st.spinner("Calling Gemini for Prompt A (run 2)..."):
                raw_a2, json_a2, err_a2 = generate_json(
                    prompt_text=prompt_a_text,
                    transcript_text=transcript,
                    schema_keys=required_keys,
                )

        # Choose a parsed JSON for metrics (prefer first successful)
        parsed_a = json_a1 or json_a2

        # --- Prompt B runs ---
        with st.spinner("Calling Gemini for Prompt B (run 1)..."):
            raw_b1, json_b1, err_b1 = generate_json(
                prompt_text=prompt_b_text,
                transcript_text=transcript,
                schema_keys=required_keys,
            )

        raw_b2 = None
        json_b2 = None
        err_b2 = None
        if run_twice:
            with st.spinner("Calling Gemini for Prompt B (run 2)..."):
                raw_b2, json_b2, err_b2 = generate_json(
                    prompt_text=prompt_b_text,
                    transcript_text=transcript,
                    schema_keys=required_keys,
                )

        parsed_b = json_b1 or json_b2

        # --- Compute metrics for A & B ---
        metrics_a = evaluate_prompt_output(
            parsed=parsed_a,
            raw_1=raw_a1 or "",
            raw_2=raw_a2,
            required_keys=required_keys,
            transcript_text=transcript,
        )
        metrics_b = evaluate_prompt_output(
            parsed=parsed_b,
            raw_1=raw_b1 or "",
            raw_2=raw_b2,
            required_keys=required_keys,
            transcript_text=transcript,
        )

        # --- Comparison table ---
        rows = []
        for label, m in [("A", metrics_a), ("B", metrics_b)]:
            rows.append(
                {
                    "Prompt": label,
                    "Compliance": m["compliance"],
                    "Coverage": f"{m['coverage']:.2f}",
                    "Consistency": f"{m['consistency']:.2f}",
                    "Risk (items without evidence)": m["risk_count"],
                    "Items Checked": m["total_items"],
                    "Normalized Risk": f"{m['normalized_risk']:.2f}",
                    "Total Score": f"{m['total_score']:.3f}",
                }
            )

        st.subheader("Prompt A vs Prompt B — Metrics Comparison")
        st.table(rows)

        # --- Declare winner ---
        score_a = metrics_a["total_score"]
        score_b = metrics_b["total_score"]

        if score_a > score_b:
            st.success(
                f"Prompt A wins with a higher total score ({score_a:.3f} vs {score_b:.3f}). "
                "This suggests Prompt A is better under the current weighting of compliance, coverage, consistency, and risk."
            )
        elif score_b > score_a:
            st.success(
                f"Prompt B wins with a higher total score ({score_b:.3f} vs {score_a:.3f}). "
                "This suggests Prompt B is better under the current weighting of compliance, coverage, consistency, and risk."
            )
        else:
            st.info(
                f"Prompt A and Prompt B have the same total score ({score_a:.3f}). "
                "You may want to inspect the raw outputs and individual metrics manually."
            )

        # --- Side-by-side detailed outputs ---
        st.markdown("---")
        st.subheader("Detailed Outputs")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("### Prompt A")
            if err_a1:
                st.warning(f"Run 1 issue: {err_a1}")
            if run_twice and err_a2:
                st.warning(f"Run 2 issue: {err_a2}")

            with st.expander("Raw outputs (Prompt A)", expanded=True):
                if raw_a1:
                    st.markdown("**Run 1**")
                    st.code(raw_a1, language="json")
                if run_twice and raw_a2:
                    st.markdown("**Run 2**")
                    st.code(raw_a2, language="json")

        with col_b:
            st.markdown("### Prompt B")
            if err_b1:
                st.warning(f"Run 1 issue: {err_b1}")
            if run_twice and err_b2:
                st.warning(f"Run 2 issue: {err_b2}")


            with st.expander("Raw outputs (Prompt B)", expanded=True):
                if raw_b1:
                    st.markdown("**Run 1**")
                    st.code(raw_b1, language="json")
                if run_twice and raw_b2:
                    st.markdown("**Run 2**")
                    st.code(raw_b2, language="json")
