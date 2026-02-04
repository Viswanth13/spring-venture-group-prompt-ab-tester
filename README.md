# Spring Venture Group Prompt A/B Tester — Trustworthy Conversational Outputs

> **Built specifically for Spring Venture Group** to demonstrate a “Start with Why” mindset for evaluating LLM prompts on real-world call transcripts.

🔗 **Live App:**  
[Live app](https://spring-venture-group-prompt-ab-tester-viswantht.streamlit.app/) <!-- TODO: replace with your Streamlit URL -->

---

## 1. Why this MVP exists

Spring Venture Group’s AI/Data roles focus on **trustworthy, experiment-ready insights** from unstructured conversations (e.g., sales and health insurance calls).

This MVP is not a generic chatbot. Instead, it:

- Takes a **single call transcript**
- Runs **Prompt A vs Prompt B** on the exact same input
- Shows **side-by-side raw model outputs** so you can visually compare structure, clarity, and usefulness

It’s a first step toward scientific A/B testing of prompts and models on call data.

---

## 2. What the app does

The app is a Streamlit UI that lets you:

1. **Load a transcript**
   - Paste directly into a text area
   - Upload a `.txt` file
   - Or click **“Load example transcript”** to use a built-in sample

2. **Define Prompt A and Prompt B**
   - Two separate prompt text areas
   - Prompts are designed to extract:
     - `summary`
     - `customer_intent`
     - `sentiment`
     - `friction_points`
     - `next_best_action`
   - Prompt B is written more strictly to illustrate quality differences

3. **Run both prompts on the same transcript**
   - Uses Google’s **Gemini** model (`gemini-2.5-flash`) via the official `google-genai` SDK
   - Optional checkbox to **run each prompt twice** (A1/A2, B1/B2) so you can eyeball consistency

4. **Display raw outputs side-by-side**
   - Column for Prompt A (Run 1 / Run 2)
   - Column for Prompt B (Run 1 / Run 2)
   - Outputs are shown as raw text in code blocks for easy comparison

> This MVP focuses on **qualitative comparison**. A future iteration can restore strict JSON parsing and metrics (compliance, coverage, consistency, hallucination risk).

---

## 3. Tech stack

**UI / Frontend**

- [Streamlit](https://streamlit.io/) for rapid app development
- Layout with `st.columns`, `st.expander`, etc.
- Branding via `assets/company_logo.png`

**LLM Engine**

- [Google Gemini](https://ai.google.dev/) via the official `google-genai` Python SDK
- Default model: `gemini-2.5-flash`
- Low temperature (`0.2`) for more stable outputs

**Runtime**

- Local: Python 3.10+
- Cloud: Streamlit Community Cloud

---

## 4. Project structure

```text
.
├─ app.py                 # Main Streamlit app
├─ requirements.txt       # Python dependencies
├─ README.md              # Project documentation
├─ assets/
│   └─ company_logo.png   # Placeholder or real SVG logo
└─ .streamlit/
    └─ config.toml        # (optional) Streamlit UI settings
