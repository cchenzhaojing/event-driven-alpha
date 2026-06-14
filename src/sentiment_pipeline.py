#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_qwen_gpu.py —— Event-level sentiment scoring with Qwen2.5-3B-Instruct (GPU-first)

Steps:
1. Read issuer panel with `final_text` and event dummy columns.
2. Drop rows where final_text is empty.
3. For each (final_text, event_topic=1), ask Qwen to give a sentiment score in [-1, 1].
4. Pivot back to wide format: one *_sent column per event topic.
5. Save final CSV.

Notes:
- Uses HuggingFace open-source model `Qwen/Qwen2.5-3B-Instruct`.
- NO OpenAI, NO API keys, NO gated repos.
- If CUDA is available, the model is moved to GPU explicitly.
"""

import math
import time
import re
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

# ======================
# Config
# ======================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "data" / "issuer_events_cleaned_textual_data.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "issuer_events_with_topic_sentiment_qwen_gpu.csv"

EVENT_COLS = [
    "Analyst & Research Events",
    "Bankruptcy / Restructuring",
    "Capital-Markets Deals",
    "Corporate Actions",
    "Dividends",
    "ESG",
    "Guidance",
    "Index Membership",
    "Leadership",
    "Legal",
    "M&A",
    "Operations & Business Changes",
    "Product Events",
    "Regulatory",
    "Strategic & Financing Intent",
]

# Short English descriptions for readability
TOPIC_DISPLAY = {
    "Analyst & Research Events": "analyst and research related events",
    "Bankruptcy / Restructuring": "bankruptcy or restructuring events",
    "Capital-Markets Deals": "capital markets deals such as issuance and placements",
    "Corporate Actions": "corporate actions such as buybacks, splits, tender offers",
    "Dividends": "dividend and distribution related events",
    "ESG": "ESG (environment, social, governance) related events",
    "Guidance": "earnings guidance and outlook related events",
    "Index Membership": "index inclusion or deletion events",
    "Leadership": "management and leadership change events",
    "Legal": "lawsuits and legal dispute events",
    "M&A": "merger and acquisition related events",
    "Operations & Business Changes": "operational and business structure change events",
    "Product Events": "product launch or product change events",
    "Regulatory": "regulatory, approval and compliance events",
    "Strategic & Financing Intent": "strategic intentions and financing plans",
}

LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

MAX_CHARS_PER_TEXT = 6000
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCH = 0.5

# ======================
# Global model cache
# ======================

_LLM_PIPE = None
_TOKENIZER = None


def get_llm_pipeline():
    """
    Load Qwen text-generation pipeline once and cache it.
    If CUDA is available, force the model onto GPU.
    """
    global _LLM_PIPE, _TOKENIZER
    if _LLM_PIPE is not None:
        return _LLM_PIPE

    print(f">>> Loading LLM model: {LLM_MODEL_NAME} ...")
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print(">>> CUDA available: using GPU")
    else:
        print(">>> CUDA NOT available: using CPU (much slower)")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
    )

    if use_cuda:
        model.to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

    # device=0 -> GPU0, device=-1 -> CPU
    device_id = 0 if use_cuda else -1

    text_gen = hf_pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device_id,
    )

    _LLM_PIPE = text_gen
    _TOKENIZER = tokenizer
    print(">>> Model loaded.")
    return _LLM_PIPE

# ======================
# LLM helpers
# ======================


def build_prompt(topic: str, text: str) -> str:
    """
    Build the user prompt in ENGLISH, focusing on one topic only.
    """
    topic_desc = TOPIC_DISPLAY.get(topic, topic)
    prompt = f"""
You are a financial analyst. You are given a piece of news or aggregated text about a company.

Your task is to evaluate the sentiment of the company **only from the perspective of the following topic**:
Topic: {topic_desc}

Instructions:
1. Only consider information directly related to this topic. Ignore unrelated parts of the text.
2. Output exactly one real number between -1 and 1:
   - A value close to 1 means very positive for this topic.
   - A value close to -1 means very negative for this topic.
   - A value close to 0 means neutral or unclear for this topic.
3. Output ONLY the numeric value, with no words, no explanation.

News text:
{text}
"""
    return prompt.strip()


def call_llm_for_task(llm_pipe, topic: str, text: str) -> float:
    """
    Call Qwen for a single (topic, text) and return a float sentiment score.
    Uses chat template to make the model follow the instruction more strictly.
    """
    global _TOKENIZER

    if isinstance(text, str) and len(text) > MAX_CHARS_PER_TEXT:
        text = text[:MAX_CHARS_PER_TEXT]

    user_prompt = build_prompt(topic, text)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a sentiment scoring tool for financial news. "
                "For each request, you must output ONLY a single number between -1 and 1. "
                "Do not output any words or explanation."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    chat_text = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        out = llm_pipe(
            chat_text,
            max_new_tokens=16,
        )
        full_text = out[0]["generated_text"]
        generated_part = full_text[len(chat_text):].strip()

        # Extract first number in the output
        match = re.search(r"-?\d+(\.\d+)?", generated_part)
        if not match:
            print(
                f"[WARN] Could not parse numeric sentiment from model output, "
                f"topic={topic}, output={generated_part!r}"
            )
            # Fallback: neutral
            return 0.0

        score = float(match.group())
        score = max(min(score, 1.0), -1.0)
    except Exception as e:
        print(f"[WARN] LLM call failed, topic={topic}, error={e}")
        score = np.nan

    return score


def run_llm_in_batches(tasks: pd.DataFrame, llm_pipe) -> Dict[str, float]:
    """
    Run sentiment analysis for all unique (topic, text) tasks.
    """
    results: Dict[str, float] = {}

    total = len(tasks)
    n_batches = math.ceil(total / BATCH_SIZE)
    print(f"Total {total} (topic, text) tasks, {n_batches} batches ...")

    for i in tqdm(range(n_batches), desc="LLM batches"):
        start = i * BATCH_SIZE
        end = min((i + 1) * BATCH_SIZE, total)
        batch = tasks.iloc[start:end]

        for _, row in batch.iterrows():
            key = row["task_key"]
            topic = row["topic"]
            text = row["final_text"]
            score = call_llm_for_task(llm_pipe, topic, text)
            results[key] = score

        if SLEEP_BETWEEN_BATCH > 0:
            time.sleep(SLEEP_BETWEEN_BATCH)

    return results

# ======================
# Main pipeline
# ======================


def main():
    print(">>> USING QWEN (EN prompt) WITH GPU SUPPORT, NO OPENAI <<<")
    print(">>> Reading data ...")
    df = pd.read_csv(INPUT_CSV)

    if "dt" in df.columns:
        df["dt"] = pd.to_datetime(df["dt"])

    if "final_text" not in df.columns:
        raise ValueError("Column 'final_text' not found in input CSV.")

    # 1. Clean final_text
    print(">>> Cleaning final_text ...")
    df["final_text"] = df["final_text"].fillna("")
    df["final_len"] = df["final_text"].str.len()

    before = len(df)
    df_clean = df[df["final_len"] > 0].copy()
    after = len(df_clean)
    print(f"Total rows: {before}, removed empty final_text: {before - after}, kept: {after}")

    # Keep original index as row_id
    df_clean = df_clean.reset_index(drop=False).rename(columns={"index": "row_id"})

    # 2. Long table of (row_id, topic) where event flag == 1
    print(">>> Building long table of (row_id, topic) where event flag == 1 ...")
    cols_needed = ["row_id", "final_text"] + EVENT_COLS
    for c in EVENT_COLS:
        if c not in df_clean.columns:
            raise ValueError(f"Missing event column: {c}")

    long_df = df_clean[cols_needed].melt(
        id_vars=["row_id", "final_text"],
        value_vars=EVENT_COLS,
        var_name="topic",
        value_name="flag",
    )

    long_df = long_df[long_df["flag"] == 1].copy()
    long_df = long_df.drop(columns=["flag"])

    print(f"Number of (row_id, topic) pairs with flag=1: {len(long_df)}")

    if long_df.empty:
        print("No event=1 records, saving original data and exit.")
        out_df = df.drop(columns=["final_len"]) if "final_len" in df.columns else df
        out_df.to_csv(OUTPUT_CSV, index=False)
        return

    # 3. Deduplicate tasks by (topic, final_text)
    print(">>> Building unique LLM task list ...")
    long_df["task_key"] = long_df["topic"] + "||" + long_df["final_text"].astype(str)
    tasks = (
        long_df[["task_key", "topic", "final_text"]]
        .drop_duplicates("task_key")
        .reset_index(drop=True)
    )
    print(f"Unique tasks to call LLM on: {len(tasks)}")

    # 4. Run LLM
    llm_pipe = get_llm_pipeline()
    scores_dict = run_llm_in_batches(tasks, llm_pipe)

    # 5. Map scores back to long_df
    print(">>> Merging LLM scores back to long table ...")
    long_df["topic_sent"] = long_df["task_key"].map(scores_dict)

    # 6. Pivot to wide format: one *_sent column per topic
    print(">>> Pivoting to wide table with *_sent columns ...")
    wide_sent = (
        long_df
        .pivot(index="row_id", columns="topic", values="topic_sent")
        .reset_index()
    )

    new_cols = {"row_id": "row_id"}
    for topic in wide_sent.columns:
        if topic == "row_id":
            continue
        new_cols[topic] = f"{topic}_sent"
    wide_sent = wide_sent.rename(columns=new_cols)

    # 7. Merge back to df_clean
    print(">>> Merging sentiment columns back to cleaned DataFrame ...")
    df_with_sent = df_clean.merge(wide_sent, on="row_id", how="left")

    # 8. Merge back to original df (including rows with empty final_text)
    print(">>> Merging back to original DataFrame (all rows) ...")
    df_merged = df.merge(
        df_with_sent.drop(columns=[c for c in df_with_sent.columns if c in df.columns and c != "row_id"]),
        how="left",
        left_index=True,
        right_on="row_id",
    )

    if "row_id" in df_merged.columns:
        df_merged = df_merged.drop(columns=["row_id"])
    if "final_len" in df_merged.columns:
        df_merged = df_merged.drop(columns=["final_len"])

    print(f">>> Saving result to: {OUTPUT_CSV}")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    print(">>> DONE.")


if __name__ == "__main__":
    main()
