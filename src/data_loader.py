"""
Data loading and sampling utilities for the HotpotQA evaluation pipeline.

Functions
---------
load_hotpotqa(data_file_path)
    Load the HotpotQA JSON and build the document library.

sample_questions(data, config)
    Apply question-centric or document-centric sampling per config settings.
"""

import json
import random
from pathlib import Path
from difflib import SequenceMatcher

import numpy as np


# =============================================================================
# Load
# =============================================================================

def load_hotpotqa(data_file_path) -> tuple:
    """
    Load HotpotQA JSON and build the document library.

    Parameters
    ----------
    data_file_path : str or Path
        Path to hotpot_train_v1.1.json.

    Returns
    -------
    (data, document_library) : tuple[list, dict]
        data             — list of question dicts (question, answer, context, …)
        document_library — {doc_id: {title, content, sentences, doc_id}, …}
    """
    print("Loading HotpotQA data...")

    if not Path(data_file_path).exists():
        raise FileNotFoundError(
            f"\n\n{'='*65}\n"
            f"  HotpotQA data file not found:\n"
            f"    {data_file_path}\n\n"
            f"  To fix this:\n"
            f"  1. Download hotpot_train_v1.1.json (~540 MB) from:\n"
            f"       https://hotpotqa.github.io/\n"
            f"  2. Place it at the path above (create the folder if needed).\n"
            f"  3. Update HOTPOTQA_DATA_PATH in your .env if using a\n"
            f"     different location.\n"
            f"{'='*65}\n"
        )

    with open(data_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data)} questions\n")

    print("Building document library...")
    document_library = {}
    doc_id = 0

    for item in data:
        context = item.get('context', [])
        for doc_title, doc_sentences in context:
            full_doc = ' '.join(doc_sentences)
            document_library[doc_id] = {
                'title':     doc_title,
                'content':   full_doc,
                'sentences': doc_sentences,
                'doc_id':    doc_id,
            }
            doc_id += 1

    print(f"✓ Built library with {len(document_library):,} documents")
    print(f"  Average document length: "
          f"{np.mean([len(d['content']) for d in document_library.values()]):.0f} chars")

    return data, document_library


# =============================================================================
# Sampling helpers
# =============================================================================

def _fuzzy_match_title(query_title, available_titles, threshold=0.85):
    """Find the best matching title using exact → substring → fuzzy matching."""
    query_lower = query_title.lower().strip()
    best_match  = None
    best_score  = 0

    for title in available_titles:
        title_lower = title.lower().strip()

        if query_lower == title_lower:
            return title, 1.0

        if query_lower in title_lower or title_lower in query_lower:
            score = min(len(query_lower), len(title_lower)) / max(len(query_lower), len(title_lower))
            if score > best_score:
                best_score = score
                best_match = title

        similarity = SequenceMatcher(None, query_lower, title_lower).ratio()
        if similarity > best_score and similarity >= threshold:
            best_score = similarity
            best_match = title

    return best_match, best_score


# =============================================================================
# Sample
# =============================================================================

def sample_questions(data, config) -> list:
    """
    Apply question-centric (or full-dataset) sampling per config settings.

    Parameters
    ----------
    data : list
        Full HotpotQA question list (from load_hotpotqa).
    config : Config
        Framework config — reads USE_SAMPLING, NUM_EVAL_QUESTIONS, RANDOM_SEED.

    Returns
    -------
    list
        Sampled (or original) question list.

    Side effects
    ------------
    Prints a detailed sampling summary to stdout.
    """
    USE_SAMPLING  = config.USE_SAMPLING
    NUM_QUESTIONS = config.NUM_EVAL_QUESTIONS
    random.seed(config.RANDOM_SEED)

    # Build document library index (needed for title lookup)
    # We rebuild it here so the function is self-contained.
    doc_id_counter = 0
    document_library = {}
    for item in data:
        for doc_title, doc_sentences in item.get('context', []):
            full_doc = ' '.join(doc_sentences)
            document_library[doc_id_counter] = {
                'title':     doc_title,
                'content':   full_doc,
                'sentences': doc_sentences,
                'doc_id':    doc_id_counter,
            }
            doc_id_counter += 1

    print("=" * 80)
    print("DATASET SAMPLING (QUESTION-CENTRIC)")
    print("=" * 80)

    if not USE_SAMPLING:
        print("\n✓ Using full dataset")
        print(f"  Questions: {len(data):,}")
        print(f"  Documents: {len(document_library):,}\n")
        return list(data)

    print(f"\n📊 Question-centric sampling")
    print(f"   Target: {NUM_QUESTIONS:,} questions")
    print(f"   Source: {len(data):,} questions, {len(document_library):,} docs")

    # Step 1 — sample questions
    sampled_questions = random.sample(data, min(NUM_QUESTIONS, len(data)))
    print(f"\n✓ Sampled {len(sampled_questions):,} questions")

    # Step 2 — collect needed document titles
    needed_titles: set = set()
    for item in sampled_questions:
        for title, _ in item.get('context', []):
            needed_titles.add(title)

    print(f"✓ Sampled questions need {len(needed_titles):,} unique documents")
    print(f"   Expected: ~{len(sampled_questions) * 10:,} (10 docs/question)")

    if len(needed_titles) > len(sampled_questions) * 10:
        print(f"\n⚠️  WARNING: {len(needed_titles):,} titles for {len(sampled_questions):,} "
              f"questions is unusually high!")
        print(f"   Expected: ~{len(sampled_questions) * 10:,} titles")

    # Step 3 — build title index
    title_to_doc = {}
    available_titles = []
    for did, doc in document_library.items():
        title_to_doc[doc['title']] = (did, doc)
        available_titles.append(doc['title'])

    print(f"✓ Indexed {len(available_titles):,} available documents")

    # Step 4 — match needed titles
    print(f"\n🔍 Matching titles...")
    sampled_document_library = {}
    title_mapping  = {}
    found_count    = 0
    missing_titles = []
    fuzzy_matches  = []

    for needed_title in needed_titles:
        if needed_title in title_to_doc:
            did, doc = title_to_doc[needed_title]
            sampled_document_library[did] = doc
            title_mapping[needed_title]   = needed_title
            found_count += 1
        else:
            matched_title, score = _fuzzy_match_title(needed_title, available_titles)
            if matched_title and score >= 0.85:
                did, doc = title_to_doc[matched_title]
                sampled_document_library[did] = doc
                title_mapping[needed_title]   = matched_title
                found_count += 1
                if score < 1.0:
                    fuzzy_matches.append({'needed': needed_title, 'found': matched_title, 'score': score})
            else:
                missing_titles.append(needed_title)

    print(f"✓ Found {found_count:,} / {len(needed_titles):,} documents "
          f"({found_count / len(needed_titles) * 100:.1f}%)")
    if fuzzy_matches:
        print(f"   Fuzzy matches: {len(fuzzy_matches)}")
    if missing_titles:
        print(f"   ⚠️  Missing: {len(missing_titles):,} ({missing_titles[:3]})")

    # Step 5 — filter questions with complete context
    print(f"\n🎯 Filtering questions...")
    final_questions  = []
    questions_dropped = 0

    for item in sampled_questions:
        question_titles = {title for title, _ in item.get('context', [])}
        if all(title in title_mapping for title in question_titles):
            final_questions.append(item)
        else:
            questions_dropped += 1

    print(f"✓ Kept {len(final_questions):,} questions with complete context")
    if questions_dropped > 0:
        print(f"   Dropped {questions_dropped} questions with missing docs")

    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"  Questions:       {len(final_questions):,}")
    print(f"  Documents:       {len(sampled_document_library):,}")
    if final_questions:
        print(f"  Docs/Question:   {len(sampled_document_library) / len(final_questions):.1f}")
    print(f"  Match rate:      {found_count / len(needed_titles) * 100:.1f}%")
    print(f"  Seed:            {config.RANDOM_SEED}")
    print("=" * 80 + "\n")

    if len(sampled_document_library) > len(final_questions) * 10:
        print("⚠️  WARNING: Document count seems too high!")
        print(f"   {len(sampled_document_library):,} docs for {len(final_questions):,} questions")
        print(f"   Expected: ~{len(final_questions) * 3:,} docs (3 per question)")

    return final_questions
