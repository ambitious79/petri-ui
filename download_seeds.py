"""
Download all succeeded seed instructions from Trishool API and save to MongoDB.

Fetches all target sets, then for each target set fetches all SUCCEEDED submissions
(paginating through results), embeds each seed instruction using OpenAI embeddings,
and saves to MongoDB. Skips any submissions with "Hidden" seed prompts.

Required environment variables:
    MONGODB_URI      - MongoDB connection string (e.g. mongodb+srv://user:pass@cluster.mongodb.net/)
    OPENAI_API_KEY   - OpenAI API key for computing embeddings

Optional environment variables:
    MONGODB_DB_NAME  - Database name (default: petri_ui)
    EMBEDDING_MODEL  - OpenAI embedding model (default: text-embedding-3-small)
    OPENAI_BASE_URL  - Custom OpenAI API base URL

Usage:
    python download_seeds.py
"""

import os
import sys
import time
import requests
from datetime import datetime
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================================
# Configuration
# ============================================================================
TRISHOOL_API_BASE = "https://api.trishool.ai/api/v1"
TARGET_SETS_URL = f"{TRISHOOL_API_BASE}/target-sets/list"
SUBMISSIONS_URL = f"{TRISHOOL_API_BASE}/submissions/"
SUBMISSIONS_PAGE_LIMIT = 100  # Maximum allowed by API

MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DB_NAME = os.environ.get('MONGODB_DB_NAME', 'petri_ui')
MONGODB_COLLECTION = 'seed_instructions'

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', None)
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'text-embedding-3-small')


def get_openai_client():
    """Initialize and return OpenAI client."""
    if not OPENAI_API_KEY:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    if OPENAI_BASE_URL:
        return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return OpenAI(api_key=OPENAI_API_KEY)


def compute_embedding(client, text):
    """Compute embedding vector for a given text using OpenAI API."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def get_mongo_collection():
    """Connect to MongoDB and return the seed_instructions collection."""
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client[MONGODB_DB_NAME]
    collection = db[MONGODB_COLLECTION]
    # Ensure unique index on instruction to avoid duplicates
    collection.create_index([("instruction", 1)], unique=True)
    return collection


def fetch_target_sets():
    """Fetch all target sets from the API."""
    print("[INFO] Fetching target sets...")
    response = requests.get(TARGET_SETS_URL)
    response.raise_for_status()
    data = response.json()
    target_sets = data.get('target_sets', [])
    print(f"[INFO] Found {len(target_sets)} target sets.")
    return target_sets


def fetch_all_submissions(target_id, target_description):
    """
    Fetch all SUCCEEDED submissions for a target set, paginating through results.
    Filters out Hidden seed prompts.
    """
    all_submissions = []
    page = 1

    while True:
        params = {
            'target_id': target_id,
            'submission_status': 'SUCCEEDED',
            'page': page,
            'limit': SUBMISSIONS_PAGE_LIMIT,
        }
        response = requests.get(SUBMISSIONS_URL, params=params)
        response.raise_for_status()
        data = response.json()

        submissions = data.get('submissions', [])
        total = data.get('total', 0)

        if not submissions:
            break

        # Filter out Hidden seed prompts
        non_hidden = [
            s for s in submissions
            if s.get('seed_prompt', '').strip().lower() != 'hidden'
        ]
        all_submissions.extend(non_hidden)

        hidden_count = len(submissions) - len(non_hidden)
        if hidden_count > 0:
            print(f"  [INFO] Page {page}: {len(non_hidden)} valid + {hidden_count} hidden (skipped)")
        else:
            print(f"  [INFO] Page {page}: {len(non_hidden)} submissions")

        # Check if we've fetched all
        fetched_so_far = page * SUBMISSIONS_PAGE_LIMIT
        if fetched_so_far >= total:
            break

        page += 1

    print(f"  [INFO] Total non-hidden submissions for '{target_description}': {len(all_submissions)}")
    return all_submissions


def save_to_mongodb(collection, openai_client, submissions, target_set_description):
    """
    Save submissions to MongoDB with embeddings.
    Skips duplicates (same instruction already in DB).
    """
    saved = 0
    skipped = 0
    errors = 0

    for i, sub in enumerate(submissions):
        instruction = sub.get('seed_prompt', '').strip()
        if not instruction:
            skipped += 1
            continue

        # Check if already exists
        existing = collection.find_one({"instruction": instruction})
        if existing:
            skipped += 1
            continue

        try:
            # Compute embedding
            embedding = compute_embedding(openai_client, instruction)

            # Build document
            doc = {
                "miner_hotkey": sub.get('miner_hotkey', ''),
                "instruction": instruction,
                "target_set": target_set_description,
                "created_at": sub.get('created_at', ''),
                "score": sub.get('mean_score', 0.0),
                "embedding": embedding,
            }

            collection.insert_one(doc)
            saved += 1

            if saved % 10 == 0:
                print(f"    [PROGRESS] Saved {saved} so far... (processing {i+1}/{len(submissions)})")

        except Exception as e:
            error_msg = str(e)
            # Duplicate key error means it was inserted between our check and insert
            if 'duplicate key' in error_msg.lower() or 'E11000' in error_msg:
                skipped += 1
            else:
                errors += 1
                print(f"    [ERROR] Failed to save submission {sub.get('id', '?')}: {error_msg}")

        # Small delay to avoid rate limiting on OpenAI API
        time.sleep(0.05)

    return saved, skipped, errors


def main():
    print("=" * 70)
    print("Trishool Seed Instruction Downloader")
    print("=" * 70)
    print(f"MongoDB URI: {MONGODB_URI[:30]}...")
    print(f"Database: {MONGODB_DB_NAME}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print("=" * 70)

    # Initialize clients
    openai_client = get_openai_client()
    collection = get_mongo_collection()

    # Get existing count
    existing_count = collection.count_documents({})
    print(f"[INFO] Existing instructions in database: {existing_count}")
    print()

    # Fetch all target sets
    target_sets = fetch_target_sets()

    total_saved = 0
    total_skipped = 0
    total_errors = 0

    for ts in target_sets:
        target_id = ts['id']
        description = ts.get('description', 'Unknown')
        status = ts.get('status', 'Unknown')

        print(f"\n{'─' * 60}")
        print(f"[TARGET SET] {description} (status: {status})")
        print(f"  ID: {target_id}")

        # Fetch all succeeded submissions
        submissions = fetch_all_submissions(target_id, description)

        if not submissions:
            print(f"  [INFO] No non-hidden submissions to save.")
            continue

        # Save to MongoDB with embeddings
        print(f"  [INFO] Saving {len(submissions)} submissions to database...")
        saved, skipped, errors = save_to_mongodb(
            collection, openai_client, submissions, description
        )

        total_saved += saved
        total_skipped += skipped
        total_errors += errors

        print(f"  [RESULT] Saved: {saved}, Skipped (duplicates): {skipped}, Errors: {errors}")

    # Final summary
    final_count = collection.count_documents({})
    print(f"\n{'=' * 70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'=' * 70}")
    print(f"Total saved (new):     {total_saved}")
    print(f"Total skipped (dupes): {total_skipped}")
    print(f"Total errors:          {total_errors}")
    print(f"Database count before: {existing_count}")
    print(f"Database count after:  {final_count}")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
