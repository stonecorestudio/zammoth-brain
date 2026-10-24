"""Export the knowledge book for the control plane (2026-08-08).

The book is served, not shipped (see knowledge.py): the backend answers
``GET /v1/knowledge`` from a JSON this command generates. Regenerate + redeploy
whenever the book changes:

    python -m orchestrator.export_knowledge > backend/app/knowledge.json
    (cd backend && fly deploy)

Run from the PRIVATE repo — this imports the private content module on purpose.
"""
from __future__ import annotations

import hashlib
import json


def payload() -> dict:
    from .knowledge_content import DEEP_TOPICS, KNOWLEDGE_BLOCK

    topics = {name: [summary, body]
              for name, (summary, body) in sorted(DEEP_TOPICS.items())}
    version = hashlib.sha256(
        (KNOWLEDGE_BLOCK + json.dumps(topics, sort_keys=True)).encode("utf-8")
    ).hexdigest()[:16]
    return {"version": version, "block": KNOWLEDGE_BLOCK, "topics": topics}


def main() -> int:
    print(json.dumps(payload(), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
