"""Sync worker: outbox JSONL → ChromaDB + reconcile định kỳ (plan §6 / B7).

Một lần chạy xử lý tối đa `--max-events` event sau cursor, ghi cursor, rồi
(chọn) reconcile từ source. Dùng cho cron hoặc --loop.

Usage:
    python scripts/sync_worker.py --outbox data/outbox.jsonl --raw-dir data/raw \
        --chroma data/chroma --embedder mock --once
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from cvjd.data.loader import load_raw_dir
from cvjd.models.embedding import create_embedder
from cvjd.storage import chroma
from cvjd.storage.outbox import Outbox, process_events, reconcile


def _resolve_with(raw_dir: Path):
    cvs, jds = load_raw_dir(raw_dir)
    lookup = {"cv": {d.doc_id: d for d in cvs}, "jd": {d.doc_id: d for d in jds}}

    def resolve(doc_type: str, doc_id: str):
        return lookup.get(doc_type, {}).get(doc_id)

    return resolve


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outbox", default=str(REPO_ROOT / "data" / "outbox.jsonl"))
    parser.add_argument("--raw-dir", default=str(REPO_ROOT / "data" / "raw"))
    parser.add_argument("--chroma", default=str(REPO_ROOT / "data" / "chroma"))
    parser.add_argument("--embedder", default="mock", choices=["mock", "st", "bge-m3"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-events", type=int, default=100)
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--once", action="store_true", help="chạy một lượt rồi thoát")
    parser.add_argument("--loop", action="store_true", help="chạy định kỳ 10s")
    parser.add_argument("--sleep-s", type=float, default=10.0)
    args = parser.parse_args()

    outbox = Outbox(args.outbox)
    embedder = create_embedder(args.embedder, device=args.device)
    model_id = getattr(embedder, "model_id", None) or args.embedder
    resolve_doc = _resolve_with(Path(args.raw_dir))

    client = chroma.build_client(args.chroma)
    chroma.ensure_collections(client)

    while True:
        start_cursor = outbox.load_cursor()
        events = outbox.read_since(start_cursor)
        if events:
            stats = process_events(
                client, events, embedder, model_id, resolve_doc,
                max_events=args.max_events,
            )
            outbox.ack(stats["cursor"])
            print(
                f"[worker] processed={stats['processed']} upserted={stats['upserted']} "
                f"deleted={stats['deleted']} skipped={stats['skipped']} "
                f"cursor={stats['cursor']}",
                flush=True,
            )
        if args.reconcile:
            cvs, jds = load_raw_dir(Path(args.raw_dir))
            source = {"cv": {d.doc_id: d for d in cvs}, "jd": {d.doc_id: d for d in jds}}
            rep = reconcile(client, source, embedder, model_id)
            print(
                f"[reconcile] upserted={rep['upserted']} deleted={rep['deleted']} "
                f"missing={rep['missing']} refreshed={rep['refreshed']}",
                flush=True,
            )
        if args.once:
            break
        if not args.loop:
            break
        time.sleep(args.sleep_s)

    if hasattr(client, "close"):
        try:
            client.close()
        except Exception:  # noqa: BLE001, S110
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())