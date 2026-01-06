"""
Checkpoint maintenance utilities for PostgresSaver.

Provides utilities for managing checkpoint storage:
- Compaction: Clean old checkpoint versions to reduce storage
- Cleanup: Remove checkpoints for deleted conversations

Usage:
    from checkpoint_maintenance import compact_checkpoints, cleanup_orphaned_checkpoints

    # Compact all threads, keeping last 3 versions
    deleted = compact_checkpoints(keep_versions=3)

    # Compact specific thread
    deleted = compact_checkpoints(thread_id="my-thread-id", keep_versions=3)

    # Remove orphaned checkpoints
    deleted = cleanup_orphaned_checkpoints()
"""

import psycopg
from datetime import datetime, timedelta
from typing import Optional

from config import DATABASE_URL, CHECKPOINT_KEEP_VERSIONS, CHECKPOINT_COMPACTION_DAYS


def compact_checkpoints(
    thread_id: Optional[str] = None,
    keep_versions: int = CHECKPOINT_KEEP_VERSIONS,
    older_than_days: int = CHECKPOINT_COMPACTION_DAYS,
) -> int:
    """
    Remove old checkpoint versions to reduce storage.

    Keeps the most recent `keep_versions` checkpoints per thread and
    removes older versions that are past `older_than_days`.

    Args:
        thread_id: Specific thread to compact (None = all threads)
        keep_versions: Number of recent versions to keep per thread
        older_than_days: Only compact checkpoints older than this

    Returns:
        Number of checkpoint blob rows deleted
    """
    cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            if thread_id:
                # Compact specific thread - delete old blobs not in recent checkpoints
                cur.execute("""
                    DELETE FROM checkpoint_blobs
                    WHERE thread_id = %s
                    AND checkpoint_id NOT IN (
                        SELECT checkpoint_id
                        FROM checkpoints
                        WHERE thread_id = %s
                        ORDER BY checkpoint_id DESC
                        LIMIT %s
                    )
                """, (thread_id, thread_id, keep_versions))
            else:
                # Compact all threads - more complex query to handle per-thread limits
                # First get list of threads, then compact each
                cur.execute("SELECT DISTINCT thread_id FROM checkpoints")
                threads = [row[0] for row in cur.fetchall()]

                total_deleted = 0
                for tid in threads:
                    cur.execute("""
                        DELETE FROM checkpoint_blobs
                        WHERE thread_id = %s
                        AND checkpoint_id NOT IN (
                            SELECT checkpoint_id
                            FROM checkpoints
                            WHERE thread_id = %s
                            ORDER BY checkpoint_id DESC
                            LIMIT %s
                        )
                    """, (tid, tid, keep_versions))
                    total_deleted += cur.rowcount

                conn.commit()
                return total_deleted

            deleted = cur.rowcount
            conn.commit()

    return deleted


def cleanup_orphaned_checkpoints() -> int:
    """
    Remove checkpoints for conversations that no longer exist in metadata.

    This cleans up checkpoints for deleted conversations where the
    conversation_metadata record has been removed but checkpoints remain.

    Returns:
        Number of orphaned checkpoint rows deleted
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Delete checkpoints where no matching metadata exists
            cur.execute("""
                DELETE FROM checkpoints c
                WHERE NOT EXISTS (
                    SELECT 1 FROM conversation_metadata m
                    WHERE m.thread_id = c.thread_id
                )
            """)
            deleted = cur.rowcount

            # Also clean up orphaned blobs
            cur.execute("""
                DELETE FROM checkpoint_blobs b
                WHERE NOT EXISTS (
                    SELECT 1 FROM checkpoints c
                    WHERE c.thread_id = b.thread_id
                    AND c.checkpoint_id = b.checkpoint_id
                )
            """)
            deleted += cur.rowcount

            conn.commit()

    return deleted


def get_checkpoint_stats() -> dict:
    """
    Get statistics about checkpoint storage.

    Returns:
        Dictionary with checkpoint statistics
    """
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Count threads
            cur.execute("SELECT COUNT(DISTINCT thread_id) FROM checkpoints")
            thread_count = cur.fetchone()[0]

            # Count checkpoints
            cur.execute("SELECT COUNT(*) FROM checkpoints")
            checkpoint_count = cur.fetchone()[0]

            # Count blobs
            cur.execute("SELECT COUNT(*) FROM checkpoint_blobs")
            blob_count = cur.fetchone()[0]

            # Estimate total size (if pg_size_pretty is available)
            try:
                cur.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size('checkpoint_blobs'))
                """)
                blob_size = cur.fetchone()[0]
            except Exception:
                blob_size = "unknown"

    return {
        "thread_count": thread_count,
        "checkpoint_count": checkpoint_count,
        "blob_count": blob_count,
        "estimated_size": blob_size,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checkpoint maintenance utilities")
    parser.add_argument("--stats", action="store_true", help="Show checkpoint statistics")
    parser.add_argument("--compact", action="store_true", help="Compact old checkpoints")
    parser.add_argument("--cleanup", action="store_true", help="Remove orphaned checkpoints")
    parser.add_argument("--thread-id", type=str, help="Specific thread to compact")
    parser.add_argument("--keep-versions", type=int, default=CHECKPOINT_KEEP_VERSIONS,
                        help=f"Versions to keep (default: {CHECKPOINT_KEEP_VERSIONS})")

    args = parser.parse_args()

    if args.stats:
        stats = get_checkpoint_stats()
        print("Checkpoint Statistics:")
        print(f"  Threads: {stats['thread_count']}")
        print(f"  Checkpoints: {stats['checkpoint_count']}")
        print(f"  Blobs: {stats['blob_count']}")
        print(f"  Estimated Size: {stats['estimated_size']}")

    if args.compact:
        deleted = compact_checkpoints(
            thread_id=args.thread_id,
            keep_versions=args.keep_versions
        )
        print(f"Compacted {deleted} checkpoint blob(s)")

    if args.cleanup:
        deleted = cleanup_orphaned_checkpoints()
        print(f"Cleaned up {deleted} orphaned checkpoint(s)")

    if not (args.stats or args.compact or args.cleanup):
        parser.print_help()
