#!/usr/bin/env python3
"""
Database Migration: Add Full-Text Search Support
Adds tsvector column and GIN index to document_chunks table
"""

import psycopg
from config import DATABASE_URL

def migrate_add_fulltext():
    """Add full-text search capabilities to document_chunks"""
    print("=" * 70)
    print("Database Migration: Adding Full-Text Search Support")
    print("=" * 70)
    print()

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Add generated tsvector column
                print("  → Adding content_tsv column...")
                cur.execute("""
                    ALTER TABLE document_chunks
                    ADD COLUMN IF NOT EXISTS content_tsv tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
                """)
                print("    ✓ Column added")

                # Create GIN index for fast text search
                print("  → Creating GIN index...")
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS document_chunks_content_tsv_idx
                    ON document_chunks USING GIN(content_tsv)
                """)
                print("    ✓ Index created")

                # Analyze table to update statistics
                print("  → Analyzing table...")
                cur.execute("ANALYZE document_chunks")
                print("    ✓ Table analyzed")

                # Verify migration
                print("  → Verifying migration...")
                cur.execute("""
                    SELECT COUNT(*) FROM document_chunks
                    WHERE content_tsv IS NOT NULL
                """)
                count = cur.fetchone()[0]

                # Check if index exists
                cur.execute("""
                    SELECT indexname FROM pg_indexes
                    WHERE tablename = 'document_chunks'
                    AND indexname = 'document_chunks_content_tsv_idx'
                """)
                index_exists = cur.fetchone() is not None

                print()
                print("=" * 70)
                if index_exists and count > 0:
                    print("✓ Migration successful!")
                    print(f"✓ {count} rows have tsvector data")
                    print("✓ GIN index created for fast full-text search")
                    print("=" * 70)
                    print()
                    print("Full-text search is now available. You can:")
                    print("  1. Use hybrid search with lambda_mult parameter")
                    print("  2. Test with: python test_hybrid_search.py")
                    print("  3. Update config: RETRIEVER_SEARCH_TYPE = 'hybrid'")
                    return 0
                else:
                    print("⚠ Migration may have issues:")
                    print(f"  - Rows with tsvector: {count}")
                    print(f"  - Index exists: {index_exists}")
                    print("=" * 70)
                    return 1

    except Exception as e:
        print()
        print("=" * 70)
        print(f"✗ Migration failed: {e}")
        print("=" * 70)
        print()
        print("Troubleshooting:")
        print("  1. Ensure PostgreSQL is running: docker compose ps")
        print("  2. Ensure database tables exist: python setup_db.py")
        print("  3. Check database connection in config.py")
        return 1


if __name__ == "__main__":
    exit(migrate_add_fulltext())
