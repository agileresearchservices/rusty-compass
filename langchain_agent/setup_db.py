#!/usr/bin/env python3
"""
Database Setup Script
Initializes the Postgres database and tables for LangChain Agent
"""

import psycopg
from psycopg import sql
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver
from config import (
    DATABASE_URL,
    DB_CONNECTION_KWARGS,
    DB_POOL_MAX_SIZE,
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    POSTGRES_DB,
)


def create_database():
    """Create the langchain_agent database if it doesn't exist"""
    print("Creating database if needed...")

    try:
        # Connect to the default postgres database to create our database
        admin_conn_string = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/postgres"

        with psycopg.connect(admin_conn_string) as conn:
            # Enable autocommit for CREATE DATABASE
            conn.autocommit = True
            with conn.cursor() as cur:
                # Check if database exists
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (POSTGRES_DB,)
                )
                if not cur.fetchone():
                    cur.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(POSTGRES_DB)))
                    print(f"✓ Database '{POSTGRES_DB}' created successfully")
                else:
                    print(f"✓ Database '{POSTGRES_DB}' already exists")
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        raise


def init_checkpoint_tables():
    """Initialize the PostgresSaver checkpoint tables"""
    print("Initializing checkpoint tables...")

    try:
        # Create connection pool
        connection_kwargs = DB_CONNECTION_KWARGS.copy()
        pool = ConnectionPool(
            conninfo=DATABASE_URL,
            max_size=DB_POOL_MAX_SIZE,
            kwargs=connection_kwargs
        )

        # Initialize the checkpointer (creates tables if they don't exist)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        print("✓ Checkpoint tables initialized successfully")

        # Close the pool
        pool.close()

    except Exception as e:
        print(f"✗ Error initializing checkpoint tables: {e}")
        raise


def init_metadata_table():
    """Initialize conversation metadata table"""
    print("Initializing conversation metadata table...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                # Create conversation metadata table if it doesn't exist
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_metadata (
                        thread_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                print("✓ Conversation metadata table initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing metadata table: {e}")
        raise


def verify_connection():
    """Verify connection to the database"""
    print("Verifying database connection...")

    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT version()")
                version = cur.fetchone()[0]
                print(f"✓ Connected to: {version.split(',')[0]}")
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        raise


def main():
    """Run all setup steps"""
    print("=" * 60)
    print("LangChain Agent - Database Setup")
    print("=" * 60)
    print()

    try:
        create_database()
        print()
        verify_connection()
        print()
        init_checkpoint_tables()
        print()
        init_metadata_table()
        print()
        print("=" * 60)
        print("✓ Database setup complete!")
        print("=" * 60)
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ Setup failed: {e}")
        print("=" * 60)
        print("\nTroubleshooting:")
        print("1. Ensure Postgres is running: docker compose ps")
        print("2. Check credentials in config.py")
        print("3. Verify Postgres is accessible: psql -U postgres -h localhost")
        exit(1)


if __name__ == "__main__":
    main()
