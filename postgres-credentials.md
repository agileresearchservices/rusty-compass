# Postgres Docker Setup - Credentials & Endpoints

**Setup Date:** 2025-12-24

## Connection Details

| Property | Value |
|----------|-------|
| **Host** | localhost |
| **Port** | 5432 |
| **User** | postgres |
| **Password** | postgres |
| **Database** | postgres |

## Connection String

```
postgresql://postgres:postgres@localhost:5432/postgres
```

## Environment Variables

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=postgres
export POSTGRES_DB=postgres
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/postgres
```

## Docker Information

| Property | Value |
|----------|-------|
| **Container Name** | postgres-container |
| **Image** | postgres:16 |
| **Volume** | downloads_postgres_data |
| **Status** | Running |

## Connect to Postgres

### Using psql CLI

```bash
psql -h localhost -U postgres -d postgres
```

### Using Docker

```bash
docker compose exec postgres psql -U postgres -d postgres
```

## Notes

- The Postgres data is persisted in the Docker volume `downloads_postgres_data`
- The container is configured with a health check that verifies Postgres is ready
- Configuration file: `/Users/kevin/Downloads/docker-compose.yml`
