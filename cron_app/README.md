# Cron App Docker Setup

This Docker setup includes all necessary dependencies to run the cron_app, including PostgreSQL client libraries to resolve the libpq issue.

## Quick Start

### Prerequisites

- Docker and Docker Compose installed on your system

### Running the Application

1. **Build and start all services:**

   ```bash
   ./run.sh
   ```

   Or manually:

   ```bash
   docker-compose up --build -d
   ```

2. **View logs:**

   ```bash
   # View all logs
   docker-compose logs -f

   # View specific service logs
   docker-compose logs -f worker
   docker-compose logs -f beat
   docker-compose logs -f app
   ```

3. **Stop services:**

   ```bash
   docker-compose down
   ```

4. **Stop and remove volumes:**
   ```bash
   docker-compose down -v
   ```

## Services Included

- **PostgreSQL**: Database server (port 5432)
- **Redis**: Message broker for Celery (port 6379)
- **Celery Worker**: Processes background tasks
- **Celery Beat**: Task scheduler
- **Flask App**: Web application (port 5000)

## Environment Variables

The following environment variables are configured:

- `DATABASE_URL`: PostgreSQL connection string
- `CELERY_BROKER_URL`: Redis broker URL
- `CELERY_RESULT_BACKEND`: Redis result backend URL

## Database Setup

The PostgreSQL database will be automatically created with the following credentials:

- Database: `cmgmtDB`
- Username: `postgres`
- Password: `postgres`
- Host: `postgres` (within Docker network) or `localhost` (from host)
- Port: `5432`

## Troubleshooting

### libpq Library Issues

This Docker setup includes the PostgreSQL client libraries (`libpq-dev`) to resolve import errors with psycopg.

### Dependency Conflicts

The `requirements-docker.txt` file contains a curated list of dependencies without conflicts.

### Volume Mounts

- PostgreSQL data is persisted in a Docker volume
- Audio files are mounted to `./downloaded_audios`

## Development

To run in development mode with code reloading:

```bash
# Mount current directory for live code changes
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```
