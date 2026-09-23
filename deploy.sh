#!/usr/bin/env bash
# Build and run the Orizon demo (frontend + backend) locally with Docker.
set -euo pipefail

cd "$(dirname "$0")"

# Load secrets (Bedrock token, etc.) so compose can pass them into the backend.
if [ -f .keys ]; then
  set -a; . ./.keys; set +a
else
  echo "Warning: .keys not found; Bedrock-backed synthetic generation will fail." >&2
fi

if docker compose version >/dev/null 2>&1; then
  DC="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  DC="docker-compose"
else
  echo "Docker Compose not found. Install Docker Desktop and try again." >&2
  exit 1
fi

echo "Building and starting containers..."
$DC up --build -d

echo
echo "Orizon is running:"
echo "  Dashboard : http://localhost:3000"
echo "  API       : http://localhost:8000  (docs at http://localhost:8000/docs)"
echo
echo "Logs : $DC logs -f"
echo "Stop : $DC down"
