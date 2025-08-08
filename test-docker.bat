@echo off
echo Building Docker image...
docker build -t bajajfin-app .

echo Running Docker container...
docker run -p 8000:8000 --env-file .env bajajfin-app

echo Docker container started. Test health endpoint: http://localhost:8000/health
pause
