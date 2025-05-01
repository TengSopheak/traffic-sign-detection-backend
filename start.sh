#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 8000

chmod +x main_service/start.sh