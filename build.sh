#!/usr/bin/env bash
# exit on error
set -o errexit

echo "==> Installing Python dependencies..."
pip install -r requirements.txt

echo "==> Changing to frontend directory..."
cd frontend

echo "==> Installing frontend dependencies..."
npm install

echo "==> Building frontend for production..."
npm run build

echo "==> Returning to root directory..."
cd ..