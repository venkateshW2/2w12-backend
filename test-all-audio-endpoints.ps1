# Simple Audio Test Script - No Complex Syntax
Write-Host "🎵 2W12 Audio Analysis Test" -ForegroundColor Green
Write-Host "===========================" -ForegroundColor Green

# Test 1: Health Check
Write-Host "`n1. Health Check..." -ForegroundColor Yellow
curl http://localhost:8001/health

# Test 2: Basic Audio Analysis
Write-Host "`n2. Testing Basic Analysis with test48.wav..." -ForegroundColor Yellow
curl -X POST "http://localhost:8001/api/audio/analyze" -F "file=@C:\2w12-backend\test48.wav"

# Test 3: Advanced Analysis
Write-Host "`n3. Testing Advanced Analysis..." -ForegroundColor Yellow
curl -X POST "http://localhost:8001/api/audio/analyze-advanced" -F "file=@C:\2w12-backend\test48.wav"

# Test 4: Genre Classification
Write-Host "`n4. Testing Genre Classification..." -ForegroundColor Yellow
curl -X POST "http://localhost:8001/api/audio/classify-genre" -F "file=@C:\2w12-backend\test48.wav"

# Test 5: Mood Detection
Write-Host "`n5. Testing Mood Detection..." -ForegroundColor Yellow
curl -X POST "http://localhost:8001/api/audio/detect-mood" -F "file=@C:\2w12-backend\test48.wav"

# Test 6: Loudness Analysis
Write-Host "`n6. Testing Loudness Analysis..." -ForegroundColor Yellow
curl -X POST "http://localhost:8001/api/audio/loudness" -F "file=@C:\2w12-backend\test48.wav"

Write-Host "`n✅ All tests completed!" -ForegroundColor Green