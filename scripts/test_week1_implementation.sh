#!/bin/bash
# NEW FILE: scripts/test_week1_implementation.sh

echo "🧪 Testing Week 1 Implementation"
echo "================================"

# Test 1: Redis connectivity
echo "1. Testing Redis connectivity..."
ssh Dell@192.168.1.16 "docker exec -it 2w12-redis redis-cli ping"

# Test 2: Enhanced health check
echo "2. Testing enhanced health check..."
curl -s http://192.168.1.16:8001/api/health/enhanced | jq '.overall_status'

# Test 3: Cache statistics (should be empty initially)
echo "3. Getting initial cache statistics..."
curl -s http://192.168.1.16:8001/api/stats/cache | jq '.cache_statistics.performance'

# Test 4: Enhanced analysis (first time - cache MISS)
echo "4. Testing enhanced analysis (cache MISS)..."
time curl -X POST "http://192.168.1.16:8001/api/audio/analyze-enhanced" \
     -F "file=@test48.wav" | jq '.analysis.cache_status'

# Test 5: Same file again (should be cache HIT)
echo "5. Testing same file again (cache HIT)..."
time curl -X POST "http://192.168.1.16:8001/api/audio/analyze-enhanced" \
     -F "file=@test48.wav" | jq '.analysis.cache_status'

# Test 6: Final cache statistics
echo "6. Final cache statistics..."
curl -s http://192.168.1.16:8001/api/stats/cache | jq '.cache_statistics.performance'

echo "✅ Week 1 testing completed!"