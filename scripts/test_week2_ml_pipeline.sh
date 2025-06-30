#!/bin/bash
# NEW FILE: scripts/test_week2_ml_pipeline.sh

echo "🧪 Testing Week 2 ML Pipeline"
echo "============================="

# Test 1: Enhanced health check with ML models
echo "1. Testing ML model availability..."
curl -s http://192.168.1.16:8001/api/health/enhanced | jq '.details.ml_models_loaded'

# Test 2: Enhanced analysis with full ML pipeline
echo "2. Testing full ML analysis pipeline..."
time curl -X POST "http://192.168.1.16:8001/api/audio/analyze-enhanced" \
     -F "file=@test_audio.wav" > week2_test_result.json

# Check ML features in result
echo "3. Checking ML features..."
jq '.analysis | keys | map(select(test("^ml_|^madmom_")))' week2_test_result.json

# Test 4: Essentia key detection
echo "4. Testing Essentia key detection..."
jq '.analysis.ml_key // "not_available"' week2_test_result.json

# Test 5: Madmom tempo detection
echo "5. Testing Madmom tempo detection..."
jq '.analysis.madmom_tempo // "not_available"' week2_test_result.json

# Test 6: Background research status
echo "6. Checking background research..."
jq '.analysis.background_research_started // false' week2_test_result.json

# Test 7: Cache performance
echo "7. Final cache performance..."
curl -s http://192.168.1.16:8001/api/stats/cache | jq '.cache_statistics.performance'

echo "✅ Week 2 ML pipeline testing completed!"