curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data @payload-27b-large.json | jq
