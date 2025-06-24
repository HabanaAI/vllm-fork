time curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data @payload-multi-caleb.json | jq
