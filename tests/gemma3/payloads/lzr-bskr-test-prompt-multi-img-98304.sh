time curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data @safe_multipage_prompt_98304_tokens.json | jq
