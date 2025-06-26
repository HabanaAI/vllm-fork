time curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data @safe_multipage_prompt_90000_tokens.json | jq
