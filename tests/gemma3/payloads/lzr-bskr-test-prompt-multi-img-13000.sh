time curl -X POST "http://localhost:8080/v1/chat/completions" \
  -H "Content-Type: application/json" \
  --data @safe_multipage_prompt_13000_tokens.json | jq
