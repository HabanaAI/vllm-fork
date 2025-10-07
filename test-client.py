from transformers import AutoTokenizer

model_name = "OpenGVLab/InternVL3-14B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Print the chat template
print("=== Chat Template ===")
print(tokenizer.chat_template)
print("=====================")

# Test applying the template with a simple string message (this should work)
messages_str = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
rendered_str = tokenizer.apply_chat_template(messages_str, tokenize=False)
print("\n=== Rendered with STRING content ===")
print(rendered_str)

# Test applying the template with a LIST of content parts (this will likely fail)
messages_list = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Hello!"},
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
        ]
    }
]

try:
    rendered_list = tokenizer.apply_chat_template(messages_list, tokenize=False)
    print("\n=== Rendered with LIST content ===")
    print(rendered_list)
except Exception as e:
    print("\n=== ERROR when using LIST content ===")
    print(f"Error: {type(e).__name__}: {e}")