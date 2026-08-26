import tiktoken

text = input("Enter your text: ")
encoding = tiktoken.get_encoding("cl100k_base")
tokens = encoding.encode(text)
token_count = len(tokens)
cost_per_token = 0.0015 
estimated_cost = (token_count / 1000) * cost_per_token

print(f"\nText: {token_count}")
print(f"Token count: {token_count}")
print(f"Estimated cost: ${estimated_cost:.8f}")

