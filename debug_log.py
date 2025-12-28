import os

filepath = 'chat_qa_log.txt'

print(f"--- DIAGNOSTIC START ---")
print(f"File exists: {os.path.exists(filepath)}")
try:
    with open(filepath, 'rb') as f:
        raw_head = f.read(200)
    print(f"RAW BYTES (Start): {raw_head}")
    
    print("\n--- TRYING UTF-8 READ ---")
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            print(f"Line {i}: {repr(line)}")
            
except Exception as e:
    print(f"ERROR: {e}")
print(f"--- DIAGNOSTIC END ---")
