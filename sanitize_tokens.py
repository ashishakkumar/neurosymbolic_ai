import os
import re

target_dirs = ['z-setup/old', 'z-finetune', 'z-inference']
files = []

for target_dir in target_dirs:
    if os.path.exists(target_dir):
        for root, _, filenames in os.walk(target_dir):
            for filename in filenames:
                if filename.endswith('.ipynb'):
                    files.append(os.path.join(root, filename))

print(f"Found {len(files)} notebook files to process")

for file_path in files:
    try:
        print(f'Processing {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        if 'hf_' in content:
            # Replace HF tokens with placeholder
            new_content = re.sub(r'\"hf_[a-zA-Z0-9]+\"', '\"YOUR_HF_TOKEN\"', content)
            new_content = re.sub(r'hf_[a-zA-Z0-9]+', '\"YOUR_HF_TOKEN\"', new_content)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)
                print(f'  Updated {file_path}')
        else:
            print(f'  No tokens found in {file_path}')
    except Exception as e:
        print(f'  Error with {file_path}: {e}')

print("Token sanitization complete") 