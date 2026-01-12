"""
Create a comprehensive JSON file with metadata + actual chunk content
"""

import pickle
import json

# Load chunks and metadata
chunks = pickle.load(open('uk_visa_db/chunks.pkl', 'rb'))
metadata = json.load(open('uk_visa_db/metadata.json', 'r'))

# Combine chunks with their metadata
full_data = []
for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
    combined = {
        "id": meta["id"],
        "visa_type": meta["visa_type"],
        "section": meta["section"],
        "process": meta["process"],
        "source": meta["source"],
        "page": meta["page"],
        "char_start": meta["char_start"],
        "char_end": meta["char_end"],
        "word_count": meta["word_count"],
        "content": chunk  # Add the actual chunk text here
    }
    full_data.append(combined)

# Save to JSON file
with open('uk_visa_db/metadata_with_content.json', 'w') as f:
    json.dump(full_data, f, indent=2)

print(f"Created metadata_with_content.json with {len(full_data)} chunks")
print(f"File saved to: uk_visa_db/metadata_with_content.json")
