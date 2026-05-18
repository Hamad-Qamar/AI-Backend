"""Build comprehensive enhanced KB with query variations"""
import json
import re

# Load the original KB
with open('buildhive_knowledge_base.json', 'r', encoding='utf-8') as f:
    original_kb = json.load(f)

print(f"Loaded {len(original_kb)} Q&A pairs from original KB")

def generate_variations(question, tags):
    """Generate query variations for a given question"""
    variations = [question]  
    q_lower = question.lower()
    
    # Pattern-based variations
    if "how do i" in q_lower:
        match = re.search(r"how do i (.+)\?", q_lower)
        if match:
            action = match.group(1)
            variations.extend([
                f"Steps to {action}",
                f"What are the steps to {action}?",
                f"Guide to {action}",
                f"How to {action}?",
                f"Can I {action}?",
                f"Process for {action}",
            ])
    
    if "what " in q_lower:
        if " is " in q_lower:
            parts = q_lower.split(" is ")
            if len(parts) == 2:
                subj = parts[1].rstrip("?").strip()
                variations.extend([
                    f"Tell me about {subj}",
                    f"Explain {subj}",
                    f"Describe {subj}",
                ])
        elif " are " in q_lower:
            variations.append(question.replace("What are", "Tell me about"))
    
    if "can i" in q_lower:
        match = re.search(r"can i (.+)\?", q_lower)
        if match:
            action = match.group(1)
            variations.extend([
                f"How to {action}?",
                f"Is it possible to {action}?",
            ])
    
    # Add tag-based variations
    for tag in tags[:3]:  
        variations.extend([
            f"Tell me about {tag}",
            f"{tag} information"
        ])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for v in variations:
        v_lower = v.lower().strip()
        if v_lower not in seen and len(unique_variations) < 12:
            seen.add(v_lower)
            unique_variations.append(v_lower)
    
    return unique_variations

# Build enhanced KB
enhanced_kb = []
for item in original_kb:
    variations = generate_variations(
        item.get("question", ""),
        item.get("tags", [])
    )
    
    enhanced_item = {
        "category": item.get("category", ""),
        "question": item.get("question", ""),
        "answer": item.get("answer", ""),
        "tags": item.get("tags", []),
        "query_variations": variations
    }
    enhanced_kb.append(enhanced_item)

# Save
with open('buildhive_knowledge_base_enhanced.json', 'w', encoding='utf-8') as f:
    json.dump(enhanced_kb, f, indent=2, ensure_ascii=False)

print(f"✓ Enhanced KB created: {len(enhanced_kb)} Q&A pairs")
total_vars = sum(len(item['query_variations']) for item in enhanced_kb)
print(f"✓ Total query variations: {total_vars}")
print(f"✓ Average per Q&A: {total_vars/len(enhanced_kb):.1f}")
print("✓ Ready for 90%+ efficiency!")
