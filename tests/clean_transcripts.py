import os
import re

def clean_transcript(text):
    """Clean transcript by removing noise and fixing formatting."""
    
    # Remove content inside brackets that are timestamps or annotations
    # Common patterns: [0:00], (0:00), music, etc.
    text = re.sub(r'\[\d+:\d+\]', '', text)
    text = re.sub(r'\(\d+:\d+\)', '', text)
    
    # Remove common filler words and noise patterns
    filler_patterns = [
        r'\bumm\b',
        r'\buh\b',
        r'\ber\b',
        r'\bhh\b',
        r'\bhmm\b',
        r'\bmm\b',
        r'\bhm\b',
        r'\[music\]',
        r'\[applause\]',
        r'\[laughter\]',
    ]
    for pattern in filler_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Fix sentence boundaries - add periods where needed
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Clean up any orphaned punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    # Fix common transcript errors
    text = re.sub(r'\bi\s', 'I ', text)  # Capitalize standalone 'i'
    
    # Remove remaining artifacts
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'www\.\S+', '', text)  # Remove www links
    
    return text.strip()


def segment_into_paragraphs(text, max_length=1000):
    """Split text into coherent paragraphs."""
    # First clean the text
    text = clean_transcript(text)
    
    # Split by sentence endings followed by capitalized words or clear breaks
    # This is a simple approach - in production you'd want more sophisticated logic
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    
    paragraphs = []
    current_para = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # Start new paragraph for major topic shifts
        # (simple heuristic based on keywords)
        topic_indicators = [
            'Now', 'First', 'But', 'So', 'Well', 'Moving',
            'In this', 'For example', 'What about', 'Let me',
            'The way', 'When', 'If', 'As', 'Because'
        ]
        
        if current_para and any(sentence.startswith(t) for t in topic_indicators):
            if current_length > 100:
                paragraphs.append(' '.join(current_para))
                current_para = [sentence]
                current_length = len(sentence)
            else:
                current_para.append(sentence)
                current_length += len(sentence)
        else:
            current_para.append(sentence)
            current_length += len(sentence)
    
    # Add final paragraph
    if current_para:
        paragraphs.append(' '.join(current_para))
    
    return paragraphs


# Process each transcript
transcripts = {
    'aircAruvnKk': '3Blue1Brown - Neural Networks',
    'wjZofJX0v4M': '3Blue1Brown - Transformers',
    'fHF22Wxuyw4': 'CampusX - Deep Learning (Hindi)',
    'C6YtPJxNULA': 'CodeWithHarry - ML & Deep Learning'
}

# Read and process each file
for filename, title in transcripts.items():
    input_path = f'temp_transcripts/{filename}.txt'
    
    if os.path.exists(input_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()
        
        # Clean and segment
        paragraphs = segment_into_paragraphs(raw_text)
        
        # Save cleaned version
        output_path = f'cleaned/{filename}_cleaned.txt'
        os.makedirs('cleaned', exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Source: {title}\n")
            f.write(f"Video ID: {filename}\n")
            f.write("=" * 50 + "\n\n")
            for para in paragraphs:
                if para.strip():
                    f.write(para.strip() + "\n\n")
        
        print(f"Processed: {filename} -> {len(paragraphs)} paragraphs")

print("\nAll transcripts cleaned and saved to 'cleaned/' folder")