from youtube_transcript_api import YouTubeTranscriptApi
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    # Handle various YouTube URL formats
    patterns = [
        r'(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})',
        r'(?:youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Full URL links array
video_links = [
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://www.youtube.com/watch?v=wjZofJX0v4M",
    "https://www.youtube.com/watch?v=fHF22Wxuyw4",
    "https://www.youtube.com/watch?v=C6YtPJxNULA"
]

# Video ID to language mapping
video_languages = {
    "aircAruvnKk": ["en"],
    "wjZofJX0v4M": ["en"],
    "fHF22Wxuyw4": ["hi"],  # Hindi video
    "C6YtPJxNULA": ["en"]
}

# Create temp folder structure
temp_base = "temp_transcripts"
os.makedirs(temp_base, exist_ok=True)

def fetch_single_transcript(video_url):
    """Fetch transcript for a single video."""
    video_id = extract_video_id(video_url)
    
    if not video_id:
        print(f"  ERROR: Could not extract video ID from {video_url}")
        return None
    
    languages = video_languages.get(video_id, ["en"])
    print(f"Fetching: {video_id} (trying: {languages})")
    
    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
        
        # Extract text from snippets
        texts = [snippet.text for snippet in transcript.snippets]
        
        # Create video-specific subfolder
        video_folder = os.path.join(temp_base, video_id)
        os.makedirs(video_folder, exist_ok=True)
        
        # Save raw transcript to file in video folder
        filename = os.path.join(video_folder, "raw.txt")
        full_text = ' '.join(texts)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(full_text)
        
        print(f"  SUCCESS: {video_id} -> {filename} ({len(texts)} snippets)")
        return {
            'video_id': video_id,
            'filename': filename,
            'snippet_count': len(texts)
        }
        
    except Exception as e:
        print(f"  ERROR fetching {video_id}: {e}")
        return None

# Fetch all transcripts in parallel
print(f"Fetching {len(video_links)} transcripts in parallel...\n")

results = []
with ThreadPoolExecutor(max_workers=4) as executor:
    # Submit all fetch tasks
    future_to_url = {
        executor.submit(fetch_single_transcript, url): url 
        for url in video_links
    }
    
    # Collect results as they complete
    for future in as_completed(future_to_url):
        result = future.result()
        if result:
            results.append(result)

# Summary
print("\n" + "=" * 50)
print(f"Fetched {len(results)}/{len(video_links)} transcripts successfully!")
print(f"Files saved in: {temp_base}/")
print("=" * 50)

# List all saved files
print("\nSaved files:")
for root, dirs, files in os.walk(temp_base):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath)
        rel_path = os.path.relpath(filepath, temp_base)
        print(f"  - {rel_path} ({size} bytes)")
