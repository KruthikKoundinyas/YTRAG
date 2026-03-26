from youtube_transcript_api import YouTubeTranscriptApi

# Fetch transcript
transcript = YouTubeTranscriptApi().fetch('aircAruvnKk')

# Extract text from snippets using attribute access
texts = [snippet.text for snippet in transcript.snippets]

# Join into paragraphs
full_text = ' '.join(texts)

# Print the full text
print(full_text)