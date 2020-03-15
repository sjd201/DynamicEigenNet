from youtube_transcript_api import YouTubeTranscriptApi
f = YouTubeTranscriptApi.get_transcript("glkQwKA5_PU")
print("\n".join("{}".format(x["text"] if x["text"] != "[Music]" else "_ _ _ _ _ _ _") for x in f))

