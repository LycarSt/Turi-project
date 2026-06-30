from youtube_transcript_api import YouTubeTranscriptApi

video_id = "2t_6SNWYdpI"

try:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=['es', 'en'])  # Ahora se usa .fetch()

    # Convertir a texto continuo
    texto_completo = " ".join([snippet.text for snippet in transcript])

    print(texto_completo)

    # Guardar a archivo
    with open(f"{video_id}_transcripcion.txt", "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"✅ Transcripción guardada en {video_id}_transcripcion.txt")

except Exception as e:
    print("❌ Error:", e)
