# pip install youtube_transcript_api pyperclip

from youtube_transcript_api import YouTubeTranscriptApi
import pyperclip
import time

video_id = "2t_6SNWYdpI"
MAX_CHARS = 14000  # tamaño de cada parte

try:
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id, languages=['es', 'en'])
    texto_completo = " ".join([snippet.text for snippet in transcript])

    # Guardar transcripción completa en archivo
    with open(f"{video_id}_transcripcion.txt", "w", encoding="utf-8") as f:
        f.write(texto_completo)

    print(f"✅ Transcripción completa guardada en {video_id}_transcripcion.txt")

    # --- Dividir en partes ---
    partes = [texto_completo[i:i + MAX_CHARS] for i in range(0, len(texto_completo), MAX_CHARS)]

    total_partes = len(partes)

    # ... resto del código igual ...

    for idx, parte in enumerate(partes, start=1):
        texto_para_copiar = parte

        if idx == total_partes:
            texto_para_copiar += "\n\nLUEGO DE ESTA HAZ UN RESUMEN"
        else:
            texto_para_copiar += f"\n\nTODAVIA NO RESPONDAS, PARTE {idx}/{total_partes}"

        pyperclip.copy(texto_para_copiar)
        print(f"\n📌 Parte {idx}/{total_partes} preparada y COPIADA al portapapeles.")
        print("👉 Ahora ve a ChatGPT y pega (Ctrl+V) y envía.")

        if idx != total_partes:
            input("🔁 Presiona ENTER para copiar la siguiente parte...")
        else:
            input("🎉 Última parte copiada. ¡Todo listo! Presiona ENTER para salir.")  # <--- Aquí



except Exception as e:
    print("❌ Error:", e)
