import os
from faster_whisper import WhisperModel

class SpeechToTextService:
    def __init__(self, model_size: str = "base.en", device: str = "cpu", compute_type: str = "int8"):
        """
        Offline Speech-to-Text engine using faster-whisper.
        'base.en' is fast, fully offline, and highly accurate for English speech.
        """
        print(f"Loading Whisper STT model ({model_size})...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Whisper model loaded successfully.")

    def transcribe(self, audio_path: str) -> str:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found at: {audio_path}")

        # تفريغ الصوت لنص إنجليزي
        segments, _ = self.model.transcribe(audio_path, language="en", beam_size=1)
        transcript = " ".join([segment.text.strip() for segment in segments])
        return transcript.strip()


if __name__ == "__main__":
    # اختبار تحميل الموديل
    stt = SpeechToTextService()
    print("STT Service ready for inference.")