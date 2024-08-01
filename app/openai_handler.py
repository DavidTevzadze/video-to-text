"""Модуль работы с OpenAI API"""
from openai import OpenAI

class OpenAIHandler:
    """Класс для работы с OpenAI API"""
    def __init__(self):
        self.client = OpenAI()

    def get_transcription(self, file_path: str) -> str:
        """Получаем транскрипцию аудиофайла"""
        audio_file = open(file_path, 'rb')
        transcription = self.client.audio.transcriptions.create(
            model='whisper-1',
            file=audio_file,
            response_format='text'
        )
        return transcription.text

    def get_completion(self, request: str) -> str:
        """Получаем ответ на сообщение пользователя"""
        completion = self.client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {'role': 'user', 'content': request}
            ]
        )
        if completion.choices[0].message.content is None:
            return ('No completion')
        return (completion.choices[0].message.content)
