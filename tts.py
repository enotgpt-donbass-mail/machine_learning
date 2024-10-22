import torch
import soundfile as sf
from num_to_rus import Converter
import requests
import json

from starlette.responses import FileResponse

conv = Converter()


def generate_audio(ticketNum, windowNum):

    language = 'ru'
    model_id = 'v4_ru'
    device = torch.device('cpu')

    model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                         model='silero_tts',
                                         language=language,
                                         speaker=model_id)
    model.to(device)  # gpu or cpu

    parts = ticketNum.split(' ')
    letter_part = parts[0]
    number_part = parts[1]
    number = int(number_part)

    sample_rate = 48000
    speaker = 'baya'
    put_accent = True
    put_yo = True

    example_text = f"""
              <speak>
              <p>Клиент с талоном <break time="500ms"/> {letter_part} <break time="200ms"/> {conv.convert(number)} <break time="500ms"/> подойд+ите к окн+у н+омер <break time="250ms"/> {conv.convert(windowNum)}.</p>
              </speak>
              """

    audio = model.apply_tts(ssml_text=example_text,
                            speaker=speaker,
                            sample_rate=sample_rate,
                            put_accent=put_accent,
                            put_yo=put_yo)

    audio_filename = f'ticket_{number}_window_{windowNum}.wav'
    sf.write(audio_filename, audio, sample_rate)

    url = 'https://s33.enotgpt.ru/upload/audio'
    headers = {
        'Accept': 'application/json',
        'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6NSwicm9sZXMiOlsiYWRtaW4iLCJ1c2VyIl0sImV4cCI6MTczOTgzNTkxMH0.Sf3bIxFPwNcuhUMIzqzI4D0sMXIOLsWF9NHpfmgJqT8'
    }

    # files = {
    #     'file': (audio_filename, open(audio_filename, 'rb'), 'audio/wav')
    # }

    #response = requests.post(url, headers=headers, files=files)
    return FileResponse(audio_filename)


# ticketNum = 'Эл 24'
# windowNum = 3
#
# audio_path = generate_audio(ticketNum, windowNum)
# print(f'Audio saved to: {audio_path}')