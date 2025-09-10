#!/usr/bin/env python3
"""
Программа для конвертации видео в текст с использованием OpenAI Whisper
"""

import os
import sys
from pathlib import Path
import whisper
from moviepy import VideoFileClip
from tqdm import tqdm
import argparse


class VideoToTextConverter:
    def __init__(self, model_name="base"):
        """
        Инициализация конвертера
        
        Args:
            model_name (str): Модель Whisper ("tiny", "base", "small", "medium", "large")
        """
        print(f"Загружаем модель Whisper: {model_name}...")
        
        # Проверяем локальную модель в папке whisper/
        local_model_path = Path("whisper") / f"{model_name}.pt"
        
        if local_model_path.exists():
            print(f"Найдена локальная модель: {local_model_path}")
            self.model = whisper.load_model(str(local_model_path))
        else:
            print(f"Локальная модель не найдена в {local_model_path}")
            print("Загружаем модель из интернета...")
            self.model = whisper.load_model(model_name)
            
        self.data_dir = Path("data")
        self.output_dir = Path("output")
        
        # Создаем папки если их нет
        self.data_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def extract_audio(self, video_path, audio_path):
        """
        Извлекает аудио из видео файла
        
        Args:
            video_path (Path): Путь к видео файлу
            audio_path (Path): Путь для сохранения аудио
        """
        print(f"Извлекаем аудио из {video_path.name}...")
        
        try:
            video = VideoFileClip(str(video_path))
            audio = video.audio
            audio.write_audiofile(str(audio_path), logger=None)
            audio.close()
            video.close()
            print(f"Аудио сохранено: {audio_path}")
        except Exception as e:
            print(f"Ошибка при извлечении аудио: {e}")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        Транскрибирует аудио в текст
        
        Args:
            audio_path (Path): Путь к аудио файлу
            
        Returns:
            str: Распознанный текст
        """
        print(f"Распознаем речь из {audio_path.name}...")
        
        try:
            result = self.model.transcribe(str(audio_path), language="ru")
            return result["text"]
        except Exception as e:
            print(f"Ошибка при распознавании речи: {e}")
            raise
    
    def save_text(self, text, output_path):
        """
        Сохраняет текст в файл
        
        Args:
            text (str): Текст для сохранения
            output_path (Path): Путь для сохранения файла
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            print(f"Текст сохранен: {output_path}")
        except Exception as e:
            print(f"Ошибка при сохранении файла: {e}")
            raise
    
    def process_video(self, video_path):
        """
        Обрабатывает одно видео: извлекает аудио, распознает речь, сохраняет текст
        
        Args:
            video_path (Path): Путь к видео файлу
        """
        print(f"\n=== Обработка видео: {video_path.name} ===")
        
        # Создаем имена для временного аудио файла и итогового текстового файла
        audio_path = self.output_dir / f"{video_path.stem}_temp.wav"
        text_path = self.output_dir / f"{video_path.stem}.txt"
        
        try:
            # Шаг 1: Извлекаем аудио
            self.extract_audio(video_path, audio_path)
            
            # Шаг 2: Распознаем речь
            text = self.transcribe_audio(audio_path)
            
            # Шаг 3: Сохраняем текст
            self.save_text(text, text_path)
            
            print(f"✅ Видео {video_path.name} успешно обработано!")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {video_path.name}: {e}")
            
        finally:
            # Удаляем временный аудио файл
            if audio_path.exists():
                audio_path.unlink()
                print(f"Временный файл {audio_path.name} удален")
    
    def process_all_videos(self):
        """
        Обрабатывает все видео файлы в папке data
        """
        # Поддерживаемые форматы видео
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        
        # Находим все видео файлы
        video_files = []
        for ext in video_extensions:
            video_files.extend(self.data_dir.glob(f"*{ext}"))
            video_files.extend(self.data_dir.glob(f"*{ext.upper()}"))
        
        if not video_files:
            print(f"❌ Видео файлы не найдены в папке {self.data_dir}")
            print(f"Поддерживаемые форматы: {', '.join(video_extensions)}")
            return
        
        print(f"Найдено видео файлов: {len(video_files)}")
        
        # Обрабатываем каждое видео
        for video_path in tqdm(video_files, desc="Обработка видео"):
            self.process_video(video_path)
        
        print(f"\n🎉 Обработка завершена! Результаты сохранены в папке {self.output_dir}")


def main():
    """Главная функция программы"""
    parser = argparse.ArgumentParser(description="Конвертация видео в текст с помощью Whisper")
    parser.add_argument(
        "--model", 
        default="base", 
        choices=["tiny", "base", "small", "medium", "large"],
        help="Модель Whisper для использования (по умолчанию: base)"
    )
    parser.add_argument(
        "--file",
        help="Обработать конкретный файл вместо всех файлов в папке data"
    )
    
    args = parser.parse_args()
    
    try:
        converter = VideoToTextConverter(model_name=args.model)
        
        if args.file:
            # Обрабатываем конкретный файл
            video_path = Path(args.file)
            if not video_path.exists():
                print(f"❌ Файл не найден: {video_path}")
                sys.exit(1)
            converter.process_video(video_path)
        else:
            # Обрабатываем все файлы в папке data
            converter.process_all_videos()
            
    except KeyboardInterrupt:
        print("\n⚠️ Обработка прервана пользователем")
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
