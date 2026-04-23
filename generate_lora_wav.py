#!/usr/bin/env python3
"""
Генератор одиночного WAV-файла с сигналом LoRa (CSS-модуляция)
Оптимизировано для анализа спектра в Adobe Audition

Использование:
    python generate_lora_wav.py
    
Или с параметрами:
    python generate_lora_wav.py --payload "HELLO WORLD" --sf 9 --output my_lora_signal.wav
"""

import numpy as np
import wave
import struct
import argparse
from datetime import datetime


def generate_chirp(n_samples, fs, f_start, f_end, chirp_type='up'):
    """
    Генерация линейного chirp-сигнала с корректной фазой
    
    Параметры:
        n_samples: количество сэмплов
        fs: частота дискретизации
        f_start: начальная частота
        f_end: конечная частота
        chirp_type: 'up' или 'down'
    
    Возвращает:
        массив сэмплов сигнала
    """
    t = np.arange(n_samples) / fs
    T = n_samples / fs
    
    if chirp_type == 'up':
        # Upchirp: частота растет от f_start до f_end
        # Фаза: phi(t) = 2*pi * (f_start*t + 0.5*k*t^2)
        chirp_rate = (f_end - f_start) / T
        phase = 2 * np.pi * (f_start * t + 0.5 * chirp_rate * t**2)
    else:
        # Downchirp: частота падает от f_end до f_start
        # Фаза: phi(t) = 2*pi * (f_end*t - 0.5*k*t^2)
        chirp_rate = (f_end - f_start) / T
        phase = 2 * np.pi * (f_end * t - 0.5 * chirp_rate * t**2)
    
    # Применяем окно Hann для предотвращения спектральных утечек
    window = np.hanning(n_samples)
    
    # Генерируем сигнал
    signal = np.cos(phase) * window
    
    return signal


def create_shifted_chirp(base_chirp, symbol_value, n_samples):
    """
    Создание символа данных путем циклического сдвига базового upchirp
    
    Параметры:
        base_chirp: базовый upchirp
        symbol_value: значение символа (0..2^SF-1)
        n_samples: количество сэмплов в символе
    
    Возвращает:
        сдвинутый chirp
    """
    # Циклический сдвиг во временной области
    shift_samples = int(symbol_value * n_samples / (2 ** len(bin(n_samples)) - 2))
    shifted = np.roll(base_chirp, shift_samples)
    return shifted


def generate_lora_signal(payload, sf=7, bw=125000, center_freq=4000, fs=48000):
    """
    Генерация полного LoRa кадра
    
    Параметры:
        payload: строка или байты для передачи
        sf: Spreading Factor (7-12)
        bw: полоса пропускания (125000, 250000, 500000)
        center_freq: центральная частота в аудио-диапазоне (Гц)
        fs: частота дискретизации (Гц)
    
    Возвращает:
        массив сэмплов сигнала
    """
    # Расчет длительности символа
    t_sym = (2 ** sf) / bw
    
    # Количество сэмплов на символ (округляем до целого для точности)
    n_samples_per_symbol = int(round(t_sym * fs))
    
    # Полоса частот в аудио-диапазоне (масштабируем)
    audio_bw = min(bw / 1000, center_freq * 0.8)  # Ограничиваем чтобы не выйти за пределы
    
    # Начальная и конечная частоты чирпа
    f_start = center_freq - audio_bw / 2
    f_end = center_freq + audio_bw / 2
    
    print(f"Параметры генерации:")
    print(f"  Spreading Factor (SF): {sf}")
    print(f"  Bandwidth (BW): {bw} Гц")
    print(f"  Частота дискретизации: {fs} Гц")
    print(f"  Центральная частота: {center_freq} Гц")
    print(f"  Аудио полоса: {audio_bw:.1f} Гц ({f_start:.1f} - {f_end:.1f} Гц)")
    print(f"  Длительность символа: {t_sym*1000:.2f} мс")
    print(f"  Сэмплов на символ: {n_samples_per_symbol}")
    
    # Преобразуем payload в байты
    if isinstance(payload, str):
        payload_bytes = payload.encode('utf-8')
    else:
        payload_bytes = payload
    
    # Создаем базовые чирпы
    print("\nГенерация базовых чирпов...")
    base_upchirp = generate_chirp(n_samples_per_symbol, fs, f_start, f_end, 'up')
    base_downchirp = generate_chirp(n_samples_per_symbol, fs, f_start, f_end, 'down')
    
    all_samples = []
    
    # 1. Преамбула (8 upchirp)
    print("Добавление преамбулы (8 символов)...")
    for i in range(8):
        all_samples.append(base_upchirp.copy())
    
    # 2. Сетевые синхросигналы (2 downchirp)
    print("Добавление синхросигналов (2 символа)...")
    for i in range(2):
        all_samples.append(base_downchirp.copy())
    
    # 3. Заголовок (упрощенный, 1 символ)
    print("Добавление заголовка...")
    header_byte = len(payload_bytes)
    header_chirp = create_shifted_chirp(base_upchirp, header_byte % (2**sf), n_samples_per_symbol)
    all_samples.append(header_chirp)
    
    # 4. Полезные данные
    print(f"Кодирование полезной нагрузки ({len(payload_bytes)} байт)...")
    for byte_idx, byte_val in enumerate(payload_bytes):
        symbol_value = byte_val % (2**sf)
        shifted_chirp = create_shifted_chirp(base_upchirp, symbol_value, n_samples_per_symbol)
        all_samples.append(shifted_chirp)
        
        # Для наглядности добавим небольшой идентификатор позиции
        if byte_idx < 10:
            print(f"  Байт {byte_idx}: 0x{byte_val:02X} -> символ {symbol_value}")
    
    # 5. CRC (2 downchirp)
    print("Добавление CRC (2 символа)...")
    for i in range(2):
        all_samples.append(base_downchirp.copy())
    
    # Объединяем все части
    print("\nСборка полного сигнала...")
    full_signal = np.concatenate(all_samples)
    
    # Нормализация амплитуды на максимум для лучшей видимости в спектре
    max_amplitude = np.max(np.abs(full_signal))
    if max_amplitude > 0:
        full_signal = full_signal / max_amplitude  # Максимальная амплитуда без клиппинга
    
    total_duration = len(full_signal) / fs
    print(f"\nРезультат:")
    print(f"  Всего сэмплов: {len(full_signal)}")
    print(f"  Длительность: {total_duration:.3f} с")
    print(f"  Символов всего: {len(all_samples)}")
    
    return full_signal, fs


def save_wav(signal, fs, output_file, bits_per_sample=16):
    """
    Сохранение сигнала в WAV-файл
    
    Параметры:
        signal: массив сэмплов (float от -1 до 1)
        fs: частота дискретизации
        output_file: имя выходного файла
        bits_per_sample: разрядность (16, 24, 32)
    """
    print(f"\nСохранение в {output_file}...")
    
    # Конвертируем в нужный формат
    if bits_per_sample == 16:
        samples = (signal * 32767).astype(np.int16)
        sample_width = 2
    elif bits_per_sample == 24:
        samples = (signal * 8388607).astype(np.int32)
        sample_width = 3
    elif bits_per_sample == 32:
        samples = (signal * 2147483647).astype(np.int32)
        sample_width = 4
    else:
        raise ValueError("bits_per_sample должен быть 16, 24 или 32")
    
    # Записываем WAV-файл
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Моно
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(fs)
        
        # Для 24-bit нужно особая обработка
        if bits_per_sample == 24:
            # Конвертируем в байты вручную
            byte_data = bytearray()
            for sample in samples:
                byte_data.extend(sample.to_bytes(3, byteorder='little', signed=True))
            wav_file.writeframes(bytes(byte_data))
        else:
            wav_file.writeframes(samples.tobytes())
    
    file_size_mb = len(samples) * sample_width / (1024 * 1024)
    print(f"  Файл сохранен: {output_file}")
    print(f"  Размер: {file_size_mb:.2f} МБ")
    print(f"  Разрядность: {bits_per_sample} бит")
    print(f"  Каналы: 1 (моно)")
    print(f"  Частота дискретизации: {fs} Гц")


def main():
    parser = argparse.ArgumentParser(
        description='Генератор WAV-файла с сигналом LoRa для Adobe Audition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s                              # Создать файл по умолчанию
  %(prog)s --payload "HELLO"            # Свой текст
  %(prog)s --sf 9 --payload "TEST123"   # SF=9
  %(prog)s --output custom.wav          # Своё имя файла
  %(prog)s --sf 12 --bw 250000          # Максимальная дальность
        """
    )
    
    parser.add_argument('--payload', type=str, default='LORA TEST SIGNAL',
                        help='Текст для кодирования (по умолчанию: "LORA TEST SIGNAL")')
    parser.add_argument('--sf', type=int, default=9, choices=[7, 8, 9, 10, 11, 12],
                        help='Spreading Factor 7-12 (по умолчанию: 9)')
    parser.add_argument('--bw', type=int, default=125000, choices=[125000, 250000, 500000],
                        help='Bandwidth в Гц (по умолчанию: 125000)')
    parser.add_argument('--center-freq', type=float, default=4000,
                        help='Центральная частота в Гц (по умолчанию: 4000)')
    parser.add_argument('--fs', type=int, default=48000,
                        help='Частота дискретизации в Гц (по умолчанию: 48000)')
    parser.add_argument('--output', '-o', type=str, default='lora_signal.wav',
                        help='Имя выходного WAV-файла (по умолчанию: lora_signal.wav)')
    parser.add_argument('--bits', type=int, default=16, choices=[16, 24, 32],
                        help='Разрядность в битах (по умолчанию: 16)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Генератор LoRa CSS-сигнала для Adobe Audition")
    print("=" * 60)
    print(f"\nПолезная нагрузка: \"{args.payload}\"")
    print(f"Длина сообщения: {len(args.payload)} символов")
    
    # Генерируем сигнал
    signal, fs = generate_lora_signal(
        payload=args.payload,
        sf=args.sf,
        bw=args.bw,
        center_freq=args.center_freq,
        fs=args.fs
    )
    
    # Сохраняем в WAV
    save_wav(signal, fs, args.output, bits_per_sample=args.bits)
    
    # Выводим рекомендации для Adobe Audition
    print("\n" + "=" * 60)
    print("Рекомендации для анализа в Adobe Audition:")
    print("=" * 60)
    print("1. Откройте файл: Файл → Открыть → выберите", args.output)
    print("2. Переключитесь на вид спектрограммы: Вид → Показать частотный анализ")
    print("3. Рекомендуемые настройки спектрограммы:")
    print("   - Размер БПФ (FFT Size): 2048 или 4096")
    print("   - Тип окна: Hann или Blackman-Harris")
    print("   - Перекрытие (Overlap): 75%")
    print("   - Масштаб частот: линейный (не логарифмический)")
    print("4. Ожидаемая картина:")
    print("   - Преамбула: 8 восходящих чирпов (прямые линии снизу вверх)")
    print("   - Синхросигнал: 2 нисходящих чирпа (линии сверху вниз)")
    print("   - Данные: восходящие чирпы со сдвигом по частоте")
    print("   - CRC: 2 нисходящих чирпа")
    print("\nЧастотный диапазон сигнала: {:.0f} - {:.0f} Гц".format(
        args.center_freq - args.bw / 2000,
        args.center_freq + args.bw / 2000
    ))
    print("=" * 60)
    print("\nГотово! Файл готов для анализа в Adobe Audition.")


if __name__ == '__main__':
    main()
