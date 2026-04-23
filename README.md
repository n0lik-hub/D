# LoRa CSS Signal Generator

Генератор цифровых аудиофайлов со спектрограммой сигнала LoRa (CSS-модуляция) для образовательных и тестовых целей.

## Описание

Программа генерирует WAV-файлы с модуляцией Chirp Spread Spectrum (CSS), аналогичной используемой в технологии LoRa. Сигнал масштабируется в аудиодиапазон для визуализации и анализа.

## Возможности

- ✅ Генерация базовых CSS-чирпов (upchirp/downchirp)
- ✅ Поддержка параметров LoRa:
  - Spreading Factor (SF): 7–12
  - Bandwidth (BW): 125, 250, 500 кГц
  - Настраиваемая центральная частота (аудиодиапазон)
  - Длительность символа: $T_{sym} = 2^{SF} / BW$
- ✅ Формирование полной структуры кадра LoRa:
  - Преамбула (8 upchirp)
  - Сетевые синхросигналы (2 downchirp)
  - Заголовок (опционально explicit/implicit)
  - Полезные данные
  - CRC
- ✅ Экспорт в WAV (PCM 16/24/32-bit)
- ✅ Генерация спектрограммы (PNG)
- ✅ Сохранение метаданных (JSON)
- ✅ CLI-интерфейс и конфигурационные файлы

## Установка

### Требования

- Python 3.9+
- numpy
- scipy
- soundfile
- matplotlib

```bash
pip install numpy scipy soundfile matplotlib
```

### Быстрый старт

```bash
# Генерация сигнала с параметрами по умолчанию
python lora_css_generator.py --payload "HELLO" --output test.wav

# Генерация с разными SF
python lora_css_generator.py --sf 7 --payload "TEST_SF7" --output sf7.wav
python lora_css_generator.py --sf 9 --payload "TEST_SF9" --output sf9.wav
python lora_css_generator.py --sf 12 --payload "TEST_SF12" --output sf12.wav

# Настройка параметров
python lora_css_generator.py --sf 9 --bw 250000 --center-freq 8000 --no-crc --output custom.wav
```

## Использование

### Командная строка

```
usage: lora_css_generator.py [-h] [--sf {7,8,9,10,11,12}] [--bw {125000,250000,500000}]
                             [--center-freq CENTER_FREQ] [--fs FS] [--bits {16,24,32}]
                             [--amplitude AMPLITUDE] [--preamble PREAMBLE] [--explicit-header]
                             [--implicit-header] [--no-crc] [--payload PAYLOAD] [--output OUTPUT]
                             [--no-spectrogram] [--spectrogram-output SPECTROGRAM_OUTPUT]
                             [--metadata-output METADATA_OUTPUT] [--config CONFIG]
```

#### Параметры

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `--sf` | Spreading Factor (7-12) | 7 |
| `--bw` | Bandwidth в Гц (125000, 250000, 500000) | 125000 |
| `--center-freq` | Центральная частота в Гц | 4000 |
| `--fs` | Частота дискретизации в Гц | 48000 |
| `--bits` | Бит на отсчет (16, 24, 32) | 16 |
| `--amplitude` | Амплитуда сигнала (0.0-1.0) | 0.9 |
| `--preamble` | Длина преамбулы в символах | 8 |
| `--payload` | Полезная нагрузка (ASCII или 0x-hex) | "TEST" |
| `--output` | Выходной WAV-файл | lora_signal.wav |
| `--no-crc` | Отключить CRC | false |
| `--no-spectrogram` | Не генерировать спектрограмму | false |

### Примеры

#### Генерация с разной длительностью символа

```bash
# Короткие символы (SF=7, быстрая передача)
python lora_css_generator.py --sf 7 --payload "DATA" --output fast.wav

# Длинные символы (SF=12, высокая чувствительность)
python lora_css_generator.py --sf 12 --payload "DATA" --output sensitive.wav
```

#### Hex-payload

```bash
# Передача байтов в hex-формате
python lora_css_generator.py --payload "0x48454C4C4F" --output hex.wav
```

#### Отключение спектрограммы для скорости

```bash
python lora_css_generator.py --payload "QUICK" --no-spectrogram --output quick.wav
```

## Структура проекта

```
/workspace
├── lora_css_generator.py    # Основной модуль генератора
├── test_lora_generator.py   # Набор тестов
├── README.md                # Документация
├── examples/                # Примеры конфигураций
│   ├── config_sf7.json
│   ├── config_sf9.json
│   └── config_sf12.json
└── output/                  # Сгенерированные файлы
    ├── *.wav               # Аудиофайлы
    ├── *.png               # Спектрограммы
    └── *.json              # Метаданные
```

## Математическая модель

### Базовый чирп

$$s(t) = A \cdot w(t) \cdot \cos\left(2\pi \left(f_c t + \frac{B}{2T_{sym}} t^2\right)\right), \quad t \in [0, T_{sym}]$$

где:
- $A$ – амплитуда (нормирована к [-1, 1])
- $w(t)$ – оконная функция Hann
- $B$ – полоса частот (аудио-масштабированная)
- $f_c$ – начальная частота чирпа
- $T_{sym} = 2^{SF} / BW$ – длительность символа

### Кодирование данных

Полезный символ формируется циклическим сдвигом upchirp на $k = \text{symbol\_value}$ позиций:

$$s_k(t) = s_{up}(t) \cdot e^{j 2\pi k t / T_{sym}}$$

## Тестирование

```bash
# Запуск всех тестов
python test_lora_generator.py

# Запуск с подробным выводом
python -m unittest test_lora_generator -v
```

### Покрытие тестов

- ✅ Конфигурация и валидация параметров
- ✅ Расчет длительности символов
- ✅ Генерация upchirp/downchirp
- ✅ Циклический сдвиг для кодирования данных
- ✅ Оконная функция (Hann)
- ✅ Структура кадра (преамбула, sync, header, payload, CRC)
- ✅ WAV экспорт (16/24/32 bit)
- ✅ Генерация спектрограммы
- ✅ JSON метаданные

## Выходные файлы

### WAV-файл

Стандартный RIFF WAV формат с поддерживаемыми форматами:
- PCM 16-bit (INT16)
- PCM 24-bit (INT24)
- PCM 32-bit (INT32)

### Спектрограмма

PNG изображение с параметрами STFT:
- `n_fft`: min(2048, samples_per_symbol)
- `hop_length`: n_fft / 4
- `window`: Hann

### Метаданные (JSON)

```json
{
  "config": { ... },
  "calculated_parameters": {
    "t_sym_seconds": 0.001024,
    "t_sym_ms": 1.024,
    "num_symbols_per_chirp": 128,
    "samples_per_symbol": 50,
    "nyquist_frequency": 24000.0
  },
  "frame_info": {
    "frame_structure": {
      "preamble": 8,
      "sync": 2,
      "header": 4,
      "payload": 10,
      "crc": 2
    },
    "symbol_count": 26,
    "payload_symbols": [...]
  }
}
```

## Ограничения

1. **Аудио-диапазон**: Сигнал генерируется в аудиодиапазоне (20 Гц - 20 кГц) только для визуализации и анализа. Не предназначен для прямой передачи в эфир.

2. **Упрощенная модель**: Реализована базовая физический уровень без:
   - MAC-уровня
   - Шифрования
   - Привязки к конкретным каналам ISM
   - Полной совместимости с аппаратными LoRa-приемниками

3. **Требования к Fs**: При недостаточной частоте дискретизации возможна потеря высокочастотных компонент. Программа выдает предупреждение если $F_s < 2 \cdot (f_c + BW)$.

## Лицензия

MIT License

## Авторы

LoRa CSS Generator Project

## Примечания

Для корректного отображения спектрограммы в сторонних программах (Audacity, Sonic Visualiser) рекомендуется:
- Использовать $F_s \geq 44.1$ кГц
- Выбирать BW, кратные 1 кГц
- Устанавливать центральную частоту в диапазоне 1-8 кГц
