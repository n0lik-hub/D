#!/usr/bin/env python3
"""
LoRa CSS (Chirp Spread Spectrum) Signal Generator

Generates digital audio WAV files with LoRa-like CSS modulation for educational
and testing purposes. The signal is scaled to audio frequency range for visualization
and analysis.

Author: LoRa CSS Generator
License: MIT
"""

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
from scipy.signal import get_window
import soundfile as sf


@dataclass
class LoraConfig:
    """Configuration parameters for LoRa CSS signal generation."""
    
    # LoRa parameters
    sf: int = 7  # Spreading Factor (7-12)
    bw: float = 125000.0  # Bandwidth in Hz (125k, 250k, 500k)
    
    # Audio scaling parameters
    center_freq: float = 4000.0  # Center frequency in Hz (audio range)
    fs: float = 48000.0  # Sampling frequency in Hz
    
    # Output parameters
    bits_per_sample: int = 16  # PCM bits: 16, 24, 32, or 32f (float)
    amplitude: float = 0.9  # Normalized amplitude (0.0 - 1.0)
    
    # Frame structure
    preamble_len: int = 8  # Number of upchirps in preamble
    explicit_header: bool = True  # Use explicit header
    crc_enabled: bool = True  # Enable CRC symbols
    
    # Payload
    payload: str = "TEST"  # ASCII or hex payload
    
    # Output files
    output_wav: str = "lora_signal.wav"
    generate_spectrogram: bool = True
    output_spectrogram: str = "lora_spectrogram.png"
    output_metadata: str = "lora_metadata.json"
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        warnings = []
        errors = []
        
        # Validate SF
        if not (7 <= self.sf <= 12):
            errors.append(f"SF must be between 7 and 12, got {self.sf}")
        
        # Validate BW
        if self.bw not in [125000, 250000, 500000]:
            warnings.append(f"Non-standard BW: {self.bw} Hz")
        
        # Validate sampling rate (Nyquist criterion)
        f_max = self.center_freq + self._get_audio_bw() / 2
        min_fs = 2 * f_max
        if self.fs < min_fs:
            errors.append(f"Fs={self.fs} too low for f_max={f_max:.1f} Hz (need >= {min_fs:.1f} Hz)")
        
        # Validate amplitude
        if not (0 < self.amplitude <= 1.0):
            errors.append(f"Amplitude must be in (0, 1.0], got {self.amplitude}")
        
        # Validate bits per sample
        valid_bits = [16, 24, 32]
        if self.bits_per_sample not in valid_bits and self.bits_per_sample != 32:
            warnings.append(f"Non-standard bits_per_sample: {self.bits_per_sample}")
        
        return errors + warnings
    
    def _get_audio_bw(self) -> float:
        """Get the audio-scaled bandwidth."""
        # Scale RF bandwidth to audio range proportionally
        # For audio visualization, we use a fraction of the original BW
        # mapped to the audio frequency range
        return self.bw * (self.center_freq / (self.bw / 2)) * 0.1
    
    @property
    def t_sym(self) -> float:
        """Symbol duration in seconds."""
        return (2 ** self.sf) / self.bw
    
    @property
    def num_symbols(self) -> int:
        """Number of symbols per chirp."""
        return 2 ** self.sf
    
    @property
    def samples_per_symbol(self) -> int:
        """Number of samples per symbol."""
        return int(np.ceil(self.t_sym * self.fs))


def parse_payload(payload_str: str, sf: int) -> List[int]:
    """
    Parse payload string into list of symbol values.
    
    Args:
        payload_str: ASCII string or hex string (prefixed with 0x)
        sf: Spreading factor determines symbols per byte
    
    Returns:
        List of symbol values (0 to 2^SF - 1)
    """
    if payload_str.startswith("0x") or payload_str.startswith("0X"):
        # Hex payload
        hex_str = payload_str[2:]
        if len(hex_str) % 2:
            hex_str = "0" + hex_str
        bytes_data = bytes.fromhex(hex_str)
    else:
        # ASCII payload
        bytes_data = payload_str.encode('utf-8')
    
    symbols = []
    symbols_per_byte = max(1, sf - 2)  # LoRa encoding: SF-2 symbols per byte for SF>=5
    
    for byte in bytes_data:
        # Simple mapping: split byte into symbols
        mask = (1 << sf) - 1
        for i in range(symbols_per_byte):
            symbol = (byte >> (i * sf)) & mask
            symbols.append(symbol)
    
    # If no symbols generated, add a default
    if not symbols:
        symbols = [0]
    
    return symbols


def generate_chirp(config: LoraConfig, 
                   chirp_type: str = 'up',
                   shift: int = 0,
                   phase_offset: float = 0.0) -> np.ndarray:
    """
    Generate a single CSS chirp symbol.
    
    Args:
        config: LoRa configuration
        chirp_type: 'up' for upchirp, 'down' for downchirp
        shift: Cyclic shift value (0 to 2^SF - 1) for data encoding
        phase_offset: Starting phase for phase continuity
    
    Returns:
        numpy array with chirp samples
    """
    n_samples = config.samples_per_symbol
    t = np.arange(n_samples) / config.fs
    
    # Calculate frequency sweep parameters
    # Map RF bandwidth to audio range
    audio_bw = config.center_freq * 0.5  # Use half of center freq as audio BW
    f_start = config.center_freq - audio_bw / 2
    f_end = config.center_freq + audio_bw / 2
    
    # Chirp rate (frequency change per second)
    chirp_rate = audio_bw / config.t_sym
    
    if chirp_type == 'up':
        # Upchirp: frequency increases linearly
        # Apply cyclic shift by adding phase term
        instantaneous_freq = f_start + chirp_rate * t
        
        # Add shift as frequency offset (equivalent to cyclic shift in time domain)
        if shift > 0:
            shift_time = shift / config.num_symbols * config.t_sym
            # Circular shift implementation
            t_shifted = (t - shift_time) % config.t_sym
            instantaneous_freq = f_start + chirp_rate * t_shifted
    else:
        # Downchirp: frequency decreases linearly
        instantaneous_freq = f_end - chirp_rate * t
    
    # Integrate frequency to get phase
    # phase = 2*pi * integral(f(t) dt)
    phase = 2 * np.pi * np.cumsum(instantaneous_freq) / config.fs + phase_offset
    
    # Generate signal with windowing
    signal = np.cos(phase)
    
    # Apply Hann window to reduce spectral leakage
    window = get_window('hann', n_samples)
    signal = signal * window
    
    # Normalize amplitude
    signal = signal * config.amplitude
    
    return signal


def calculate_crc16(data: List[int]) -> int:
    """Calculate CRC-16 for LoRa header/data."""
    crc = 0xFFFF
    poly = 0x1021
    
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    
    return crc


def generate_lora_frame(config: LoraConfig) -> Tuple[np.ndarray, dict]:
    """
    Generate complete LoRa frame with preamble, sync, header, payload, and CRC.
    
    Args:
        config: LoRa configuration
    
    Returns:
        Tuple of (signal array, metadata dict)
    """
    symbols = []
    metadata = {
        'frame_structure': {},
        'symbol_count': 0,
        'payload_symbols': [],
    }
    
    # 1. Preamble: series of upchirps
    preamble_symbols = [0] * config.preamble_len
    symbols.extend(preamble_symbols)
    metadata['frame_structure']['preamble'] = config.preamble_len
    
    # 2. Sync word: 2 downchirps (network identifier)
    # In real LoRa, sync word is encoded differently
    symbols.extend([-1, -1])  # -1 indicates downchirp
    metadata['frame_structure']['sync'] = 2
    
    # 3. Header (if explicit mode)
    if config.explicit_header:
        # Header contains: CR, SF, BW, payload length, CRC enabled
        # Simplified: just add some header symbols
        header_bytes = [len(parse_payload(config.payload, config.sf))]
        header_symbols = parse_payload(bytes(header_bytes).decode('latin-1'), config.sf)
        symbols.extend(header_symbols[:4])  # Header is typically 8 symbols coded
        metadata['frame_structure']['header'] = len(header_symbols)
    
    # 4. Payload
    payload_symbols = parse_payload(config.payload, config.sf)
    symbols.extend(payload_symbols)
    metadata['payload_symbols'] = payload_symbols
    metadata['frame_structure']['payload'] = len(payload_symbols)
    
    # 5. CRC (if enabled): 2 downchirps or coded symbols
    if config.crc_enabled:
        # Calculate simple CRC over payload
        crc_val = calculate_crc16([s % 256 for s in payload_symbols])
        crc_symbols = [(crc_val >> 8) & 0xFF, crc_val & 0xFF]
        # Encode as regular symbols (not downchirps in this simplified version)
        symbols.extend([s % config.num_symbols for s in crc_symbols])
        metadata['frame_structure']['crc'] = 2
    
    metadata['symbol_count'] = len(symbols)
    
    # Generate signal from symbols
    signal_parts = []
    phase_offset = 0.0
    
    for i, symbol in enumerate(symbols):
        if symbol < 0:
            # Downchirp (sync/CRC markers)
            chirp = generate_chirp(config, chirp_type='down', phase_offset=phase_offset)
        else:
            # Upchirp with cyclic shift for data
            chirp = generate_chirp(config, chirp_type='up', shift=symbol % config.num_symbols,
                                   phase_offset=phase_offset)
        
        signal_parts.append(chirp)
        
        # Maintain phase continuity (approximate)
        if i < len(symbols) - 1:
            phase_offset = phase_offset + 2 * np.pi * config.center_freq * config.t_sym
    
    # Concatenate all parts
    signal = np.concatenate(signal_parts)
    
    return signal, metadata


def write_wav(signal: np.ndarray, config: LoraConfig, filepath: str) -> None:
    """
    Write signal to WAV file with specified bit depth.
    
    Args:
        signal: Audio signal array (float32, range -1 to 1)
        config: Configuration with bit depth
        filepath: Output file path
    """
    # Ensure signal is in valid range
    signal = np.clip(signal, -1.0, 1.0)
    
    # Convert to appropriate format
    if config.bits_per_sample == 32 and str(config.bits_per_sample).endswith('f'):
        # 32-bit float
        data = signal.astype(np.float32)
        subtype = 'FLOAT'
    elif config.bits_per_sample == 16:
        # 16-bit PCM
        data = (signal * 32767).astype(np.int16)
        subtype = 'PCM_16'
    elif config.bits_per_sample == 24:
        # 24-bit PCM
        data = (signal * 8388607).astype(np.int32)
        subtype = 'PCM_24'
    elif config.bits_per_sample == 32:
        # 32-bit PCM
        data = (signal * 2147483647).astype(np.int32)
        subtype = 'PCM_32'
    else:
        # Default to 16-bit
        data = (signal * 32767).astype(np.int16)
        subtype = 'PCM_16'
    
    # Write using soundfile
    sf.write(filepath, data, int(config.fs), subtype=subtype)
    print(f"WAV file written: {filepath}")
    print(f"  Format: {subtype}, Sample rate: {config.fs} Hz, Duration: {len(signal)/config.fs:.3f} s")


def generate_spectrogram(signal: np.ndarray, config: LoraConfig, filepath: str) -> None:
    """
    Generate and save spectrogram of the signal.
    
    Args:
        signal: Audio signal array
        config: Configuration
        filepath: Output PNG path
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram
    
    # Compute spectrogram
    n_fft = min(2048, config.samples_per_symbol)
    hop_length = n_fft // 4
    
    f, t, Sxx = spectrogram(signal, config.fs, window='hann', 
                            nperseg=n_fft, noverlap=n_fft - hop_length)
    
    # Convert to dB
    Sxx_db = 10 * np.log10(Sxx_db + 1e-10) if (Sxx_db := Sxx.copy()).size else 10 * np.log10(Sxx + 1e-10)
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot spectrogram
    im = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
    
    # Labels and title
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'LoRa CSS Spectrogram (SF={config.sf}, BW={config.bw/1000:.0f} kHz)')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Power (dB)')
    
    # Mark expected frequency range
    audio_bw = config.center_freq * 0.5
    ax.axhline(y=config.center_freq - audio_bw/2, color='r', linestyle='--', 
               alpha=0.5, label='Band edges')
    ax.axhline(y=config.center_freq + audio_bw/2, color='r', linestyle='--', alpha=0.5)
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # Save
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spectrogram saved: {filepath}")


def save_metadata(config: LoraConfig, frame_metadata: dict, filepath: str) -> None:
    """Save generation metadata to JSON file."""
    metadata = {
        'config': asdict(config),
        'calculated_parameters': {
            't_sym_seconds': config.t_sym,
            't_sym_ms': config.t_sym * 1000,
            'num_symbols_per_chirp': config.num_symbols,
            'samples_per_symbol': config.samples_per_symbol,
            'nyquist_frequency': config.fs / 2,
        },
        'frame_info': frame_metadata,
    }
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved: {filepath}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Generate LoRa CSS-modulated audio signals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --sf 7 --payload "HELLO" --output test.wav
  %(prog)s --sf 9 --bw 250000 --center-freq 8000 --no-crc
  %(prog)s --config config.yaml
        """
    )
    
    # LoRa parameters
    parser.add_argument('--sf', type=int, default=7, choices=range(7, 13),
                        help='Spreading Factor (7-12, default: 7)')
    parser.add_argument('--bw', type=float, default=125000, 
                        choices=[125000, 250000, 500000],
                        help='Bandwidth in Hz (default: 125000)')
    
    # Audio parameters
    parser.add_argument('--center-freq', type=float, default=4000,
                        help='Center frequency in Hz (default: 4000)')
    parser.add_argument('--fs', type=float, default=48000,
                        help='Sampling frequency in Hz (default: 48000)')
    parser.add_argument('--bits', type=int, default=16, choices=[16, 24, 32],
                        help='Bits per sample (default: 16)')
    parser.add_argument('--amplitude', type=float, default=0.9,
                        help='Signal amplitude 0.0-1.0 (default: 0.9)')
    
    # Frame structure
    parser.add_argument('--preamble', type=int, default=8,
                        help='Preamble length in symbols (default: 8)')
    parser.add_argument('--explicit-header', action='store_true', default=True,
                        help='Use explicit header (default)')
    parser.add_argument('--implicit-header', action='store_false', dest='explicit_header',
                        help='Use implicit header mode')
    parser.add_argument('--no-crc', action='store_true',
                        help='Disable CRC symbols')
    
    # Payload
    parser.add_argument('--payload', type=str, default='TEST',
                        help='Payload string (ASCII or 0x-prefixed hex)')
    
    # Output
    parser.add_argument('--output', '-o', type=str, default='lora_signal.wav',
                        help='Output WAV file (default: lora_signal.wav)')
    parser.add_argument('--no-spectrogram', action='store_true',
                        help='Skip spectrogram generation')
    parser.add_argument('--spectrogram-output', type=str, default='lora_spectrogram.png',
                        help='Spectrogram output file')
    parser.add_argument('--metadata-output', type=str, default='lora_metadata.json',
                        help='Metadata JSON output file')
    
    # Config file
    parser.add_argument('--config', type=str,
                        help='Load configuration from JSON/YAML file')
    
    args = parser.parse_args()
    
    # Load config from file if provided
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix in ['.yaml', '.yml']:
            try:
                import yaml
                with open(config_path) as f:
                    config_dict = yaml.safe_load(f)
            except ImportError:
                print("Error: PyYAML not installed. Install with: pip install pyyaml")
                sys.exit(1)
        else:
            with open(config_path) as f:
                config_dict = json.load(f)
        
        # Update args from config
        for key, value in config_dict.items():
            if hasattr(args, key.replace('-', '_')):
                setattr(args, key.replace('-', '_'), value)
    
    # Create configuration
    config = LoraConfig(
        sf=args.sf,
        bw=args.bw,
        center_freq=args.center_freq,
        fs=args.fs,
        bits_per_sample=args.bits,
        amplitude=args.amplitude,
        preamble_len=args.preamble,
        explicit_header=args.explicit_header,
        crc_enabled=not args.no_crc,
        payload=args.payload,
        output_wav=args.output,
        generate_spectrogram=not args.no_spectrogram,
        output_spectrogram=args.spectrogram_output,
        output_metadata=args.metadata_output,
    )
    
    # Validate configuration
    errors, warnings = [], []
    validation_result = config.validate()
    for msg in validation_result:
        if msg.startswith("ERROR"):
            errors.append(msg)
        else:
            warnings.append(msg)
    
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  ⚠ {w}")
    
    if errors:
        print("Errors:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    
    # Print configuration
    print("=" * 60)
    print("LoRa CSS Signal Generator")
    print("=" * 60)
    print(f"Spreading Factor: {config.sf}")
    print(f"Bandwidth: {config.bw / 1000:.0f} kHz")
    print(f"Symbol Duration: {config.t_sym * 1000:.2f} ms")
    print(f"Samples/Symbol: {config.samples_per_symbol}")
    print(f"Center Frequency: {config.center_freq:.0f} Hz")
    print(f"Sample Rate: {config.fs:.0f} Hz")
    print(f"Payload: '{config.payload}'")
    print("=" * 60)
    
    # Generate frame
    print("\nGenerating LoRa frame...")
    signal, metadata = generate_lora_frame(config)
    
    print(f"Total symbols: {metadata['symbol_count']}")
    print(f"Signal duration: {len(signal) / config.fs:.3f} s")
    print(f"Signal samples: {len(signal)}")
    
    # Write WAV
    print("\nWriting WAV file...")
    write_wav(signal, config, config.output_wav)
    
    # Generate spectrogram
    if config.generate_spectrogram:
        print("\nGenerating spectrogram...")
        generate_spectrogram(signal, config, config.output_spectrogram)
    
    # Save metadata
    print("\nSaving metadata...")
    save_metadata(config, metadata, config.output_metadata)
    
    print("\n" + "=" * 60)
    print("Generation complete!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
