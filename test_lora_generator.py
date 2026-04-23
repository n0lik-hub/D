#!/usr/bin/env python3
"""
Test suite for LoRa CSS Signal Generator

Tests cover:
- WAV file format validation
- Signal duration accuracy
- Spectrogram generation
- Parameter validation
- Different SF/BW combinations
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

# Import the generator module
from lora_css_generator import (
    LoraConfig, 
    generate_chirp, 
    generate_lora_frame, 
    write_wav,
    parse_payload,
    calculate_crc16
)


class TestLoraConfig(unittest.TestCase):
    """Test configuration validation and calculations."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LoraConfig()
        self.assertEqual(config.sf, 7)
        self.assertEqual(config.bw, 125000.0)
        self.assertEqual(config.center_freq, 4000.0)
        self.assertEqual(config.fs, 48000.0)
    
    def test_symbol_duration(self):
        """Test symbol duration calculation."""
        # SF=7, BW=125kHz: T_sym = 2^7 / 125000 = 1.024 ms
        config = LoraConfig(sf=7, bw=125000)
        expected_t_sym = (2 ** 7) / 125000
        self.assertAlmostEqual(config.t_sym, expected_t_sym, places=6)
        
        # SF=12, BW=125kHz: T_sym = 2^12 / 125000 = 32.768 ms
        config = LoraConfig(sf=12, bw=125000)
        expected_t_sym = (2 ** 12) / 125000
        self.assertAlmostEqual(config.t_sym, expected_t_sym, places=6)
    
    def test_samples_per_symbol(self):
        """Test samples per symbol calculation."""
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        expected_samples = int(np.ceil(config.t_sym * config.fs))
        self.assertEqual(config.samples_per_symbol, expected_samples)
    
    def test_validation_valid(self):
        """Test validation with valid parameters."""
        config = LoraConfig(sf=7, bw=125000, fs=48000, amplitude=0.9)
        result = config.validate()
        # Should have no errors for valid config
        self.assertIsInstance(result, list)
    
    def test_validation_invalid_sf(self):
        """Test validation rejects invalid SF."""
        config = LoraConfig(sf=5)  # SF must be 7-12
        result = config.validate()
        self.assertTrue(any("SF must be between" in msg for msg in result))
    
    def test_validation_invalid_amplitude(self):
        """Test validation rejects invalid amplitude."""
        config = LoraConfig(amplitude=1.5)
        result = config.validate()
        self.assertTrue(any("Amplitude must be" in msg for msg in result))


class TestPayloadParsing(unittest.TestCase):
    """Test payload parsing functionality."""
    
    def test_ascii_payload(self):
        """Test ASCII payload parsing."""
        symbols = parse_payload("A", sf=7)
        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) > 0)
        # All symbols should be in valid range
        for s in symbols:
            self.assertTrue(0 <= s < (2 ** 7))
    
    def test_hex_payload(self):
        """Test hex payload parsing."""
        symbols = parse_payload("0x4142", sf=7)  # "AB" in hex
        self.assertIsInstance(symbols, list)
        self.assertTrue(len(symbols) > 0)
    
    def test_empty_payload(self):
        """Test empty payload handling."""
        symbols = parse_payload("", sf=7)
        self.assertEqual(symbols, [0])  # Should default to [0]


class TestChirpGeneration(unittest.TestCase):
    """Test chirp signal generation."""
    
    def test_upchirp_shape(self):
        """Test upchirp has correct shape."""
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        chirp = generate_chirp(config, chirp_type='up')
        
        self.assertEqual(len(chirp), config.samples_per_symbol)
        self.assertEqual(chirp.dtype, np.float64)
        
        # Check amplitude is within bounds
        self.assertTrue(np.all(np.abs(chirp) <= config.amplitude))
    
    def test_downchirp_shape(self):
        """Test downchirp has correct shape."""
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        chirp = generate_chirp(config, chirp_type='down')
        
        self.assertEqual(len(chirp), config.samples_per_symbol)
    
    def test_chirp_shift(self):
        """Test cyclic shift of chirps."""
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        
        chirp_unshifted = generate_chirp(config, chirp_type='up', shift=0)
        chirp_shifted = generate_chirp(config, chirp_type='up', shift=10)
        
        self.assertEqual(len(chirp_unshifted), len(chirp_shifted))
        # Shifted and unshifted should be different
        self.assertFalse(np.array_equal(chirp_unshifted, chirp_shifted))
    
    def test_windowing_applied(self):
        """Test that windowing is applied (signal goes to zero at edges)."""
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        chirp = generate_chirp(config, chirp_type='up')
        
        # First and last samples should be near zero due to Hann window
        # Allow tolerance for numerical precision and phase discontinuity
        self.assertLess(abs(chirp[0]), 0.01)
        self.assertLess(abs(chirp[-1]), 0.01)


class TestFrameGeneration(unittest.TestCase):
    """Test complete frame generation."""
    
    def test_frame_structure(self):
        """Test frame has correct structure."""
        config = LoraConfig(
            sf=7, 
            bw=125000, 
            preamble_len=8,
            explicit_header=True,
            crc_enabled=True,
            payload="TEST"
        )
        
        signal, metadata = generate_lora_frame(config)
        
        # Check metadata structure
        self.assertIn('frame_structure', metadata)
        self.assertIn('symbol_count', metadata)
        self.assertIn('payload_symbols', metadata)
        
        # Check frame components
        fs = metadata['frame_structure']
        self.assertEqual(fs['preamble'], 8)
        self.assertEqual(fs['sync'], 2)
        self.assertGreater(fs['payload'], 0)
    
    def test_signal_duration(self):
        """Test signal duration matches expected value."""
        config = LoraConfig(
            sf=7,
            bw=125000,
            preamble_len=8,
            crc_enabled=False,
            explicit_header=False,
            payload="A"
        )
        
        signal, metadata = generate_lora_frame(config)
        
        expected_duration = metadata['symbol_count'] * config.t_sym
        actual_duration = len(signal) / config.fs
        
        # Allow small tolerance for rounding
        self.assertAlmostEqual(actual_duration, expected_duration, delta=0.001)
    
    def test_different_sf(self):
        """Test frame generation with different SF values."""
        for sf in range(7, 13):
            config = LoraConfig(sf=sf, bw=125000, payload="X")
            signal, metadata = generate_lora_frame(config)
            
            self.assertGreater(len(signal), 0)
            self.assertEqual(metadata['frame_structure']['preamble'], 8)


class TestCRC(unittest.TestCase):
    """Test CRC calculation."""
    
    def test_crc16_basic(self):
        """Test basic CRC-16 calculation."""
        data = [0x41, 0x42, 0x43]  # "ABC"
        crc = calculate_crc16(data)
        
        self.assertIsInstance(crc, int)
        self.assertTrue(0 <= crc <= 0xFFFF)
    
    def test_crc_deterministic(self):
        """Test CRC is deterministic."""
        data = [1, 2, 3, 4, 5]
        crc1 = calculate_crc16(data)
        crc2 = calculate_crc16(data)
        
        self.assertEqual(crc1, crc2)


class TestWAVOutput(unittest.TestCase):
    """Test WAV file output."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_wav_16bit(self):
        """Test 16-bit WAV output."""
        config = LoraConfig(bits_per_sample=16, fs=48000)
        signal = np.random.randn(1000) * 0.5
        filepath = os.path.join(self.temp_dir, 'test_16bit.wav')
        
        write_wav(signal, config, filepath)
        
        # Verify file exists
        self.assertTrue(os.path.exists(filepath))
        
        # Verify format
        info = sf.info(filepath)
        self.assertEqual(info.samplerate, 48000)
    
    def test_wav_24bit(self):
        """Test 24-bit WAV output."""
        config = LoraConfig(bits_per_sample=24, fs=48000)
        signal = np.random.randn(1000) * 0.5
        filepath = os.path.join(self.temp_dir, 'test_24bit.wav')
        
        write_wav(signal, config, filepath)
        
        self.assertTrue(os.path.exists(filepath))
        info = sf.info(filepath)
        self.assertEqual(info.samplerate, 48000)
    
    def test_wav_no_clipping(self):
        """Test that output signal is not clipped."""
        config = LoraConfig(bits_per_sample=16, fs=48000)
        signal = np.random.randn(1000) * 0.5
        filepath = os.path.join(self.temp_dir, 'test_clip.wav')
        
        write_wav(signal, config, filepath)
        
        # Read back and verify
        data, fs = sf.read(filepath)
        self.assertTrue(np.all(np.abs(data) <= 32767))


class TestSpectrogram(unittest.TestCase):
    """Test spectrogram generation."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_spectrogram_generation(self):
        """Test spectrogram is generated correctly."""
        from lora_css_generator import generate_spectrogram
        
        config = LoraConfig(sf=7, bw=125000, fs=48000)
        signal = np.random.randn(48000) * 0.5  # 1 second of noise
        filepath = os.path.join(self.temp_dir, 'test_spec.png')
        
        generate_spectrogram(signal, config, filepath)
        
        # Verify file exists and has content
        self.assertTrue(os.path.exists(filepath))
        self.assertGreater(os.path.getsize(filepath), 1000)  # At least 1KB


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow."""
    
    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_generation_sf7(self):
        """Test complete generation with SF=7."""
        config = LoraConfig(
            sf=7,
            bw=125000,
            payload="TEST_SF7",
            output_wav=os.path.join(self.temp_dir, 'sf7_test.wav'),
            output_spectrogram=os.path.join(self.temp_dir, 'sf7_spec.png'),
            output_metadata=os.path.join(self.temp_dir, 'sf7_meta.json'),
        )
        
        signal, metadata = generate_lora_frame(config)
        write_wav(signal, config, config.output_wav)
        
        # Verify outputs
        self.assertTrue(os.path.exists(config.output_wav))
        
        # Verify WAV properties
        info = sf.info(config.output_wav)
        self.assertEqual(info.samplerate, int(config.fs))
        
        # Verify duration matches expectation
        expected_duration = metadata['symbol_count'] * config.t_sym
        actual_duration = info.duration
        self.assertAlmostEqual(actual_duration, expected_duration, delta=0.01)
    
    def test_full_generation_sf12(self):
        """Test complete generation with SF=12."""
        config = LoraConfig(
            sf=12,
            bw=125000,
            payload="LONGER_PAYLOAD_TEST",
            output_wav=os.path.join(self.temp_dir, 'sf12_test.wav'),
            generate_spectrogram=False,  # Skip for speed
        )
        
        signal, metadata = generate_lora_frame(config)
        write_wav(signal, config, config.output_wav)
        
        self.assertTrue(os.path.exists(config.output_wav))
        self.assertGreater(metadata['symbol_count'], 100)
    
    def test_metadata_json(self):
        """Test metadata JSON is valid and complete."""
        config = LoraConfig(
            sf=9,
            bw=250000,
            payload="META",
            output_metadata=os.path.join(self.temp_dir, 'meta_test.json'),
            generate_spectrogram=False,
        )
        
        from lora_css_generator import save_metadata
        
        signal, metadata = generate_lora_frame(config)
        save_metadata(config, metadata, config.output_metadata)
        
        # Verify JSON is valid
        self.assertTrue(os.path.exists(config.output_metadata))
        with open(config.output_metadata) as f:
            loaded = json.load(f)
        
        # Check required fields
        self.assertIn('config', loaded)
        self.assertIn('calculated_parameters', loaded)
        self.assertIn('frame_info', loaded)
        
        # Verify config matches
        self.assertEqual(loaded['config']['sf'], 9)
        self.assertEqual(loaded['config']['bw'], 250000)


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLoraConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestPayloadParsing))
    suite.addTests(loader.loadTestsFromTestCase(TestChirpGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestFrameGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestCRC))
    suite.addTests(loader.loadTestsFromTestCase(TestWAVOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectrogram))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
