"""
Shared pytest fixtures and configuration for the Video Super-Resolution test suite.
"""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import numpy as np
import pytest
import yaml
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory that is cleaned up after the test."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide a sample configuration dictionary for testing."""
    return {
        "name": "test_experiment",
        "model": "ESPCN",
        "scale": 4,
        "datasets": {
            "train": {
                "name": "sample_dataset",
                "batch_size": 16,
                "num_workers": 4,
            },
            "val": {
                "name": "sample_dataset",
                "batch_size": 1,
                "num_workers": 1,
            }
        },
        "network": {
            "type": "ESPCN",
            "num_channels": 3,
            "scale": 4,
        },
        "train": {
            "lr": 0.0001,
            "lr_scheme": "MultiStepLR",
            "lr_steps": [10000, 20000],
            "lr_gamma": 0.5,
            "iterations": 30000,
            "val_freq": 1000,
            "save_freq": 5000,
        }
    }


@pytest.fixture
def sample_yaml_config(temp_dir: Path, sample_config: Dict[str, Any]) -> Path:
    """Create a sample YAML configuration file."""
    config_path = temp_dir / "config.yml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_image(temp_dir: Path) -> Path:
    """Create a sample image for testing."""
    image_path = temp_dir / "sample_image.png"
    # Create a simple 64x64 RGB image with random pixels
    image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    image = Image.fromarray(image_array)
    image.save(image_path)
    return image_path


@pytest.fixture
def sample_video_frames(temp_dir: Path, num_frames: int = 5) -> Path:
    """Create a directory with sample video frames."""
    frames_dir = temp_dir / "video_frames"
    frames_dir.mkdir()
    
    for i in range(num_frames):
        frame_path = frames_dir / f"frame_{i:04d}.png"
        # Create frames with slight variations
        base_color = (i * 50) % 255
        image_array = np.full((64, 64, 3), base_color, dtype=np.uint8)
        # Add some noise to make frames different
        noise = np.random.randint(-10, 10, (64, 64, 3))
        image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(image_array)
        image.save(frame_path)
    
    return frames_dir


@pytest.fixture
def mock_model_checkpoint(temp_dir: Path) -> Path:
    """Create a mock model checkpoint file."""
    checkpoint_path = temp_dir / "model_checkpoint.pth"
    # Create a dummy file (in real tests, you'd save actual model state)
    checkpoint_path.touch()
    return checkpoint_path


@pytest.fixture
def mock_lmdb_dataset(temp_dir: Path) -> Path:
    """Create a mock LMDB dataset directory."""
    lmdb_dir = temp_dir / "lmdb_dataset"
    lmdb_dir.mkdir()
    # In real tests, you'd create actual LMDB files here
    (lmdb_dir / "data.mdb").touch()
    (lmdb_dir / "lock.mdb").touch()
    return lmdb_dir


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Ensure matplotlib doesn't create unwanted GUI windows during tests."""
    import matplotlib
    matplotlib.use('Agg')
    yield
    import matplotlib.pyplot as plt
    plt.close('all')


@pytest.fixture
def capture_stdout(monkeypatch):
    """Capture stdout for testing print statements."""
    import io
    import sys
    
    captured_output = io.StringIO()
    monkeypatch.setattr(sys, 'stdout', captured_output)
    
    yield captured_output
    
    # Reset stdout
    monkeypatch.undo()


@pytest.fixture
def mock_cuda_available(monkeypatch):
    """Mock CUDA availability for testing GPU-related code."""
    def mock_is_available():
        return False
    
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', mock_is_available)
    
    yield


@pytest.fixture
def sample_metrics() -> Dict[str, float]:
    """Provide sample metrics for testing."""
    return {
        'psnr': 28.5,
        'ssim': 0.85,
        'lpips': 0.12,
        'runtime': 1.234,
    }


@pytest.fixture
def env_setup(monkeypatch):
    """Set up environment variables for testing."""
    test_env = {
        'CUDA_VISIBLE_DEVICES': '-1',  # Disable GPU for tests
        'PYTHONPATH': str(Path.cwd()),
    }
    
    for key, value in test_env.items():
        monkeypatch.setenv(key, value)
    
    yield test_env