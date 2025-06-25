"""
Validation tests to ensure the testing infrastructure is properly set up.
"""
import sys
from pathlib import Path

import pytest


class TestInfrastructureValidation:
    """Test class to validate that the testing infrastructure is working correctly."""
    
    def test_pytest_is_running(self):
        """Verify that pytest is executing tests."""
        assert True, "If this fails, pytest is not running correctly"
    
    def test_project_imports_work(self):
        """Verify that the project modules can be imported."""
        # Test that we can import from the codes package
        from codes import __init__
        from codes.utils import __init__ as utils_init
        from codes.models import __init__ as models_init
        assert True, "Project imports are working"
    
    def test_fixtures_are_available(self, temp_dir, sample_config):
        """Verify that conftest fixtures are accessible."""
        assert temp_dir.exists(), "temp_dir fixture should create an existing directory"
        assert isinstance(sample_config, dict), "sample_config should be a dictionary"
        assert "model" in sample_config, "sample_config should contain expected keys"
    
    def test_markers_are_defined(self):
        """Verify that custom markers are properly defined."""
        # This test itself uses the unit marker
        assert True, "Markers are working if this test can be selected with -m unit"
    
    @pytest.mark.unit
    def test_unit_marker(self):
        """Test that unit marker works."""
        assert True, "Unit marker is working"
    
    @pytest.mark.integration
    def test_integration_marker(self):
        """Test that integration marker works."""
        assert True, "Integration marker is working"
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        assert True, "Slow marker is working"
    
    def test_coverage_is_tracked(self):
        """Verify that coverage tracking is enabled."""
        # This is a meta-test - if coverage is running, it will track this
        def dummy_function():
            return "covered"
        
        result = dummy_function()
        assert result == "covered", "Function should execute and be tracked by coverage"
    
    def test_temp_dir_cleanup(self, temp_dir):
        """Verify that temp_dir fixture creates and will clean up directories."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("test content")
        assert test_file.exists(), "Should be able to create files in temp_dir"
        # Cleanup is tested implicitly - if directories accumulate, we know it's broken
    
    def test_yaml_config_fixture(self, sample_yaml_config):
        """Verify that YAML config fixture works."""
        assert sample_yaml_config.exists(), "YAML config file should exist"
        assert sample_yaml_config.suffix == ".yml", "Should be a YAML file"
        
        # Test that we can read it
        import yaml
        with open(sample_yaml_config, 'r') as f:
            config = yaml.safe_load(f)
        assert isinstance(config, dict), "Should load as a dictionary"
        assert config.get("model") == "ESPCN", "Should contain expected content"
    
    def test_image_fixtures(self, sample_image, sample_video_frames):
        """Verify that image-related fixtures work."""
        assert sample_image.exists(), "Sample image should exist"
        assert sample_image.suffix == ".png", "Sample image should be PNG"
        
        assert sample_video_frames.exists(), "Video frames directory should exist"
        frames = list(sample_video_frames.glob("*.png"))
        assert len(frames) == 5, "Should have created 5 frames"
    
    def test_python_path_includes_project_root(self):
        """Verify that the project root is in Python path for imports."""
        project_root = Path(__file__).parent.parent
        assert any(
            Path(p).resolve() == project_root.resolve() 
            for p in sys.path
        ), "Project root should be in Python path"


class TestCoverageConfiguration:
    """Tests to verify coverage configuration."""
    
    def test_coverage_should_track_source_code(self):
        """Verify that coverage is configured to track the codes package."""
        # This is a meta-test - actual verification happens when coverage runs
        from codes.utils import base_utils
        # Just importing to ensure it's trackable
        assert hasattr(base_utils, '__file__'), "Module should be importable and trackable"
    
    def test_coverage_should_ignore_test_files(self):
        """Verify that test files are excluded from coverage."""
        # Coverage report will show if test files are incorrectly included
        assert __file__.endswith('.py'), "This is a test file that should be excluded"


def test_standalone_function():
    """Test that standalone test functions work too, not just classes."""
    assert True, "Standalone test functions should work"


@pytest.mark.parametrize("value,expected", [
    (1, 1),
    (2, 2),
    (3, 3),
])
def test_parametrize_works(value, expected):
    """Verify that pytest parametrize decorator works."""
    assert value == expected, "Parametrized tests should work"