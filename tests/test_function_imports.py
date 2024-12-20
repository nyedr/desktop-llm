import os
import sys
import json
import pytest
import logging
import importlib
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from functions.base import FunctionError, ValidationError, ModuleImportError

def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate the function configuration file."""
    try:
        logger.info(f"Loading config from: {config_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        if not isinstance(config, dict):
            raise ValidationError(f"Invalid config format. Expected dict, got: {type(config)}")
            
        if "functions" not in config:
            raise ValidationError("No 'functions' key in config")
            
        if not isinstance(config["functions"], list):
            raise ValidationError(f"Invalid functions format. Expected list, got: {type(config['functions'])}")
            
        logger.info(f"Successfully loaded config with {len(config['functions'])} functions")
        return config
        
    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in config file: {str(e)}")
    except Exception as e:
        raise FunctionError(f"Error loading config: {str(e)}")

def validate_function_config(func_config: Dict[str, Any]) -> None:
    """Validate a single function configuration."""
    required_fields = ["name", "module_path", "function_name", "type", "input_schema", "output_schema", "description"]
    missing_fields = [field for field in required_fields if field not in func_config]
    
    if missing_fields:
        raise ValidationError(f"Missing required fields in function config: {missing_fields}")
        
    if not isinstance(func_config["input_schema"], dict):
        raise ValidationError(f"Invalid input_schema format. Expected dict, got: {type(func_config['input_schema'])}")
        
    if not isinstance(func_config["output_schema"], dict):
        raise ValidationError(f"Invalid output_schema format. Expected dict, got: {type(func_config['output_schema'])}")

def import_function(module_path: str, function_name: str) -> Any:
    """Import a function from a module."""
    try:
        logger.debug(f"Attempting to import {function_name} from {module_path}")
        
        # First try direct import
        try:
            module = importlib.import_module(module_path)
            if not hasattr(module, function_name):
                raise AttributeError(f"Function {function_name} not found in module {module_path}")
                
            function = getattr(module, function_name)
            if not callable(function):
                raise TypeError(f"Object {function_name} in module {module_path} is not callable")
                
            logger.info(f"Successfully imported {function_name} from {module_path}")
            return function
            
        except ImportError as e:
            logger.warning(f"Direct import failed: {str(e)}")
            
            # Try file-based import
            module_parts = module_path.split('.')
            module_file = os.path.join(*module_parts) + '.py'
            abs_path = os.path.join(project_root, module_file)
            
            if not os.path.exists(abs_path):
                raise ModuleImportError(f"Module file not found: {abs_path}")
                
            spec = importlib.util.spec_from_file_location(module_path, abs_path)
            if spec is None or spec.loader is None:
                raise ModuleImportError(f"Failed to create module spec for {module_path}")
                
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_path] = module
            spec.loader.exec_module(module)
            
            if not hasattr(module, function_name):
                raise AttributeError(f"Function {function_name} not found in module {module_path}")
                
            function = getattr(module, function_name)
            if not callable(function):
                raise TypeError(f"Object {function_name} in module {module_path} is not callable")
                
            logger.info(f"Successfully imported {function_name} from {module_path} using file-based import")
            return function
            
    except Exception as e:
        raise ModuleImportError(f"Failed to import {function_name} from {module_path}: {str(e)}")

def test_load_config():
    """Test loading the function configuration file."""
    config_path = os.path.join(project_root, "functions", "config.json")
    config = load_config(config_path)
    assert "functions" in config
    assert isinstance(config["functions"], list)
    assert len(config["functions"]) > 0
    
    print("\nAvailable functions:")
    for func in config["functions"]:
        print(f"  - {func['name']} ({func['module_path']}.{func['function_name']})")

def test_validate_function_configs():
    """Test validating all function configurations."""
    config_path = os.path.join(project_root, "functions", "config.json")
    config = load_config(config_path)
    
    print("\nValidating functions:")
    for func_config in config["functions"]:
        print(f"  - Validating {func_config['name']}...")
        validate_function_config(func_config)
        print(f"    [OK] {func_config['name']} configuration is valid")

def test_import_functions():
    """Test importing all functions from the configuration."""
    config_path = os.path.join(project_root, "functions", "config.json")
    config = load_config(config_path)
    
    print("\nImporting functions:")
    for func_config in config["functions"]:
        print(f"  - Importing {func_config['name']}...")
        function = import_function(func_config["module_path"], func_config["function_name"])
        assert callable(function)
        print(f"    [OK] {func_config['name']} imported successfully")

def test_function_signatures():
    """Test that imported functions match their schema definitions."""
    config_path = os.path.join(project_root, "functions", "config.json")
    config = load_config(config_path)
    
    print("\nValidating function signatures:")
    for func_config in config["functions"]:
        print(f"  - Checking {func_config['name']} signature...")
        function = import_function(func_config["module_path"], func_config["function_name"])
        
        # Get function signature
        import inspect
        sig = inspect.signature(function)
        
        # Check required parameters
        required_params = {
            name for name, param in sig.parameters.items()
            if param.default == inspect.Parameter.empty and param.kind != inspect.Parameter.VAR_POSITIONAL
        }
        
        # Get required parameters from schema
        schema_required = set(func_config["input_schema"].get("required", []))
        
        # Print signature details
        print(f"    Function: {func_config['name']}")
        print(f"    Signature: {sig}")
        print(f"    Required params: {required_params}")
        print(f"    Schema required: {schema_required}")
        
        # Verify required parameters match
        assert required_params == schema_required, \
            f"Function {func_config['name']} signature does not match schema requirements"
        print(f"    [OK] Signature validation passed")

if __name__ == "__main__":
    # Run tests with more detailed output
    logging.basicConfig(level=logging.DEBUG)
    pytest.main([__file__, "-v"])
