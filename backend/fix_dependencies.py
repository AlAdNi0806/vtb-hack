#!/usr/bin/env python3
"""
Fix dependency compatibility issues
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(cmd):
    """Run a shell command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Command failed: {cmd}")
            logger.error(f"Error: {result.stderr}")
            return False
        logger.info(f"Command succeeded: {cmd}")
        return True
    except Exception as e:
        logger.error(f"Exception running command {cmd}: {e}")
        return False

def fix_pyarrow_compatibility():
    """Fix PyArrow compatibility issues"""
    logger.info("Fixing PyArrow compatibility...")
    
    # Downgrade PyArrow to a compatible version
    commands = [
        "pip install pyarrow==12.0.1",
        "pip install datasets==2.14.0",
    ]
    
    for cmd in commands:
        if not run_command(cmd):
            logger.error(f"Failed to run: {cmd}")
            return False
    
    return True

def install_optional_dependencies():
    """Install optional dependencies that might be missing"""
    logger.info("Installing optional dependencies...")
    
    commands = [
        "pip install pydub",
        "pip install pygame",
        "pip install portaudio",
    ]
    
    for cmd in commands:
        run_command(cmd)  # Don't fail if these don't install

def main():
    """Main function to fix dependencies"""
    logger.info("Starting dependency fixes...")
    
    # Fix PyArrow compatibility
    if not fix_pyarrow_compatibility():
        logger.error("Failed to fix PyArrow compatibility")
        sys.exit(1)
    
    # Install optional dependencies
    install_optional_dependencies()
    
    logger.info("Dependency fixes completed!")
    logger.info("You can now try running: python main.py")

if __name__ == "__main__":
    main()
