#!/usr/bin/env python3
"""
EEG Backend Development Helper
=============================

Development utilities for working with the EEG backend:
- Environment info
- Dependency management
- Quick testing
- Server management
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout, e.stderr, e.returncode

def check_environment():
    """Check the development environment"""
    print("🔍 Environment Check")
    print("=" * 20)
    
    # Check uv
    stdout, stderr, code = run_command("uv --version", check=False)
    if code == 0:
        print(f"✅ uv: {stdout}")
    else:
        print("❌ uv not found")
        return False
    
    # Check Python via uv
    stdout, stderr, code = run_command("uv run python --version", check=False)
    if code == 0:
        print(f"✅ Python (via uv): {stdout}")
    else:
        print("❌ Python not available via uv")
        return False
    
    return True

def show_dependencies():
    """Show current dependencies"""
    print("\n📦 Dependencies")
    print("=" * 15)
    
    stdout, stderr, code = run_command("uv tree", check=False)
    if code == 0:
        print(stdout)
    else:
        print("❌ Could not get dependency tree")

def quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Quick Test")
    print("=" * 12)
    
    # Test imports
    test_code = """
import fastapi
import uvicorn
import numpy
import pandas
import scipy
print("✅ All core imports successful")
"""
    
    stdout, stderr, code = run_command(f'uv run python -c "{test_code}"', check=False)
    if code == 0:
        print(stdout)
    else:
        print(f"❌ Import test failed: {stderr}")

def show_project_info():
    """Show project information"""
    print("🧠 EEG Backend Project Info")
    print("=" * 28)
    
    # Read pyproject.toml
    try:
        with open("pyproject.toml", "r") as f:
            content = f.read()
            
        lines = content.split('\n')
        for line in lines[:10]:  # Show first 10 lines
            if line.strip() and not line.startswith('#'):
                print(f"  {line}")
        print("  ...")
        
    except FileNotFoundError:
        print("❌ pyproject.toml not found")

def main():
    """Main development helper"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "env":
            check_environment()
        elif command == "deps":
            show_dependencies()
        elif command == "test":
            quick_test()
        elif command == "sync":
            print("📦 Syncing dependencies...")
            stdout, stderr, code = run_command("uv sync")
            if code == 0:
                print("✅ Dependencies synced")
            else:
                print(f"❌ Sync failed: {stderr}")
        elif command == "server":
            print("🚀 Starting development server...")
            print("   Server: http://localhost:8000")
            print("   Docs: http://localhost:8000/docs")
            print("   Press Ctrl+C to stop")
            print()
            os.system("uv run uvicorn main:app --reload --host 127.0.0.1 --port 8000")
        else:
            print(f"Unknown command: {command}")
            print("Available commands: env, deps, test, sync, server")
    else:
        show_project_info()
        print()
        check_environment()
        print()
        print("💡 Usage:")
        print("  python dev.py env     - Check environment")
        print("  python dev.py deps    - Show dependencies")  
        print("  python dev.py test    - Quick functionality test")
        print("  python dev.py sync    - Sync dependencies")
        print("  python dev.py server  - Start development server")

if __name__ == "__main__":
    main()
