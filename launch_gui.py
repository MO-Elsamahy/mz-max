#!/usr/bin/env python3
"""
Launch MZ Max Professional Desktop GUI

This script launches the professional desktop GUI application 
with modern styling and comprehensive ML features.
"""

import sys
import subprocess
from pathlib import Path
import os

def main():
    """Launch the MZ Max professional desktop GUI."""
    print("🖥️ Launching MZ Max Professional Desktop GUI...")
    print("=" * 60)
    
    try:
        # Check for optional dependencies
        missing_deps = []
        
        try:
            import matplotlib
        except ImportError:
            missing_deps.append("matplotlib")
            
        try:
            import seaborn
        except ImportError:
            missing_deps.append("seaborn")
            
        if missing_deps:
            print(f"📦 Installing visualization dependencies: {', '.join(missing_deps)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_deps
            ])
            print("✅ Visualization dependencies installed!")
        
        # Add the package to Python path
        package_path = Path(__file__).parent
        if str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))
        
        print("🚀 Starting professional desktop application...")
        print("💻 Professional features:")
        print("  ✅ Modern native desktop interface")
        print("  ✅ Multi-tab workflow organization")
        print("  ✅ Interactive data exploration")
        print("  ✅ Real-time machine learning training")
        print("  ✅ Advanced data visualization")
        print("  ✅ Enterprise security center")
        print("  ✅ Professional styling and themes")
        print("  ✅ Background task processing")
        
        print("\n🎨 Interface highlights:")
        print("  • Professional color scheme")
        print("  • Responsive layout design")
        print("  • Real-time progress tracking")
        print("  • Interactive matplotlib plots")
        print("  • Secure data encryption tools")
        
        print("\n💡 The professional GUI will open in a new window")
        print("⏹️  Close the window or press Ctrl+C to exit")
        print("=" * 60)
        
        # Import and launch GUI
        from mz_max.ui.gui import launch_gui
        launch_gui()
        
    except KeyboardInterrupt:
        print("\n👋 Desktop GUI stopped. Thank you for using MZ Max Professional!")
        print("🌟 Visit https://github.com/mzmax/mz-max for updates")
    except Exception as e:
        print(f"❌ Error launching desktop GUI: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure tkinter is installed (usually comes with Python)")
        print("  2. Install visualization: pip install matplotlib seaborn")
        print("  3. Check Python version (3.8+ recommended)")
        
        # Platform-specific advice
        if sys.platform == "linux":
            print("  4. Linux: sudo apt-get install python3-tk")
        elif sys.platform == "darwin":
            print("  4. macOS: tkinter should be included with Python")
        elif sys.platform == "win32":
            print("  4. Windows: tkinter should be included with Python")

if __name__ == "__main__":
    main()