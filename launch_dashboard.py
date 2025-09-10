#!/usr/bin/env python3
"""
Launch MZ Max Professional Dashboard

This script launches the professional Streamlit dashboard for MZ Max
with enterprise-grade features and modern UI.
"""

import sys
import subprocess
from pathlib import Path
import webbrowser
import time

def main():
    """Launch the MZ Max professional dashboard."""
    print("🚀 Launching MZ Max Professional Dashboard...")
    print("=" * 60)
    
    try:
        # Check dependencies
        missing_deps = []
        
        try:
            import streamlit
        except ImportError:
            missing_deps.append("streamlit")
            
        try:
            import plotly
        except ImportError:
            missing_deps.append("plotly")
            
        if missing_deps:
            print(f"📦 Installing missing dependencies: {', '.join(missing_deps)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_deps, "seaborn", "matplotlib"
            ])
            print("✅ Dependencies installed successfully!")
        
        # Get the dashboard script path
        dashboard_script = Path(__file__).parent / "mz_max" / "ui" / "dashboard.py"
        
        if not dashboard_script.exists():
            print("❌ Dashboard script not found!")
            print(f"Expected location: {dashboard_script}")
            return
        
        print("🌐 Starting professional dashboard server...")
        print("📊 Dashboard features:")
        print("  ✅ Interactive data exploration")
        print("  ✅ Professional AutoML studio")
        print("  ✅ Enterprise security center")
        print("  ✅ Real-time analytics")
        print("  ✅ Modern responsive UI")
        
        print(f"\n🔗 Dashboard URL: http://localhost:8501")
        print("💡 The dashboard will open automatically in your browser")
        print("\n⏹️  Press Ctrl+C to stop the dashboard")
        print("=" * 60)
        
        # Launch streamlit with professional configuration
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_script),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false",
            "--theme.base=light",
            "--theme.primaryColor=#2E86AB",
            "--theme.backgroundColor=#FFFFFF",
            "--theme.secondaryBackgroundColor=#F5F5F5"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using MZ Max Professional!")
        print("🌟 Visit https://github.com/mzmax/mz-max for updates")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Install dependencies: pip install streamlit plotly seaborn")
        print("  2. Check Python version (3.8+ required)")
        print("  3. Ensure MZ Max is properly installed")

if __name__ == "__main__":
    main()
