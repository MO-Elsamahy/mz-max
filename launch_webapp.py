#!/usr/bin/env python3
"""
Launch MZ Max Professional Web Application

This script launches the enterprise-grade FastAPI web application 
with modern UI and comprehensive API features.
"""

import sys
import subprocess
from pathlib import Path
import webbrowser
import time
import threading

def main():
    """Launch the MZ Max professional web application."""
    print("🚀 Launching MZ Max Professional Web Application...")
    print("=" * 60)
    
    try:
        # Check dependencies
        missing_deps = []
        
        try:
            import fastapi
        except ImportError:
            missing_deps.append("fastapi")
            
        try:
            import uvicorn
        except ImportError:
            missing_deps.append("uvicorn")
            
        try:
            import jinja2
        except ImportError:
            missing_deps.append("jinja2")
            
        if missing_deps:
            print(f"📦 Installing missing dependencies: {', '.join(missing_deps)}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_deps, "python-multipart"
            ])
            print("✅ Dependencies installed successfully!")
        
        # Add the package to Python path
        package_path = Path(__file__).parent
        if str(package_path) not in sys.path:
            sys.path.insert(0, str(package_path))
        
        print("🌐 Starting enterprise web application server...")
        print("🏢 Professional features:")
        print("  ✅ Modern Bootstrap 5 web interface")
        print("  ✅ RESTful API with auto-documentation")
        print("  ✅ Enterprise security endpoints")
        print("  ✅ Real-time model serving")
        print("  ✅ File upload and processing")
        print("  ✅ Background job processing")
        print("  ✅ Comprehensive error handling")
        
        print(f"\n🔗 Web Interface: http://localhost:8000")
        print(f"📖 API Documentation: http://localhost:8000/docs")
        print(f"📚 ReDoc Documentation: http://localhost:8000/redoc")
        print(f"🔍 Health Check: http://localhost:8000/api/v1/health")
        
        print("\n💡 API Endpoints:")
        print("  POST /api/v1/data/load - Load datasets")
        print("  POST /api/v1/ml/predict - Make predictions")
        print("  POST /api/v1/ml/train - Train models")
        print("  POST /api/v1/security/encrypt - Encrypt data")
        print("  GET  /api/v1/info - System information")
        
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        # Launch browser after a short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open("http://localhost:8000")
                print("🌐 Web application opened in your browser")
            except:
                pass
        
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Import and launch the web app
        from mz_max.ui.web_app import launch_web_app
        launch_web_app(host="0.0.0.0", port=8000, reload=False)
        
    except KeyboardInterrupt:
        print("\n👋 Web application stopped. Thank you for using MZ Max Professional!")
        print("🌟 Visit https://github.com/mzmax/mz-max for updates and support")
    except Exception as e:
        print(f"❌ Error launching web application: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Install dependencies: pip install fastapi uvicorn jinja2")
        print("  2. Check if port 8000 is available")
        print("  3. Ensure MZ Max is properly installed")
        print("  4. Try running with different port: python -c \"from mz_max.ui.web_app import launch_web_app; launch_web_app(port=8001)\"")

if __name__ == "__main__":
    main()