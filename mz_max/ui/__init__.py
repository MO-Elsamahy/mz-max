"""
User Interface module for MZ Max

This module provides user-friendly interfaces including web dashboards,
desktop GUI applications, and interactive widgets.
"""

# Import with error handling
try:
    from .dashboard import create_dashboard
    dashboard_available = True
except ImportError as e:
    print(f"Warning: Dashboard not available: {e}")
    create_dashboard = None
    dashboard_available = False

try:
    from .web_app import launch_web_app
    webapp_available = True
except ImportError as e:
    print(f"Warning: Web app not available: {e}")
    launch_web_app = None
    webapp_available = False

try:
    from .gui import launch_gui
    gui_available = True
except ImportError as e:
    print(f"Warning: GUI not available: {e}")
    launch_gui = None
    gui_available = False

try:
    from .widgets import DataExplorationWidget, AutoMLWidget, PredictionWidget
    widgets_available = True
except ImportError as e:
    print(f"Warning: Widgets not available: {e}")
    DataExplorationWidget = AutoMLWidget = PredictionWidget = None
    widgets_available = False

__all__ = [
    'create_dashboard',
    'launch_web_app', 
    'launch_gui',
    'DataExplorationWidget',
    'AutoMLWidget',
    'PredictionWidget',
]