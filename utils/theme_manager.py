"""
Theme Manager for SeisProc application.

Provides light and dark theme support with consistent styling across all views.
"""
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from typing import Literal

ThemeType = Literal['light', 'dark', 'system']


class ThemeManager:
    """
    Manages application themes (light/dark mode).

    Provides consistent color schemes and stylesheets for the entire application.
    Theme preference is persisted via AppSettings (JSON file).
    """

    # Light theme colors
    LIGHT_COLORS = {
        'window': '#f5f5f5',
        'window_text': '#1a1a1a',
        'base': '#ffffff',
        'alternate_base': '#f0f0f0',
        'text': '#1a1a1a',
        'button': '#e0e0e0',
        'button_text': '#1a1a1a',
        'highlight': '#3498db',
        'highlight_text': '#ffffff',
        'link': '#2980b9',
        'border': '#cccccc',
        'disabled_text': '#808080',
        'plot_background': '#ffffff',
        'grid': '#e0e0e0',
    }

    # Dark theme colors
    DARK_COLORS = {
        'window': '#2b2b2b',
        'window_text': '#e0e0e0',
        'base': '#1e1e1e',
        'alternate_base': '#353535',
        'text': '#e0e0e0',
        'button': '#404040',
        'button_text': '#e0e0e0',
        'highlight': '#3498db',
        'highlight_text': '#ffffff',
        'link': '#5dade2',
        'border': '#555555',
        'disabled_text': '#707070',
        'plot_background': '#1e1e1e',
        'grid': '#404040',
    }

    _instance = None
    _current_theme: ThemeType = 'light'

    def __new__(cls):
        """Singleton pattern - only one theme manager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize theme manager."""
        if self._initialized:
            return
        self._initialized = True
        self._load_theme_preference()

    def _load_theme_preference(self):
        """Load theme preference from settings."""
        try:
            from models.app_settings import get_settings
            settings = get_settings()
            session = settings.get_session_state()
            self._current_theme = session.get('theme', 'light')
        except Exception:
            self._current_theme = 'light'

        if self._current_theme not in ('light', 'dark', 'system'):
            self._current_theme = 'light'

    def _save_theme_preference(self):
        """Save theme preference to settings."""
        try:
            from models.app_settings import get_settings
            settings = get_settings()
            session = settings.get_session_state()
            session['theme'] = self._current_theme
            settings.save_session_state(session)
        except Exception:
            pass  # Fail silently if settings unavailable

    @property
    def current_theme(self) -> ThemeType:
        """Get current theme name."""
        return self._current_theme

    @property
    def colors(self) -> dict:
        """Get current theme colors."""
        if self._current_theme == 'dark':
            return self.DARK_COLORS
        return self.LIGHT_COLORS

    @property
    def is_dark(self) -> bool:
        """Check if current theme is dark."""
        return self._current_theme == 'dark'

    def set_theme(self, theme: ThemeType):
        """
        Set application theme.

        Args:
            theme: 'light', 'dark', or 'system'
        """
        if theme not in ('light', 'dark', 'system'):
            raise ValueError(f"Invalid theme: {theme}")

        self._current_theme = theme
        self._save_theme_preference()
        self._apply_theme()

    def toggle_theme(self):
        """Toggle between light and dark themes."""
        new_theme = 'light' if self._current_theme == 'dark' else 'dark'
        self.set_theme(new_theme)

    def _apply_theme(self):
        """Apply current theme to application."""
        app = QApplication.instance()
        if app is None:
            return

        # Apply palette
        palette = self._create_palette()
        app.setPalette(palette)

        # Apply stylesheet
        stylesheet = self._create_stylesheet()
        app.setStyleSheet(stylesheet)

    def _create_palette(self) -> QPalette:
        """Create QPalette for current theme."""
        colors = self.colors
        palette = QPalette()

        # Window colors
        palette.setColor(QPalette.ColorRole.Window, QColor(colors['window']))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors['window_text']))

        # Base colors (for text inputs, lists, etc.)
        palette.setColor(QPalette.ColorRole.Base, QColor(colors['base']))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors['alternate_base']))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors['text']))

        # Button colors
        palette.setColor(QPalette.ColorRole.Button, QColor(colors['button']))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors['button_text']))

        # Highlight colors
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors['highlight']))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors['highlight_text']))

        # Link color
        palette.setColor(QPalette.ColorRole.Link, QColor(colors['link']))

        # Disabled text
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                        QColor(colors['disabled_text']))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
                        QColor(colors['disabled_text']))

        return palette

    def _create_stylesheet(self) -> str:
        """Create stylesheet for current theme."""
        colors = self.colors

        return f"""
            /* Global styles */
            QMainWindow, QDialog {{
                background-color: {colors['window']};
            }}

            /* Group boxes */
            QGroupBox {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}

            /* Buttons */
            QPushButton {{
                background-color: {colors['button']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 5px 15px;
                min-height: 20px;
            }}
            QPushButton:hover {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
            QPushButton:pressed {{
                background-color: {colors['highlight']};
            }}
            QPushButton:disabled {{
                background-color: {colors['alternate_base']};
                color: {colors['disabled_text']};
            }}

            /* Combo boxes */
            QComboBox {{
                background-color: {colors['base']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QComboBox:hover {{
                border-color: {colors['highlight']};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}

            /* Spin boxes */
            QSpinBox, QDoubleSpinBox {{
                background-color: {colors['base']};
                border: 1px solid {colors['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {colors['highlight']};
            }}

            /* Sliders */
            QSlider::groove:horizontal {{
                border: 1px solid {colors['border']};
                height: 6px;
                background: {colors['alternate_base']};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {colors['highlight']};
                border: 1px solid {colors['highlight']};
                width: 14px;
                margin: -5px 0;
                border-radius: 7px;
            }}

            /* Checkboxes */
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 1px solid {colors['border']};
                border-radius: 3px;
                background-color: {colors['base']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {colors['highlight']};
                border-color: {colors['highlight']};
            }}

            /* Tabs */
            QTabWidget::pane {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
            }}
            QTabBar::tab {{
                background-color: {colors['button']};
                border: 1px solid {colors['border']};
                padding: 6px 12px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}

            /* Scroll bars */
            QScrollBar:vertical {{
                background: {colors['alternate_base']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {colors['button']};
                min-height: 20px;
                border-radius: 6px;
            }}
            QScrollBar:horizontal {{
                background: {colors['alternate_base']};
                height: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:horizontal {{
                background: {colors['button']};
                min-width: 20px;
                border-radius: 6px;
            }}

            /* Menu bar */
            QMenuBar {{
                background-color: {colors['window']};
                border-bottom: 1px solid {colors['border']};
            }}
            QMenuBar::item:selected {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}
            QMenu {{
                background-color: {colors['base']};
                border: 1px solid {colors['border']};
            }}
            QMenu::item:selected {{
                background-color: {colors['highlight']};
                color: {colors['highlight_text']};
            }}

            /* Status bar */
            QStatusBar {{
                background-color: {colors['window']};
                border-top: 1px solid {colors['border']};
            }}

            /* Tool tips */
            QToolTip {{
                background-color: {colors['base']};
                color: {colors['text']};
                border: 1px solid {colors['border']};
                padding: 4px;
            }}

            /* Progress bar */
            QProgressBar {{
                border: 1px solid {colors['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {colors['alternate_base']};
            }}
            QProgressBar::chunk {{
                background-color: {colors['highlight']};
                border-radius: 3px;
            }}
        """

    def apply_to_app(self):
        """Apply theme to the application (call after QApplication is created)."""
        self._apply_theme()

    def get_plot_colors(self) -> dict:
        """
        Get colors suitable for plots/graphs.

        Returns:
            Dictionary with 'background', 'foreground', 'grid' colors
        """
        colors = self.colors
        return {
            'background': colors['plot_background'],
            'foreground': colors['text'],
            'grid': colors['grid'],
        }


# Convenience function to get the singleton instance
def get_theme_manager() -> ThemeManager:
    """Get the ThemeManager singleton instance."""
    return ThemeManager()
