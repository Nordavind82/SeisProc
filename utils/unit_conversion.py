"""
Unit conversion utilities for spatial measurements.

All internal calculations use METERS as the base unit.
Display values are converted based on user preferences.
"""
from typing import Union, Optional
from models.app_settings import get_settings, AppSettings

# Conversion constants
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 1.0 / METERS_TO_FEET


class UnitConverter:
    """
    Handles conversion between meters and feet for spatial measurements.

    Internal storage: Always in METERS
    Display: Based on user settings (meters or feet)
    """

    @staticmethod
    def meters_to_feet(value_m: Union[float, int]) -> float:
        """
        Convert meters to feet.

        Args:
            value_m: Value in meters

        Returns:
            Value in feet
        """
        return value_m * METERS_TO_FEET

    @staticmethod
    def feet_to_meters(value_ft: Union[float, int]) -> float:
        """
        Convert feet to meters.

        Args:
            value_ft: Value in feet

        Returns:
            Value in meters
        """
        return value_ft * FEET_TO_METERS

    @staticmethod
    def to_display_units(value_m: Union[float, int],
                         units: Optional[str] = None) -> float:
        """
        Convert from meters (internal) to display units.

        Args:
            value_m: Value in meters (internal representation)
            units: Target units ('meters' or 'feet'). If None, uses app settings.

        Returns:
            Value in display units
        """
        if units is None:
            units = get_settings().get_spatial_units()

        if units == AppSettings.FEET:
            return UnitConverter.meters_to_feet(value_m)
        else:
            return value_m

    @staticmethod
    def from_display_units(value_display: Union[float, int],
                          units: Optional[str] = None) -> float:
        """
        Convert from display units to meters (internal).

        Args:
            value_display: Value in display units
            units: Source units ('meters' or 'feet'). If None, uses app settings.

        Returns:
            Value in meters (internal representation)
        """
        if units is None:
            units = get_settings().get_spatial_units()

        if units == AppSettings.FEET:
            return UnitConverter.feet_to_meters(value_display)
        else:
            return value_display

    @staticmethod
    def get_distance_label(units: Optional[str] = None) -> str:
        """
        Get the label for distance in the current units.

        Args:
            units: Units to use. If None, uses app settings.

        Returns:
            Label string (e.g., "meters", "feet", "m", "ft")
        """
        if units is None:
            units = get_settings().get_spatial_units()

        return units

    @staticmethod
    def get_distance_abbrev(units: Optional[str] = None) -> str:
        """
        Get the abbreviation for distance in the current units.

        Args:
            units: Units to use. If None, uses app settings.

        Returns:
            Abbreviation (e.g., "m", "ft")
        """
        if units is None:
            units = get_settings().get_spatial_units()

        return 'm' if units == AppSettings.METERS else 'ft'

    @staticmethod
    def get_velocity_label(units: Optional[str] = None) -> str:
        """
        Get the label for velocity in the current units.

        Args:
            units: Units to use. If None, uses app settings.

        Returns:
            Label string (e.g., "m/s", "ft/s")
        """
        abbrev = UnitConverter.get_distance_abbrev(units)
        return f"{abbrev}/s"

    @staticmethod
    def get_wavenumber_label(units: Optional[str] = None) -> str:
        """
        Get the label for wavenumber in the current units.

        Args:
            units: Units to use. If None, uses app settings.

        Returns:
            Label string (e.g., "cycles/m", "cycles/ft")
        """
        abbrev = UnitConverter.get_distance_abbrev(units)
        return f"cycles/{abbrev}"

    @staticmethod
    def get_dip_label(units: Optional[str] = None) -> str:
        """
        Get the label for dip in the current units.

        Args:
            units: Units to use. If None, uses app settings.

        Returns:
            Label string (e.g., "s/m", "s/ft")
        """
        abbrev = UnitConverter.get_distance_abbrev(units)
        return f"s/{abbrev}"

    @staticmethod
    def format_distance(value_m: Union[float, int],
                       decimals: int = 1,
                       units: Optional[str] = None,
                       show_units: bool = True) -> str:
        """
        Format distance value for display.

        Args:
            value_m: Value in meters (internal representation)
            decimals: Number of decimal places
            units: Target units. If None, uses app settings.
            show_units: Whether to append unit label

        Returns:
            Formatted string (e.g., "123.5 m" or "405.2 ft")
        """
        display_value = UnitConverter.to_display_units(value_m, units)
        formatted = f"{display_value:.{decimals}f}"

        if show_units:
            abbrev = UnitConverter.get_distance_abbrev(units)
            formatted += f" {abbrev}"

        return formatted

    @staticmethod
    def format_velocity(value_ms: Union[float, int],
                       decimals: int = 0,
                       units: Optional[str] = None,
                       show_units: bool = True) -> str:
        """
        Format velocity value for display.

        Args:
            value_ms: Value in m/s (internal representation)
            decimals: Number of decimal places
            units: Target units. If None, uses app settings.
            show_units: Whether to append unit label

        Returns:
            Formatted string (e.g., "1500 m/s" or "4921 ft/s")
        """
        display_value = UnitConverter.to_display_units(value_ms, units)
        formatted = f"{display_value:.{decimals}f}"

        if show_units:
            formatted += f" {UnitConverter.get_velocity_label(units)}"

        return formatted

    @staticmethod
    def convert_wavenumber(k_m: float, units: Optional[str] = None) -> float:
        """
        Convert wavenumber from cycles/m to display units.

        Wavenumber conversion:
        - k_feet = k_meters / METERS_TO_FEET

        Args:
            k_m: Wavenumber in cycles/meter
            units: Target units. If None, uses app settings.

        Returns:
            Wavenumber in cycles/[display unit]
        """
        if units is None:
            units = get_settings().get_spatial_units()

        if units == AppSettings.FEET:
            return k_m / METERS_TO_FEET
        else:
            return k_m


# Convenience functions for common conversions
def m_to_ft(meters: float) -> float:
    """Convert meters to feet."""
    return UnitConverter.meters_to_feet(meters)


def ft_to_m(feet: float) -> float:
    """Convert feet to meters."""
    return UnitConverter.feet_to_meters(feet)


def format_distance(value_m: float, decimals: int = 1,
                   units: Optional[str] = None) -> str:
    """Format distance for display with units."""
    return UnitConverter.format_distance(value_m, decimals, units, show_units=True)


def format_velocity(value_ms: float, decimals: int = 0,
                   units: Optional[str] = None) -> str:
    """Format velocity for display with units."""
    return UnitConverter.format_velocity(value_ms, decimals, units, show_units=True)
