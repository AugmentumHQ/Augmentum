"""Unit converter tool — convert between common measurement units."""

from __future__ import annotations

from augmentum.tools.base import Tool, ToolCategory, ToolResult

# Conversion factors to a base unit for each category
# Format: {unit_name: (factor_to_base, base_unit_name)}
_LENGTH = {
    "m": 1.0, "meter": 1.0, "meters": 1.0,
    "km": 1000.0, "kilometer": 1000.0, "kilometers": 1000.0,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "um": 1e-6, "micrometer": 1e-6, "micrometers": 1e-6,
    "nm": 1e-9, "nanometer": 1e-9, "nanometers": 1e-9,
    "mi": 1609.344, "mile": 1609.344, "miles": 1609.344,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254,
    "nmi": 1852.0, "nautical_mile": 1852.0,
    "ly": 9.461e15, "light_year": 9.461e15,
    "au": 1.496e11, "astronomical_unit": 1.496e11,
}

_MASS = {
    "kg": 1.0, "kilogram": 1.0, "kilograms": 1.0,
    "g": 0.001, "gram": 0.001, "grams": 0.001,
    "mg": 1e-6, "milligram": 1e-6, "milligrams": 1e-6,
    "lb": 0.453592, "pound": 0.453592, "pounds": 0.453592,
    "oz": 0.0283495, "ounce": 0.0283495, "ounces": 0.0283495,
    "t": 1000.0, "tonne": 1000.0, "metric_ton": 1000.0,
    "ton": 907.185, "short_ton": 907.185,
    "st": 6.35029, "stone": 6.35029,
}

_VOLUME = {
    "l": 1.0, "liter": 1.0, "liters": 1.0, "litre": 1.0, "litres": 1.0,
    "ml": 0.001, "milliliter": 0.001, "milliliters": 0.001,
    "gal": 3.78541, "gallon": 3.78541, "gallons": 3.78541,
    "qt": 0.946353, "quart": 0.946353, "quarts": 0.946353,
    "pt": 0.473176, "pint": 0.473176, "pints": 0.473176,
    "cup": 0.236588, "cups": 0.236588,
    "fl_oz": 0.0295735, "fluid_ounce": 0.0295735,
    "tbsp": 0.0147868, "tablespoon": 0.0147868,
    "tsp": 0.00492892, "teaspoon": 0.00492892,
    "m3": 1000.0, "cubic_meter": 1000.0,
    "cm3": 0.001, "cubic_centimeter": 0.001, "cc": 0.001,
}

_SPEED = {
    "m/s": 1.0, "mps": 1.0,
    "km/h": 0.277778, "kph": 0.277778, "kmh": 0.277778,
    "mph": 0.44704, "mi/h": 0.44704,
    "ft/s": 0.3048, "fps": 0.3048,
    "knot": 0.514444, "knots": 0.514444, "kn": 0.514444,
    "mach": 343.0,
    "c": 299792458.0, "light_speed": 299792458.0,
}

_TIME = {
    "s": 1.0, "second": 1.0, "seconds": 1.0, "sec": 1.0,
    "ms": 0.001, "millisecond": 0.001, "milliseconds": 0.001,
    "us": 1e-6, "microsecond": 1e-6, "microseconds": 1e-6,
    "ns": 1e-9, "nanosecond": 1e-9, "nanoseconds": 1e-9,
    "min": 60.0, "minute": 60.0, "minutes": 60.0,
    "h": 3600.0, "hour": 3600.0, "hours": 3600.0, "hr": 3600.0,
    "d": 86400.0, "day": 86400.0, "days": 86400.0,
    "week": 604800.0, "weeks": 604800.0, "wk": 604800.0,
    "month": 2592000.0, "months": 2592000.0,  # 30 days approx
    "year": 31557600.0, "years": 31557600.0, "yr": 31557600.0,  # 365.25 days
}

_AREA = {
    "m2": 1.0, "sq_m": 1.0, "square_meter": 1.0,
    "km2": 1e6, "sq_km": 1e6, "square_kilometer": 1e6,
    "cm2": 1e-4, "sq_cm": 1e-4, "square_centimeter": 1e-4,
    "ha": 10000.0, "hectare": 10000.0, "hectares": 10000.0,
    "acre": 4046.86, "acres": 4046.86,
    "sq_ft": 0.092903, "square_foot": 0.092903, "ft2": 0.092903,
    "sq_mi": 2.59e6, "square_mile": 2.59e6, "mi2": 2.59e6,
    "sq_in": 0.00064516, "square_inch": 0.00064516, "in2": 0.00064516,
    "sq_yd": 0.836127, "square_yard": 0.836127, "yd2": 0.836127,
}

_DATA = {
    "b": 1.0, "bit": 1.0, "bits": 1.0,
    "B": 8.0, "byte": 8.0, "bytes": 8.0,
    "kb": 1000.0, "kilobit": 1000.0,
    "kB": 8000.0, "kilobyte": 8000.0, "KB": 8000.0,
    "mb": 1e6, "megabit": 1e6, "Mb": 1e6,
    "MB": 8e6, "megabyte": 8e6,
    "gb": 1e9, "gigabit": 1e9, "Gb": 1e9,
    "GB": 8e9, "gigabyte": 8e9,
    "tb": 1e12, "terabit": 1e12, "Tb": 1e12,
    "TB": 8e12, "terabyte": 8e12,
    "KiB": 8192.0, "kibibyte": 8192.0,
    "MiB": 8388608.0, "mebibyte": 8388608.0,
    "GiB": 8589934592.0, "gibibyte": 8589934592.0,
    "TiB": 8796093022208.0, "tebibyte": 8796093022208.0,
}

_CATEGORIES: dict[str, dict[str, float]] = {
    "length": _LENGTH,
    "mass": _MASS,
    "volume": _VOLUME,
    "speed": _SPEED,
    "time": _TIME,
    "area": _AREA,
    "data": _DATA,
}


def _find_category(unit: str) -> tuple[str, dict[str, float]] | None:
    """Find which category a unit belongs to."""
    for cat_name, cat_units in _CATEGORIES.items():
        if unit in cat_units:
            return cat_name, cat_units
    return None


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Handle temperature conversions separately (non-linear)."""
    # Normalize unit names
    from_u = from_unit.lower().strip("°")
    to_u = to_unit.lower().strip("°")

    # Convert to Celsius first
    if from_u in ("c", "celsius"):
        celsius = value
    elif from_u in ("f", "fahrenheit"):
        celsius = (value - 32) * 5 / 9
    elif from_u in ("k", "kelvin"):
        celsius = value - 273.15
    elif from_u in ("r", "rankine"):
        celsius = (value - 491.67) * 5 / 9
    else:
        raise ValueError(f"Unknown temperature unit: {from_unit}")

    # Convert from Celsius to target
    if to_u in ("c", "celsius"):
        return celsius
    if to_u in ("f", "fahrenheit"):
        return celsius * 9 / 5 + 32
    if to_u in ("k", "kelvin"):
        return celsius + 273.15
    if to_u in ("r", "rankine"):
        return (celsius + 273.15) * 9 / 5
    raise ValueError(f"Unknown temperature unit: {to_unit}")


_TEMP_UNITS = {"c", "celsius", "f", "fahrenheit", "k", "kelvin", "r", "rankine", "°c", "°f", "°k"}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a value between units."""
    from_lower = from_unit.lower()
    to_lower = to_unit.lower()

    # Check temperature first (non-linear)
    if from_lower in _TEMP_UNITS or to_lower in _TEMP_UNITS:
        return _convert_temperature(value, from_unit, to_unit)

    # Find categories
    from_cat = _find_category(from_unit)
    to_cat = _find_category(to_unit)

    if from_cat is None:
        # Try lowercase
        from_cat = _find_category(from_lower)
        from_unit = from_lower
    if to_cat is None:
        to_cat = _find_category(to_lower)
        to_unit = to_lower

    if from_cat is None:
        raise ValueError(f"Unknown unit: {from_unit}")
    if to_cat is None:
        raise ValueError(f"Unknown unit: {to_unit}")

    if from_cat[0] != to_cat[0]:
        raise ValueError(
            f"Cannot convert between {from_cat[0]} ({from_unit}) and {to_cat[0]} ({to_unit})"
        )

    from_factor = from_cat[1][from_unit]
    to_factor = to_cat[1][to_unit]

    # Convert: value * from_factor gives base units, divide by to_factor
    return value * from_factor / to_factor


class UnitConverterTool(Tool):
    """Convert between measurement units across categories."""

    @property
    def name(self) -> str:
        return "unit_converter"

    @property
    def description(self) -> str:
        return (
            "Convert between units of measurement. Supports: length, mass, volume, "
            "speed, time, area, data/storage, and temperature. "
            "Example: convert 100 km to miles, or 72 °F to °C."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.VERIFY

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "value": {"type": "number", "description": "The value to convert"},
                "from_unit": {"type": "string", "description": "Source unit (e.g. 'km', 'lb', '°F')"},
                "to_unit": {"type": "string", "description": "Target unit (e.g. 'mi', 'kg', '°C')"},
            },
            "required": ["value", "from_unit", "to_unit"],
        }

    async def execute(self, **kwargs) -> ToolResult:
        value = kwargs.get("value")
        from_unit = kwargs.get("from_unit", "")
        to_unit = kwargs.get("to_unit", "")

        if value is None:
            return ToolResult(success=False, error="No value provided")
        if not from_unit or not to_unit:
            return ToolResult(success=False, error="Both from_unit and to_unit are required")

        try:
            result = convert(float(value), from_unit, to_unit)
            return ToolResult(
                success=True,
                output=f"{value} {from_unit} = {result} {to_unit}",
                metadata={
                    "value": value,
                    "from_unit": from_unit,
                    "to_unit": to_unit,
                    "result": result,
                },
            )
        except (ValueError, ZeroDivisionError) as e:
            return ToolResult(success=False, error=str(e))
