#!/usr/bin/env python3
"""Fetch the current weather from the Open-Meteo API and print it.

No API key is required. Optionally pass a latitude and longitude as
command-line arguments, e.g.:

    python weather.py 51.5072 -0.1276
"""

import json
import sys
import urllib.parse
import urllib.request

DEFAULT_LATITUDE = 51.5072
DEFAULT_LONGITUDE = -0.1276

API_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(latitude, longitude):
    """Return the current weather data for the given coordinates."""
    params = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "current_weather": "true",
            "timezone": "auto",
        }
    )
    url = f"{API_URL}?{params}"

    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def print_weather(weather):
    """Print a short human-readable summary of the weather."""
    current = weather.get("current_weather", {})
    temperature = current.get("temperature")
    windspeed = current.get("windspeed")
    winddirection = current.get("winddirection")

    if temperature is None:
        print("No weather data available.")
        return

    print(f"Temperature: {temperature}°C")
    if windspeed is not None:
        print(f"Wind speed: {windspeed} km/h")
    if winddirection is not None:
        print(f"Wind direction: {winddirection}°")


def main():
    args = sys.argv[1:]
    try:
        if len(args) >= 2:
            latitude = float(args[0])
            longitude = float(args[1])
        else:
            latitude = DEFAULT_LATITUDE
            longitude = DEFAULT_LONGITUDE
    except ValueError:
        print("Usage: python weather.py [latitude longitude]", file=sys.stderr)
        sys.exit(1)

    try:
        weather = fetch_weather(latitude, longitude)
    except Exception as error:
        print(f"Failed to fetch weather: {error}", file=sys.stderr)
        sys.exit(1)

    print_weather(weather)


if __name__ == "__main__":
    main()
