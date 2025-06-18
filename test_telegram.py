#!/usr/bin/env python3
import sys
import os

print("Python path:")
for path in sys.path:
    print(f"  {path}")

print("\nTrying to import telegram module:")
try:
    import telegram
    print(f"telegram module location: {telegram.__file__}")
    print(f"telegram module attributes: {dir(telegram)}")
except Exception as e:
    print(f"Error importing telegram: {e}")

print("\nTrying to import Update from telegram:")
try:
    from telegram import Update
    print("Successfully imported Update")
except Exception as e:
    print(f"Error importing Update: {e}")

print("\nTrying to import telegram.ext:")
try:
    from telegram import ext
    print("Successfully imported telegram.ext")
    print(f"ext attributes: {dir(ext)}")
except Exception as e:
    print(f"Error importing telegram.ext: {e}")