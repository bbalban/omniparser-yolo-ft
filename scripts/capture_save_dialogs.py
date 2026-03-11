#!/usr/bin/env python3
"""Capture save/close dialog screenshots for shots 7-10."""

import subprocess
import time
from pathlib import Path

OUT = Path("data/raw/screenshots")
OUT.mkdir(parents=True, exist_ok=True)


def shell(cmd):
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def screenshot(path):
    subprocess.run(["scrot", str(path)], check=True)


def cleanup_all():
    shell("pkill -f gedit 2>/dev/null; pkill -f mousepad 2>/dev/null")
    time.sleep(1)


def capture(idx, name, steps):
    print(f"\n=== Scenario {idx}: {name} ===")
    cleanup_all()

    for desc, cmd, wait in steps:
        print(f"  -> {desc}")
        shell(cmd)
        time.sleep(wait)

    fname = f"shot_{idx:04d}_{name}.png"
    fpath = OUT / fname
    screenshot(fpath)
    print(f"  [captured] {fname} ({fpath.stat().st_size} bytes)")
    cleanup_all()


if __name__ == "__main__":
    # 7) Gedit Save As dialog
    capture(7, "gedit_save_as", [
        ("Launch gedit", "gedit &", 3.0),
        ("Type content", "xdotool type --delay 40 'This is a test document for training data.'", 1.5),
        ("Ctrl+Shift+S save-as", "xdotool key ctrl+shift+s", 2.5),
    ])

    # 8) Mousepad Save dialog
    capture(8, "mousepad_save_dialog", [
        ("Launch mousepad", "mousepad &", 2.0),
        ("Type content", "xdotool type --delay 40 'Important document content here.'", 1.5),
        ("Ctrl+S save", "xdotool key ctrl+s", 2.5),
    ])

    # 9) Gedit close-without-saving
    capture(9, "gedit_close_unsaved", [
        ("Launch gedit", "gedit &", 3.0),
        ("Type content", "xdotool type --delay 40 'Unsaved changes in this file.'", 1.5),
        ("Alt+F4 close", "xdotool key alt+F4", 2.5),
    ])

    # 10) Mousepad close-without-saving
    capture(10, "mousepad_close_unsaved", [
        ("Launch mousepad", "mousepad &", 2.0),
        ("Type content", "xdotool type --delay 40 'Data that has not been saved yet.'", 1.5),
        ("Alt+F4 close", "xdotool key alt+F4", 2.5),
    ])

    print("\nAll 4 screenshots captured!")
