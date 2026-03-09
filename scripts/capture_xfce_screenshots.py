#!/usr/bin/env python3
"""
Automated XFCE screenshot capture.

Launches a sequence of XFCE applications and navigates to diverse UI states
(dialogs, menus, save prompts, settings panels) to produce training data
for YOLO fine-tuning on desktop UI elements.

Requirements: xdotool, scrot, and an active XFCE X session ($DISPLAY set).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


@dataclass
class ScenarioStep:
    description: str
    commands: List[str]
    wait_after: float = 1.5


@dataclass
class Scenario:
    name: str
    launch_cmd: str
    steps: List[ScenarioStep] = field(default_factory=list)
    cleanup_cmd: Optional[str] = None


SCENARIOS: List[Scenario] = [
    # 1) Text editor - new file + save-as dialog
    Scenario(
        name="mousepad_save_as",
        launch_cmd="mousepad &",
        steps=[
            ScenarioStep("Type sample text", [
                "xdotool type --delay 40 'Hello World test file'",
            ], wait_after=1.0),
            ScenarioStep("Open Save As dialog (Ctrl+Shift+S)", [
                "xdotool key ctrl+shift+s",
            ], wait_after=2.0),
        ],
        cleanup_cmd="pkill -f mousepad || true",
    ),
    # 2) File manager - browse view with icons
    Scenario(
        name="thunar_home",
        launch_cmd="thunar ~ &",
        steps=[
            ScenarioStep("Wait for Thunar to load", [], wait_after=2.0),
        ],
        cleanup_cmd="pkill -f thunar || true",
    ),
    # 3) File manager - right-click context menu
    Scenario(
        name="thunar_context_menu",
        launch_cmd="thunar ~ &",
        steps=[
            ScenarioStep("Wait for Thunar", [], wait_after=2.0),
            ScenarioStep("Right-click center of screen", [
                "xdotool mousemove 600 400 click 3",
            ], wait_after=1.5),
        ],
        cleanup_cmd="pkill -f thunar || true",
    ),
    # 4) XFCE Settings Manager
    Scenario(
        name="xfce_settings",
        launch_cmd="xfce4-settings-manager &",
        steps=[
            ScenarioStep("Wait for Settings to load", [], wait_after=2.5),
        ],
        cleanup_cmd="pkill -f xfce4-settings-manager || true",
    ),
    # 5) Terminal emulator
    Scenario(
        name="xfce_terminal",
        launch_cmd="xfce4-terminal &",
        steps=[
            ScenarioStep("Wait for terminal", [], wait_after=1.5),
            ScenarioStep("Type a command", [
                "xdotool type --delay 30 'ls -la'",
            ], wait_after=0.5),
        ],
        cleanup_cmd="pkill -f xfce4-terminal || true",
    ),
    # 6) Mousepad with menu bar open (Edit menu)
    Scenario(
        name="mousepad_edit_menu",
        launch_cmd="mousepad &",
        steps=[
            ScenarioStep("Wait for Mousepad", [], wait_after=1.5),
            ScenarioStep("Open Edit menu", [
                "xdotool key alt+e",
            ], wait_after=1.0),
        ],
        cleanup_cmd="pkill -f mousepad || true",
    ),
    # 7) XFCE appearance settings (checkboxes, dropdowns)
    Scenario(
        name="xfce_appearance",
        launch_cmd="xfce4-appearance-settings &",
        steps=[
            ScenarioStep("Wait for appearance dialog", [], wait_after=2.0),
        ],
        cleanup_cmd="pkill -f xfce4-appearance || true",
    ),
    # 8) Thunar rename dialog (F2 on a file)
    Scenario(
        name="thunar_rename",
        launch_cmd="thunar ~ &",
        steps=[
            ScenarioStep("Wait for Thunar", [], wait_after=2.0),
            ScenarioStep("Click on first item area and press F2", [
                "xdotool mousemove 400 300 click 1",
            ], wait_after=0.8),
            ScenarioStep("Press F2 to rename", [
                "xdotool key F2",
            ], wait_after=1.5),
        ],
        cleanup_cmd="pkill -f thunar || true",
    ),
    # 9) Mousepad Find/Replace dialog
    Scenario(
        name="mousepad_find_replace",
        launch_cmd="mousepad &",
        steps=[
            ScenarioStep("Type some text first", [
                "xdotool type --delay 30 'search target text'",
            ], wait_after=0.8),
            ScenarioStep("Open Find and Replace (Ctrl+H)", [
                "xdotool key ctrl+h",
            ], wait_after=1.5),
        ],
        cleanup_cmd="pkill -f mousepad || true",
    ),
    # 10) Clean desktop (panel, wallpaper only)
    Scenario(
        name="clean_desktop",
        launch_cmd="true",
        steps=[
            ScenarioStep("Minimize all windows", [
                "wmctrl -k on 2>/dev/null || xdotool key super+d || true",
            ], wait_after=1.5),
        ],
    ),
]


def shell(cmd: str) -> None:
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def take_screenshot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["scrot", str(path)], check=True)


def run_scenario(scenario: Scenario, out_dir: Path, manifest_f, shot_idx: int) -> int:
    print(f"\n[capture] === Scenario: {scenario.name} ===")
    shell(scenario.launch_cmd)
    time.sleep(2.0)

    for step in scenario.steps:
        print(f"  -> {step.description}")
        for cmd in step.commands:
            shell(cmd)
        time.sleep(step.wait_after)

    filename = f"shot_{shot_idx:04d}_{scenario.name}.png"
    filepath = out_dir / filename
    take_screenshot(filepath)

    entry = {
        "image": filename,
        "abs_path": str(filepath.resolve()),
        "scenario": scenario.name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_f.write(json.dumps(entry) + "\n")
    print(f"  [captured] {filename}")

    if scenario.cleanup_cmd:
        time.sleep(0.3)
        shell(scenario.cleanup_cmd)
        time.sleep(0.5)

    return shot_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated XFCE screenshot capture for training data.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("data/raw/screenshots"),
        help="Output directory for PNG screenshots.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/raw/manifest.jsonl"),
        help="Manifest JSONL path.",
    )
    parser.add_argument(
        "--count", type=int, default=10,
        help="Max number of screenshots (caps at number of scenarios).",
    )
    parser.add_argument(
        "--scenarios", type=str, default="",
        help="Comma-separated scenario names to run (empty = all).",
    )
    args = parser.parse_args()

    display = os.environ.get("DISPLAY")
    if not display:
        print("[capture] ERROR: $DISPLAY not set. Run inside an XFCE X session.")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    if args.scenarios:
        selected_names = {s.strip() for s in args.scenarios.split(",")}
        scenarios = [s for s in SCENARIOS if s.name in selected_names]
    else:
        scenarios = SCENARIOS[: args.count]

    print(f"[capture] Will run {len(scenarios)} scenarios, output to {args.out_dir}")

    with args.manifest.open("a", encoding="utf-8") as mf:
        idx = 1
        for scenario in scenarios:
            idx = run_scenario(scenario, args.out_dir, mf, idx)

    print(f"\n[capture] Done. {len(scenarios)} screenshots in {args.out_dir}")
    print(f"[capture] Manifest: {args.manifest}")


if __name__ == "__main__":
    main()
