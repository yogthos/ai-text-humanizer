"""Utility functions for text processing and length calculations."""

import json
from typing import Dict, Optional, Tuple


def calculate_length_ratio(text1: str, text2: str) -> float:
    """Calculate word count ratio between two texts.

    Args:
        text1: First text to compare.
        text2: Second text to compare (used as denominator).

    Returns:
        Ratio of text1 word count to text2 word count. Returns 1.0 if text2 is empty.
    """
    words1 = len(text1.split()) if text1 else 0
    words2 = len(text2.split()) if text2 else 0
    if words2 == 0:
        return 1.0
    return words1 / words2


def is_very_different_length(ratio: float, config: Optional[Dict] = None, config_path: str = "config.json") -> bool:
    """Check if length ratio indicates very different lengths.

    Args:
        ratio: Length ratio to check.
        config: Optional config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).

    Returns:
        True if ratio indicates very different lengths (outside thresholds).
    """
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)

    length_config = config.get("length_gate", {})
    threshold_low = length_config.get("very_different_threshold_low", 0.5)
    threshold_high = length_config.get("very_different_threshold_high", 2.0)
    return ratio < threshold_low or ratio > threshold_high


def is_moderate_different_length(ratio: float, config: Optional[Dict] = None, config_path: str = "config.json") -> bool:
    """Check if length ratio indicates moderately different lengths.

    Args:
        ratio: Length ratio to check.
        config: Optional config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).

    Returns:
        True if ratio indicates moderately different lengths.
    """
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)

    length_config = config.get("length_gate", {})
    threshold_low = length_config.get("moderate_different_threshold_low", 0.67)
    threshold_high = length_config.get("moderate_different_threshold_high", 1.5)
    return (ratio < threshold_low or ratio > threshold_high) and not is_very_different_length(ratio, config, config_path)


def get_length_gate_ratios(config: Optional[Dict] = None, config_path: str = "config.json", use_lenient: bool = False) -> Tuple[float, float]:
    """Get min and max ratios for length gate.

    Args:
        config: Optional config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).
        use_lenient: If True, return lenient ratios; otherwise return default ratios.

    Returns:
        Tuple of (min_ratio, max_ratio).
    """
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)

    length_config = config.get("length_gate", {})
    if use_lenient:
        min_ratio = length_config.get("lenient_min_ratio", 0.2)
        max_ratio = length_config.get("lenient_max_ratio", 3.0)
    else:
        min_ratio = length_config.get("default_min_ratio", 0.6)
        max_ratio = length_config.get("default_max_ratio", 1.5)

    return min_ratio, max_ratio


def should_skip_length_gate(structure_input_ratio: float, config: Optional[Dict] = None, config_path: str = "config.json") -> bool:
    """Check if length gate should be skipped based on structure-input ratio.

    Args:
        structure_input_ratio: Ratio of structure match length to input length.
        config: Optional config dict. If None, loads from config_path.
        config_path: Path to config file (used if config is None).

    Returns:
        True if length gate should be skipped.
    """
    if config is None:
        with open(config_path, 'r') as f:
            config = json.load(f)

    length_config = config.get("length_gate", {})
    skip_when_very_different = length_config.get("skip_gate_when_very_different", True)

    if not skip_when_very_different:
        return False

    return is_very_different_length(structure_input_ratio, config, config_path)

