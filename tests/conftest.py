import pandas as pd
import pytest

from maritime_flow.data.cleaning import clean_daily, to_series
from maritime_flow.data.synthetic import demo_dataset, make_chokepoint_frame


@pytest.fixture(scope="session")
def demo_panel() -> pd.DataFrame:
    panel, _ = clean_daily(demo_dataset())
    return panel


@pytest.fixture(scope="session")
def shocked_series() -> pd.Series:
    """Known truth: traffic drops to 50% on 2023-12-01."""
    raw = make_chokepoint_frame("cp", "Test", start="2020-01-01", end="2025-06-30",
                                base=60, shift_date="2023-12-01", shift_ratio=0.5, seed=3)
    panel, _ = clean_daily(raw)
    return to_series(panel, "cp")


@pytest.fixture(scope="session")
def flat_series() -> pd.Series:
    raw = make_chokepoint_frame("cp", "Flat", start="2021-01-01", end="2025-06-30", base=40, seed=11)
    panel, _ = clean_daily(raw)
    return to_series(panel, "cp")
