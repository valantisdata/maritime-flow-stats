"""Reference events used to annotate charts and as default impact dates.

Only events with a well-documented start date are listed, each with a source.
Add your own with ``--event YYYY-MM-DD`` on the command line.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Event:
    key: str
    date: str
    label: str
    source: str


EVENTS = {
    "ever_given": Event(
        "ever_given",
        "2021-03-23",
        "Ever Given blocks the Suez Canal",
        "https://en.wikipedia.org/wiki/2021_Suez_Canal_obstruction",
    ),
    "red_sea": Event(
        "red_sea",
        "2023-11-19",
        "Galaxy Leader seized; start of Red Sea attacks on shipping",
        "https://en.wikipedia.org/wiki/Galaxy_Leader_hijacking",
    ),
}
