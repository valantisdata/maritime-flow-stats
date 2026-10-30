from __future__ import annotations

import os
from pathlib import Path

ARCGIS_ROOT = "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"

LAYERS = {
    "chokepoints_daily": f"{ARCGIS_ROOT}/Daily_Chokepoints_Data/FeatureServer/0/query",
    "chokepoints_meta": f"{ARCGIS_ROOT}/PortWatch_chokepoints_database/FeatureServer/0/query",
    "ports_daily": f"{ARCGIS_ROOT}/Daily_Trade_Data/FeatureServer/0/query",
    "ports_meta": f"{ARCGIS_ROOT}/PortWatch_ports_database/FeatureServer/0/query",
}

# IMF terms for statistical data require attribution, and a statement when the
# data has been materially transformed. Every figure and report carries this.
ATTRIBUTION = (
    "Source: International Monetary Fund, IMF PortWatch (https://portwatch.imf.org), "
    "based on UN Global Platform AIS data. Aggregated and modelled by the author; "
    "figures are derived estimates, not official IMF statistics."
)
ATTRIBUTION_SHORT = "Source: IMF PortWatch (portwatch.imf.org); transformed by the author"

DEFAULT_CACHE_DIR = Path(os.environ.get("MARITIME_FLOW_CACHE", Path.home() / ".cache" / "maritime-flow"))

VESSEL_TYPES = ["container", "dry_bulk", "general_cargo", "roro", "tanker"]
CARGO_TYPES = ["container", "dry_bulk", "general_cargo", "roro"]
COUNT_COLUMNS = [f"n_{v}" for v in VESSEL_TYPES] + ["n_cargo", "n_total"]
CAPACITY_COLUMNS = [f"capacity_{v}" for v in VESSEL_TYPES] + ["capacity_cargo", "capacity"]
