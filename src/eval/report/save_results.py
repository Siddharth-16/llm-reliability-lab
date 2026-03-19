import json
from datetime import datetime
from pathlib import Path

def save_report(report: dict, output_dir: str = "reports") -> str:
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    filepath = Path(output_dir) / filename

    with open(filepath, "w") as f:
        json.dump(report, f, indent=2)

    return str(filepath)