"""Apply the pinned dependency's narrow Android cgroup compatibility fix."""
import importlib.metadata
import importlib.util
import sys
from pathlib import Path


def main():
    specification = importlib.util.find_spec("tinygrad")
    if specification is None:
        return
    helpers = Path(specification.origin).parent / "helpers.py"
    if not helpers.resolve().is_relative_to(Path(sys.prefix).resolve()) or sys.prefix == sys.base_prefix:
        raise RuntimeError("Only patch tinygrad inside this application's private virtual environment.")
    if importlib.metadata.version("tinygrad") != "0.14.0":
        raise RuntimeError("Review the cgroup compatibility patch when changing the pinned tinygrad version.")
    original = "except (FileNotFoundError, ValueError, ZeroDivisionError): pass"
    corrected = "except (FileNotFoundError, PermissionError, ValueError, ZeroDivisionError): pass"
    contents = helpers.read_text()
    if corrected in contents:
        return
    if contents.count(original) != 1:
        raise RuntimeError("The pinned tinygrad CPU-quota probe has changed; refusing a blind patch.")
    helpers.write_text(contents.replace(original, corrected))
    print("Applied tinygrad's unreadable-cgroup fallback for Android/PRoot.")


if __name__ == "__main__":
    main()
