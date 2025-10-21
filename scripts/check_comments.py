import sys
from pathlib import Path


def remove_non_va_comments(filepath: Path) -> bool:
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return False

    new_lines = []
    modified = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("# (VA)"):
            modified = True
            continue
        new_lines.append(line)

    if modified:
        with open(filepath, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    return modified


def check_comments(filepath: Path) -> list[str]:
    errors = []
    try:
        with open(filepath, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception:
        return []
    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            if not stripped.startswith("# (VA)"):
                errors.append(
                    f"{filepath}:{line_no}: "
                    f"Comment without '(VA)' prefix: {stripped[:50]}"
                )
    return errors


def main() -> int:
    fix_mode = "--fix" in sys.argv
    filepaths = [arg for arg in sys.argv[1:] if arg != "--fix"]

    if fix_mode:
        fixed_count = 0
        for filepath in filepaths:
            path = Path(filepath)
            if path.suffix == ".py":
                if remove_non_va_comments(path):
                    print(f"Fixed: {filepath}")
                    fixed_count += 1
        return 0
    else:
        errors = []
        for filepath in filepaths:
            path = Path(filepath)
            if path.suffix == ".py":
                errors.extend(check_comments(path))
        if errors:
            for error in errors:
                print(error)
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
