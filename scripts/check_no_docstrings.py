import ast
import sys
from pathlib import Path


def remove_docstrings(filepath: Path) -> bool:
    with open(filepath, encoding="utf-8") as f:
        lines = f.readlines()

    try:
        tree = ast.parse("".join(lines), filename=str(filepath))
    except SyntaxError:
        return False

    docstring_ranges = []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            docstring = ast.get_docstring(node)
            if docstring:
                if isinstance(node, ast.Module):
                    start_line = 0
                else:
                    start_line = node.body[0].lineno - 1 if node.body else node.lineno

                expr = (
                    node.body[0]
                    if node.body and isinstance(node.body[0], ast.Expr)
                    else None
                )
                if expr and isinstance(expr.value, ast.Constant):
                    end_line = expr.end_lineno
                    docstring_ranges.append((start_line, end_line))

    if not docstring_ranges:
        return False

    docstring_ranges.sort(reverse=True)

    for start, end in docstring_ranges:
        del lines[start:end]

    with open(filepath, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True


def check_docstrings(filepath: Path) -> list[str]:
    errors = []
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(filepath))
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)
        ):
            docstring = ast.get_docstring(node)
            if docstring:
                line_no = node.lineno if hasattr(node, "lineno") else 0
                node_type = type(node).__name__
                errors.append(f"{filepath}:{line_no}: Docstring found in {node_type}")
    return errors


def main() -> int:
    fix_mode = "--fix" in sys.argv
    filepaths = [arg for arg in sys.argv[1:] if arg != "--fix"]

    if fix_mode:
        fixed_count = 0
        for filepath in filepaths:
            path = Path(filepath)
            if path.suffix == ".py":
                if remove_docstrings(path):
                    print(f"Fixed: {filepath}")
                    fixed_count += 1
        return 0
    else:
        errors = []
        for filepath in filepaths:
            path = Path(filepath)
            if path.suffix == ".py":
                errors.extend(check_docstrings(path))
        if errors:
            for error in errors:
                print(error)
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())
