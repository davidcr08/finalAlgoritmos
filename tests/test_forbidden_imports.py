"""Test automático: falla si hay imports prohibidos dentro de src/."""
import ast
import sys
from pathlib import Path

# Imports prohibidos en src/ (solo permitidos en tests/)
FORBIDDEN_IMPORTS = {
    "scipy",
    "sklearn",
    "sklearn.",
    "nltk",
    "difflib",
}

# Módulos permitidos solo como apoyo (deben declararse en documento)
ALLOWED_WITH_DECLARATION = {"numpy", "matplotlib"}

# Funciones built-in prohibidas para implementación de algoritmos
FORBIDDEN_BUILTINS = {"find", "index", "rfind", "rindex"}


def scan_file(filepath: Path) -> list[str]:
    """Escanea un archivo .py y devuelve lista de violaciones."""
    violations = []
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception as e:
        violations.append(f"{filepath}: error parseando ({e})")
        return violations

    for node in ast.walk(tree):
        # Imports: import X, from X import Y
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in FORBIDDEN_IMPORTS or any(
                    alias.name.startswith(f + ".") for f in FORBIDDEN_IMPORTS
                ):
                    violations.append(
                        f"{filepath}:{node.lineno}: import prohibido '{alias.name}'"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module = node.module.split(".")[0]
                if module in FORBIDDEN_IMPORTS or any(
                    node.module.startswith(f + ".") for f in FORBIDDEN_IMPORTS
                ):
                    violations.append(
                        f"{filepath}:{node.lineno}: from '{node.module}' import ... prohibido"
                    )

        # Llamadas a str.find, str.index, etc. (Attribute + Call)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_BUILTINS:
                # Verificar que sea sobre str (heurística simple)
                violations.append(
                    f"{filepath}:{node.lineno}: uso de str.{node.func.attr}() prohibido"
                )

    return violations


def test_no_forbidden_imports_in_src():
    """Test principal: escanea todo src/ y falla si hay violaciones."""
    src_root = Path(__file__).parent.parent / "src"
    all_violations = []

    for py_file in src_root.rglob("*.py"):
        # Saltar __pycache__ y archivos de test si estuvieran en src
        if "__pycache__" in py_file.parts:
            continue
        violations = scan_file(py_file)
        all_violations.extend(violations)

    if all_violations:
        msg = "IMPORTS PROHIBIDOS ENCONTRADOS EN src/:\n" + "\n".join(all_violations)
        raise AssertionError(msg)


if __name__ == "__main__":
    # Ejecutable directo para debugging
    try:
        test_no_forbidden_imports_in_src()
        print("✅ Sin imports prohibidos en src/")
        sys.exit(0)
    except AssertionError as e:
        print(f"❌ {e}")
        sys.exit(1)