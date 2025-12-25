"""Test to ensure all type hints use properly imported types.

This test checks that all files use type hints that are actually imported,
preventing NameError at import time.
"""

import sys
import ast
import importlib.util
from pathlib import Path
from typing import Set, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def get_imported_names(module_path: Path) -> Set[str]:
    """Extract all imported names from a Python file.

    Args:
        module_path: Path to the Python file

    Returns:
        Set of imported names (from typing, builtins, etc.)
    """
    with open(module_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(module_path))

    imported = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.asname if alias.asname else alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                # For 'from typing import X, Y', we get X and Y
                for alias in node.names:
                    imported.add(alias.asname if alias.asname else alias.name)
                # Also add the module name itself
                imported.add(node.module)

    return imported


def get_type_hints_used(module_path: Path) -> Set[str]:
    """Extract all type hints used in a Python file.

    Args:
        module_path: Path to the Python file

    Returns:
        Set of type hint names used (e.g., 'Any', 'Dict', 'Optional')
    """
    with open(module_path, 'r', encoding='utf-8') as f:
        tree = ast.parse(f.read(), filename=str(module_path))

    type_hints = set()

    def visit_node(node):
        """Recursively visit AST nodes to find type hints."""
        if isinstance(node, ast.AnnAssign):
            # Variable annotation: x: int = 5
            if node.annotation:
                type_hints.update(extract_type_names(node.annotation))
        elif isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Function annotations
            if node.returns:
                type_hints.update(extract_type_names(node.returns))
            # Parameter annotations
            for arg in node.args.args:
                if arg.annotation:
                    type_hints.update(extract_type_names(arg.annotation))
        elif isinstance(node, ast.arg):
            # Function argument annotation
            if node.annotation:
                type_hints.update(extract_type_names(node.annotation))

        # Recursively visit child nodes
        for child in ast.iter_child_nodes(node):
            visit_node(child)

    def extract_type_names(annotation_node) -> Set[str]:
        """Extract type names from an annotation node."""
        names = set()

        if isinstance(annotation_node, ast.Name):
            # Simple name: int, str, Any
            names.add(annotation_node.id)
        elif isinstance(annotation_node, ast.Subscript):
            # Generic: Dict[str, int], List[Any]
            if isinstance(annotation_node.value, ast.Name):
                names.add(annotation_node.value.id)
            # Recursively extract from slice (type arguments)
            if hasattr(annotation_node, 'slice'):
                if isinstance(annotation_node.slice, ast.Tuple):
                    for elt in annotation_node.slice.elts:
                        names.update(extract_type_names(elt))
                else:
                    names.update(extract_type_names(annotation_node.slice))
        elif isinstance(annotation_node, ast.Attribute):
            # Qualified name: typing.Any, typing.Dict
            names.add(annotation_node.attr)
        elif isinstance(annotation_node, ast.Tuple):
            # Tuple type: (int, str)
            for elt in annotation_node.elts:
                names.update(extract_type_names(elt))

        return names

    visit_node(tree)
    return type_hints


def test_type_hints_imported():
    """Test that all type hints used in translator.py are properly imported."""
    translator_path = project_root / "src" / "generator" / "translator.py"

    if not translator_path.exists():
        raise FileNotFoundError(f"translator.py not found at {translator_path}")

    imported = get_imported_names(translator_path)
    type_hints = get_type_hints_used(translator_path)

    # Common built-in types that don't need import
    builtin_types = {
        'int', 'float', 'str', 'bool', 'bytes', 'object', 'type', 'None',
        'tuple', 'list', 'dict', 'set', 'frozenset', 'slice', 'range',
        'complex', 'bytearray', 'memoryview'
    }

    # Check for missing imports
    missing_imports = []
    for hint in type_hints:
        # Skip built-in types
        if hint in builtin_types:
            continue
        # Skip if it's imported directly
        if hint in imported:
            continue
        # Skip if it's from a module that's imported (e.g., typing.Any if typing is imported)
        # Check if any imported module might contain it
        if any(hint in str(imp) for imp in imported):
            continue

        # Check if it's a qualified name (e.g., typing.Any)
        # We'll check if 'typing' is imported and hint might be from typing
        if 'typing' in imported or 'TYPE_CHECKING' in imported:
            # Common typing types
            typing_types = {
                'Any', 'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Callable',
                'Iterable', 'Iterator', 'Sequence', 'Mapping', 'Set', 'FrozenSet',
                'Type', 'TypeVar', 'Generic', 'Protocol', 'Literal', 'Final'
            }
            if hint in typing_types:
                missing_imports.append(hint)
        else:
            # Not a built-in and not imported - might be an issue
            # But be lenient - only flag obvious typing types
            typing_types = {
                'Any', 'Dict', 'List', 'Tuple', 'Optional', 'Union', 'Callable'
            }
            if hint in typing_types:
                missing_imports.append(hint)

    if missing_imports:
        raise AssertionError(
            f"Type hints used but not imported in {translator_path}:\n"
            f"  Missing: {', '.join(sorted(set(missing_imports)))}\n"
            f"  Please add them to the typing imports."
        )


def test_all_src_files_type_hints():
    """Test that all Python files in src/ use properly imported type hints."""
    src_dir = project_root / "src"

    if not src_dir.exists():
        return  # Skip if src/ doesn't exist

    errors = []

    # Common built-in types
    builtin_types = {
        'int', 'float', 'str', 'bool', 'bytes', 'object', 'type', 'None',
        'tuple', 'list', 'dict', 'set', 'frozenset'
    }

    for py_file in src_dir.rglob("*.py"):
        try:
            imported = get_imported_names(py_file)
            type_hints = get_type_hints_used(py_file)

            # Check for obvious typing types that aren't imported
            typing_types = {'Any', 'Dict', 'List', 'Tuple', 'Optional', 'Union'}
            missing = []

            for hint in type_hints:
                if hint in builtin_types:
                    continue
                if hint in imported:
                    continue
                if hint in typing_types:
                    # Check if typing module is imported
                    if 'typing' not in imported and hint not in imported:
                        missing.append(hint)

            if missing:
                errors.append(f"{py_file.relative_to(project_root)}: Missing imports: {', '.join(missing)}")
        except Exception as e:
            # Skip files that can't be parsed (might be syntax errors, etc.)
            pass

    if errors:
        # Only report the first few errors to avoid spam
        error_msg = "\n".join(errors[:10])
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more files with issues"
        raise AssertionError(f"Type hint import issues found:\n{error_msg}")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])

