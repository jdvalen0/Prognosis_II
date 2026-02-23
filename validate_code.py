#!/usr/bin/env python3
"""
Script de validación sintáctica y lógica del código
No requiere dependencias externas, solo valida sintaxis Python.
"""

import ast
import sys
import os
from pathlib import Path

def validate_python_file(file_path):
    """Valida que un archivo Python tenga sintaxis correcta."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Error de sintaxis en línea {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_imports(file_path):
    """Verifica que los imports sean correctos."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            code = f.read()
        tree = ast.parse(code)
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        return []

def validate_directory(directory):
    """Valida todos los archivos Python en un directorio."""
    results = {
        'valid': [],
        'invalid': [],
        'total': 0
    }
    
    for py_file in Path(directory).rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        results['total'] += 1
        is_valid, error = validate_python_file(py_file)
        
        if is_valid:
            results['valid'].append(str(py_file))
        else:
            results['invalid'].append((str(py_file), error))
    
    return results

def main():
    """Función principal."""
    print("=" * 80)
    print("🔍 VALIDACIÓN SINTÁCTICA - PROGNOSIS II v2.0")
    print("=" * 80)
    print()
    
    # Directorios a validar
    directories = [
        'src',
        'tests'
    ]
    
    all_results = {
        'valid': [],
        'invalid': [],
        'total': 0
    }
    
    for directory in directories:
        if not os.path.exists(directory):
            print(f"⚠️  Directorio {directory} no existe, saltando...")
            continue
            
        print(f"Validando {directory}/...")
        results = validate_directory(directory)
        
        all_results['valid'].extend(results['valid'])
        all_results['invalid'].extend(results['invalid'])
        all_results['total'] += results['total']
        
        print(f"  ✅ Válidos: {len(results['valid'])}")
        if results['invalid']:
            print(f"  ❌ Inválidos: {len(results['invalid'])}")
        print()
    
    # Resumen
    print("=" * 80)
    print("📊 RESUMEN")
    print("=" * 80)
    print(f"Total de archivos Python: {all_results['total']}")
    print(f"✅ Válidos: {len(all_results['valid'])}")
    print(f"❌ Inválidos: {len(all_results['invalid'])}")
    print()
    
    if all_results['invalid']:
        print("🔴 ARCHIVOS CON ERRORES:")
        for file_path, error in all_results['invalid']:
            print(f"  - {file_path}")
            print(f"    {error}")
        print()
        return 1
    else:
        print("✅ TODOS LOS ARCHIVOS TIENEN SINTAXIS CORRECTA")
        print()
        return 0

if __name__ == '__main__':
    sys.exit(main())
