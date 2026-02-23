#!/usr/bin/env python3
"""
Script de ejecución de pruebas para Prognosis II v2.0
Ejecuta todos los tests y genera reporte de resultados.
"""

import sys
import unittest
import os
from io import StringIO

# Agregar ruta raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    """Ejecuta todos los tests y genera reporte."""
    
    print("=" * 80)
    print("🧪 EJECUTANDO TESTS - PROGNOSIS II v2.0")
    print("=" * 80)
    print()
    
    # Descubrir y cargar todos los tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(os.path.dirname(__file__), 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Ejecutar tests
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Mostrar resultados
    print(stream.getvalue())
    
    print("=" * 80)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 80)
    print(f"Tests ejecutados: {result.testsRun}")
    print(f"✅ Exitosos: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Fallidos: {len(result.failures)}")
    print(f"⚠️  Errores: {len(result.errors)}")
    print()
    
    if result.failures:
        print("🔴 TESTS FALLIDOS:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback.split(chr(10))[-2]}")
        print()
    
    if result.errors:
        print("⚠️  ERRORES:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback.split(chr(10))[-2]}")
        print()
    
    # Determinar código de salida
    if result.wasSuccessful():
        print("✅ TODOS LOS TESTS PASARON")
        return 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
