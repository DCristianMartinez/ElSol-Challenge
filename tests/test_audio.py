import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.audio import extract_structured_info


def test_extract_structured_info():
    text = "Nombre: Pepito Gomez\nEdad: 45\nDiagnóstico: Fiebre"
    data = extract_structured_info(text)
    assert data["nombre"] == "Pepito Gomez"
    assert data["edad"] == "45"
    assert data["diagnostico"].lower() == "fiebre"
