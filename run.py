import sys
import subprocess
import os
import importlib
import importlib.util
import importlib.metadata
import venv
import yaml
from pathlib import Path

VENV_NAME = "AEROGPT_ENV"
VENV_PATH = Path(VENV_NAME)

CONFIG_YAML = Path("config/requirements.yaml")  # tu archivo .yaml con paquetes

def check_python_version():
    """Verifica que la versión de Python sea >= 3.8"""
    if sys.version_info < (3, 8):
        print(f"❌ Error: Se requiere Python 3.8+, tienes {sys.version}")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

def create_venv():
    """Crea el entorno virtual si no existe"""
    if not VENV_PATH.exists():
        print(f"\nCreando entorno virtual {VENV_NAME}...")
        venv.create(VENV_PATH, with_pip=True)
        print(f"✅ Entorno virtual {VENV_NAME} creado.")
    else:
        print(f"\nEntorno virtual {VENV_NAME} ya existe.")

def load_requirements_from_yaml():
    """Carga la lista de paquetes desde el YAML"""
    if not CONFIG_YAML.exists():
        print(f"❌ No se encontró {CONFIG_YAML}")
        sys.exit(1)
    with open(CONFIG_YAML, "r") as f:
        data = yaml.safe_load(f)
    # Suponemos que el YAML tiene la clave 'dependencies' con la lista de paquetes
    return data.get("dependencies", [])

def get_installed_version(package_name):
    """Devuelve la versión instalada del paquete o None"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None

def install_package(package):
    """Instala un paquete usando pip dentro del venv y muestra progreso"""
    print(f"\n📦 Instalando {package} ...")
    subprocess.run([
        str(VENV_PATH / "Scripts" / "python.exe"),
        "-m", "pip", "install", package
    ], check=True)

def check_and_install_packages(packages):
    """Verifica paquetes instalados e instala los que falten"""
    to_install = []
    for pkg in packages:
        name = pkg.split("==")[0].split(">=")[0].strip()
        if get_installed_version(name) is None:
            print(f"❌ {name} no está instalado")
            to_install.append(pkg)
        else:
            print(f"✅ {name} ya instalado")

    for pkg in to_install:
        install_package(pkg)

def check_env_file():
    """Verifica que exista .env"""
    if not Path(".env").exists():
        print("⚠️ No se encontró .env. Crea un archivo con OPENAI_API_KEY=tu-api-key")
        return False
    return True

def run_streamlit():
    """Ejecuta la app Streamlit desde el venv"""
    python_exe = VENV_PATH / "Scripts" / "python.exe"
    if not Path("app.py").exists():
        print("❌ No se encontró app.py")
        sys.exit(1)
    subprocess.run([str(python_exe), "-m", "streamlit", "run", "app.py"])

def main():
    print("="*60)
    print("SETUP Y EJECUCIÓN DE AEROGPT")
    print("="*60)

    check_python_version()
    create_venv()

    # Relanzar script dentro del venv si no estamos en él
    if sys.prefix != str(VENV_PATH.resolve()):
        print("\nRelanzando script dentro del entorno virtual...\n")
        python_exe = VENV_PATH / "Scripts" / "python.exe"
        os.execv(str(python_exe), [str(python_exe)] + sys.argv)

    print("\nCargando dependencias desde YAML...")
    packages = load_requirements_from_yaml()
    print(f"Se encontraron {len(packages)} dependencias.")

    check_and_install_packages(packages)
    check_env_file()
    run_streamlit()

if __name__ == "__main__":
    main()
