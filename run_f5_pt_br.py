import argparse
import subprocess
import os
import sys


def check_and_download_model(model_path):
    """Garante que o diretório exista e baixa o modelo caso não esteja presente."""
    destination_dir = os.path.dirname(model_path)

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"Modelo não encontrado em {model_path}. Iniciando download...")
        # URL direta para o arquivo model_last.pt
        url = "https://huggingface.co/firstpixel/F5-TTS-pt-br/resolve/main/pt-br/model_last.pt"

        try:
            # Tenta usar wget ou curl para baixar o binário de 5.4GB
            if subprocess.run(["which", "wget"], capture_output=True).returncode == 0:
                subprocess.run(["wget", "-O", model_path, url], check=True)
            elif subprocess.run(["which", "curl"], capture_output=True).returncode == 0:
                subprocess.run(["curl", "-L", "-o", model_path, url], check=True)
            else:
                print("Erro: wget ou curl não encontrados. Instale um deles para baixar o modelo.")
                sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"Erro ao baixar o modelo: {e}")
            sys.exit(1)


def run_f5_tts():
    current_dir = os.path.abspath(os.getcwd())
    default_text = "a verdade é uma frequência que sempre esteve lá, esperando apenas o sintonismo correto."

    parser = argparse.ArgumentParser(description="F5-TTS Docker Wrapper for GFX12")
    parser.add_argument(
        "-r", "--ref_audio", default="ckpts/pt-br/tests/matrix.mp3", help="Caminho do áudio de referência"
    )
    parser.add_argument("-t", "--gen_text", default=default_text, help="Texto para gerar o áudio")
    parser.add_argument(
        "-p", "--model_path", default="ckpts/pt-br/model_last.pt", help="Caminho do arquivo model_last.pt"
    )

    args = parser.parse_args()

    # Caminho absoluto para o modelo no host
    absolute_model_path = os.path.join(current_dir, args.model_path)

    # Verifica/Cria pasta e baixa o modelo se necessário
    check_and_download_model(absolute_model_path)

    output_filename = "teste_sucesso.wav"

    command = [
        "docker",
        "run",
        "-it",
        "--rm",
        "--name",
        "f5-tts-debug",
        "--network",
        "host",
        "--privileged",
        "--device=/dev/kfd",
        "--device=/dev/dri",
        "--shm-size=8gb",
        "-v",
        f"{current_dir}:/app",
        "-e",
        "HSA_OVERRIDE_GFX_VERSION=12.0.1",
        "-e",
        "PYTORCH_ROCM_ARCH=gfx1201",
        "-e",
        "FORCE_ROCM=1",
        "-w",
        "/app",
        "--entrypoint",
        "/opt/venv/bin/python3",
        "killown/f5rocm:latest",
        "/opt/venv/bin/f5-tts_infer-cli",
        "-p",
        args.model_path,
        "-mc",
        "/app/src/f5_tts/configs/F5TTS_Base.yaml",
        "-r",
        args.ref_audio,
        "-s",
        "o que é real como você define o real",
        "-t",
        args.gen_text,
        "-w",
        output_filename,
        "--device",
        "cuda",
    ]

    try:
        subprocess.run(command, check=True)

        full_output_path = os.path.join(current_dir, output_filename)
        fallback_path = os.path.join(current_dir, "tests", output_filename)

        if os.path.exists(full_output_path):
            print(f"\nReproduzindo: {full_output_path}")
            subprocess.run(["mpv", full_output_path])
        elif os.path.exists(fallback_path):
            print(f"\nReproduzindo (fallback): {fallback_path}")
            subprocess.run(["mpv", fallback_path])
        else:
            print(f"\nErro: O arquivo não foi encontrado em {full_output_path} nem em {fallback_path}")

    except subprocess.CalledProcessError as e:
        print(f"Erro na execução do Docker: {e}")
    except KeyboardInterrupt:
        print("\nProcesso interrompido.")


if __name__ == "__main__":
    run_f5_tts()
