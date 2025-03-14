import requests
import os


class Downloader:
    def __init__(self, model_folder='data/models/'):
        self.model_folder = model_folder

    def download_model(self, model_url):
        model_name = self._get_name_model(model_url)
        model_path = os.path.join(self.model_folder, model_name)

        if os.path.exists(model_path):
            print(f"Modelo ya existe: {model_path}")
            return model_path

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        print("Descargando modelo...")
        try:
            response = requests.get(model_url)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                f.write(response.content)

            print(f"Modelo descargado: {model_path}")
            return model_path

        except requests.exceptions.RequestException as e:
            print(f"Error al descargar el modelo: {e}")
            return None

    def _get_name_model(self, url_model):
        return url_model.split('/')[-1]
