"""
Fase 1+2 - Descarga las monedas nuevas (Peseta, Escudo, Guilder, Lira, Colon
salvadoreno, Balboa, Dirham marroqui, Dinar tunecino, Dinar libio) desde
Wikimedia Commons y las agrega al dataset de CoinVision.

Como correrlo:
    1. Copia este archivo y `new_classes_manifest.csv` a la carpeta CoinVision/
       (junto a Codevision.ipynb), o pega este codigo como una celda nueva al
       PRINCIPIO del notebook (antes de la celda "Load Dataset").
    2. pip install requests pillow  (si no los tienes ya)
    3. python build_new_classes.py
       (o corre la celda si lo pegaste en el notebook)

Que hace:
    - Lee new_classes_manifest.csv (denominacion, moneda, pais, categoria/lista
      de archivos de Wikimedia Commons por cada clase nueva).
    - Para cada clase, pide a la API de Wikimedia Commons la lista de archivos
      (o usa la lista fija si el modo es "filelist"), y descarga cada imagen.
    - Convierte todo a .jpg y las guarda en data/train/<class_id>/ y
      data/test/<class_id>/, con el MISMO formato de nombre que ya usas
      (NNN__<denominacion>_<pais>.jpg), separando un ~15% para test.
    - Actualiza cat_to_name.json agregando las clases nuevas AL FINAL, sin
      tocar ni una sola de las que ya existen.
    - Guarda un log de atribucion (data/new_classes_attribution.csv) con
      autor/licencia/URL de cada imagen descargada, por si luego necesitas
      dar credito (las fotos de Commons son de uso libre pero casi todas
      piden atribucion).

Que NO hace (a proposito):
    - No toca class_dictionary.csv, dataset_manifest.csv, ni label_mapping.json.
      Esos los regenera el propio notebook (celdas "Parse Classes into a
      Table" -> "Count Images per Class" -> ... -> "Create Label Mapping")
      cuando vuelvas a correrlo desde el inicio.
    - No borra ni modifica ninguna imagen de las clases existentes.

--- Robustez ante "429 Too many requests" (2026-09-05) ---
Wikimedia Commons empieza a devolver 429 si le pegas muchas llamadas
seguidas a su API sin pausas. La primera version de este script llamaba a
get_image_info() UNA VEZ POR ARCHIVO sin ningun try/except alrededor, asi
que en cuanto Commons devolvia un solo 429, la excepcion no se atajaba en
ningun lado y tronaba TODO el script -- perdiendo el progreso de todas las
clases que faltaban, no solo la que estaba fallando. Se corrigio asi:
  1. commons_get_with_retry(): reintenta con espera creciente (respeta el
     header Retry-After si Commons lo manda) ante 429 o errores 5xx, hasta
     MAX_RETRIES veces, antes de rendirse en ESE pedido puntual.
  2. get_image_info_batch(): en vez de una llamada por archivo, la API de
     Commons acepta hasta 50 titulos por llamada (separados por "|") -- esto
     reduce el numero total de pedidos hasta 50x, la forma mas efectiva de
     no toparse con el limite en primer lugar.
  3. Cada archivo y cada clase estan envueltos en su propio try/except: un
     error irrecuperable en un archivo puntual, o incluso en una clase
     entera, se loguea y se salta -- nunca vuelve a tronar todo el proceso.
  4. Pausas mas generosas entre llamadas a la API (REQUEST_DELAY) y entre
     descargas de imagenes.
"""
import csv
import json
import os
import time
import urllib.parse
import urllib.request

import requests
from PIL import Image

# ---------------------------------------------------------------------------
# Configuracion - ajusta DATA_DIR si hace falta (debe ser la misma carpeta
# "data" que ya usa Codevision.ipynb)
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
CAT_TO_NAME_PATH = os.path.join(DATA_DIR, "cat_to_name.json")
MANIFEST_CSV = os.path.join(HERE, "new_classes_manifest.csv")
ATTRIBUTION_CSV = os.path.join(DATA_DIR, "new_classes_attribution.csv")

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "CoinVision-dataset-builder/1.0 (personal portfolio project)"
MIN_WIDTH_PX = 150  # descarta iconos/miniaturas irrelevantes
TEST_FRACTION = 0.15  # ~lo mismo que el resto del dataset (~4 de 28-32)
MIN_TEST_IMAGES = 1

MAX_RETRIES = 5
REQUEST_DELAY = 0.5   # pausa base entre llamadas a la API (antes: nada entre get_image_info)
DOWNLOAD_DELAY = 0.4  # pausa entre descargas de imagenes (subida de 0.2 -> 0.3 -> 0.4)
BATCH_SIZE = 50       # maximo de titulos por llamada a la API (limite de MediaWiki)
THUMB_WIDTH = 800     # bajamos miniaturas de 800px, no el original -- el modelo
                      # redimensiona a 300x300 igual, y las miniaturas pasan por un
                      # servidor mucho menos estricto que el de originales.


def commons_get_with_retry(params, what=""):
    """Como requests.get + raise_for_status, pero reintenta con espera
    creciente ante 429/5xx en vez de tronar. Devuelve None (nunca levanta)
    si se agotan los reintentos -- el que llama debe tratar None como
    'no se pudo, seguir con lo siguiente'."""
    params = dict(params, format="json")
    delay = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(COMMONS_API, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"    (Commons devolvio {resp.status_code} en {what}, "
                      f"reintento {attempt}/{MAX_RETRIES} en {wait:.1f}s...)")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as exc:
            print(f"    (error de red en {what}, reintento {attempt}/{MAX_RETRIES} en {delay:.1f}s: {exc})")
            time.sleep(delay)
            delay *= 2
    print(f"    ADVERTENCIA: se agotaron los reintentos para {what}, se omite.")
    return None


def list_category_files(category_name):
    """Devuelve los titulos 'File:...' dentro de una categoria de Commons."""
    titles = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": f"Category:{category_name}",
            "cmtype": "file",
            "cmlimit": "500",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = commons_get_with_retry(params, what=f"listar categoria {category_name!r}")
        if data is None:
            break
        members = data.get("query", {}).get("categorymembers", [])
        titles.extend(m["title"] for m in members)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(REQUEST_DELAY)
    return titles


def _parse_imageinfo_page(page):
    info = (page.get("imageinfo") or [None])[0]
    if not info:
        return None
    meta = info.get("extmetadata", {})
    license_name = meta.get("LicenseShortName", {}).get("value", "unknown")
    artist = meta.get("Artist", {}).get("value", "unknown")
    # Preferimos la miniatura (thumburl, pedida con iiurlwidth mas abajo) en
    # vez del archivo original (url): Commons devuelve 429 "please... instead
    # use thumbnail images" cuando se piden muchos originales seguidos -- las
    # miniaturas pasan por un servidor distinto y mucho menos estricto, y de
    # cualquier forma el modelo redimensiona todo a 300x300, asi que un
    # original de 4000px no aporta nada. Si Commons no genero thumburl para
    # este archivo (pasa con algunos formatos), caemos al original.
    download_url = info.get("thumburl") or info["url"]
    return {
        "url": download_url,
        # width/height aqui siguen siendo los del ORIGINAL (no de la
        # miniatura) -- es lo que queremos para el filtro de "muy chica"
        # unas lineas mas abajo: no queremos descartar una foto real por
        # verse chica en el pedido de miniatura, sino por ser realmente un
        # icono pequeno en Commons.
        "width": info.get("width", 0),
        "height": info.get("height", 0),
        "license": license_name,
        "artist": artist,
    }


def get_image_info_batch(file_titles):
    """Devuelve {titulo: info_o_None} para una lista de titulos, pidiendolos
    en lotes de hasta BATCH_SIZE por llamada (en vez de uno por uno) para
    reducir drasticamente el numero de pedidos a la API."""
    results = {title: None for title in file_titles}
    for i in range(0, len(file_titles), BATCH_SIZE):
        chunk = file_titles[i : i + BATCH_SIZE]
        data = commons_get_with_retry(
            {
                "action": "query",
                "titles": "|".join(chunk),
                "prop": "imageinfo",
                "iiprop": "url|size|extmetadata",
                "iiurlwidth": THUMB_WIDTH,  # pide tambien thumburl (ver _parse_imageinfo_page)
            },
            what=f"info de {len(chunk)} archivos",
        )
        if data is None:
            continue  # esta tanda se pierde, pero las demas clases siguen
        pages = data.get("query", {}).get("pages", {})
        # Mapear de vuelta por titulo normalizado (Commons puede normalizar
        # guiones bajos/mayusculas en la respuesta).
        normalized = {}
        for norm in data.get("query", {}).get("normalized", []):
            normalized[norm["to"]] = norm["from"]
        for page in pages.values():
            title = page.get("title")
            original_title = normalized.get(title, title)
            results[original_title] = _parse_imageinfo_page(page)
        if i + BATCH_SIZE < len(file_titles):
            time.sleep(REQUEST_DELAY)
    return results


def download_image(url, dest_path):
    """Baja el archivo con los mismos reintentos/backoff que commons_get_with_retry
    (el servidor de imagenes de Commons tira 429 en descargas individuales de
    vez en cuando, no solo en los pedidos de metadatos -- antes esto no se
    reintentaba, se perdia la foto directamente)."""
    delay = 2.0
    data = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else delay
                print(f"    (Commons devolvio {resp.status_code} bajando la imagen, "
                      f"reintento {attempt}/{MAX_RETRIES} en {wait:.1f}s...)")
                time.sleep(wait)
                delay *= 2
                continue
            resp.raise_for_status()
            data = resp.content
            break
        except requests.exceptions.RequestException as exc:
            print(f"    (error de red bajando la imagen, reintento {attempt}/{MAX_RETRIES} en {delay:.1f}s: {exc})")
            time.sleep(delay)
            delay *= 2
    if data is None:
        raise RuntimeError("se agotaron los reintentos bajando la imagen")

    tmp_path = dest_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    # Convertimos todo a JPEG RGB, igual que hace CoinDataset.__getitem__
    # (Image.open(...).convert("RGB")), asi los archivos quedan uniformes
    # con el resto del dataset.
    with Image.open(tmp_path) as im:
        im.convert("RGB").save(dest_path, "JPEG", quality=90)
    os.remove(tmp_path)


def safe_filename_piece(text):
    return text.replace("/", "-").replace("\\", "-")


def process_class(row, cat_to_name):
    """Procesa una fila del manifiesto. Cualquier error se loguea y la
    funcion devuelve (saved, n_test) en vez de dejar escapar la excepcion --
    asi un problema con ESTA clase nunca aborta las que faltan."""
    class_id = int(row["class_id"])
    denom = row["denomination"]
    currency = row["currency"]
    country = row["country"]
    label = f"{denom},{currency},{country}"

    print(f"\n=== Clase {class_id}: {label} ===")

    if row["source_mode"] == "category":
        file_titles = list_category_files(row["source"])
    else:  # filelist
        file_titles = [f"File:{name}" for name in row["source"].split("|")]

    if not file_titles:
        print(f"  ADVERTENCIA: no se encontraron archivos para {row['source']!r}, se salta esta clase.")
        return [], 0, 0

    train_dir = os.path.join(TRAIN_DIR, str(class_id))
    test_dir = os.path.join(TEST_DIR, str(class_id))
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    infos = get_image_info_batch(file_titles)

    if len(file_titles) < 3:
        # Con tan pocas fotos, reservar una para test deja la clase
        # sin NINGUNA imagen de entrenamiento. Mejor meter todo a train.
        n_test = 0
    else:
        n_test = max(MIN_TEST_IMAGES, round(len(file_titles) * TEST_FRACTION))
    saved = 0
    attribution_rows = []
    for idx, title in enumerate(file_titles):
        try:
            info = infos.get(title)
            if not info:
                print(f"  (sin info) {title}")
                continue
            if info["width"] and info["width"] < MIN_WIDTH_PX:
                print(f"  (muy chica, {info['width']}px) {title}")
                continue

            split_dir = test_dir if idx < n_test else train_dir
            seq = idx + 1
            fname = f"{seq:03d}__{safe_filename_piece(denom)}_{country}.jpg"
            dest_path = os.path.join(split_dir, fname)

            download_image(info["url"], dest_path)
            saved += 1
            attribution_rows.append(
                {
                    "class_id": class_id,
                    "denomination": denom,
                    "file": fname,
                    "commons_title": title,
                    "source_url": info["url"],
                    "license": info["license"],
                    "artist": info["artist"],
                }
            )
            print(f"  OK  {title}  ->  {os.path.relpath(dest_path, DATA_DIR)}")
        except Exception as exc:  # noqa: BLE001 - un archivo malo no debe tumbar la clase
            print(f"  ERROR con {title}: {exc}")

        time.sleep(DOWNLOAD_DELAY)

    return attribution_rows, saved, n_test


def main():
    with open(MANIFEST_CSV, encoding="utf-8") as f:
        classes = list(csv.DictReader(f))

    cat_to_name = json.load(open(CAT_TO_NAME_PATH, encoding="utf-8"))
    existing_ids = set(int(k) for k in cat_to_name.keys())

    all_attribution_rows = []
    summary = []

    for row in classes:
        class_id = int(row["class_id"])
        if class_id in existing_ids:
            print(f"[{class_id}] ya existe en cat_to_name.json, se salta (no se toca).")
            continue

        try:
            attribution_rows, saved, n_test = process_class(row, cat_to_name)
        except Exception as exc:  # noqa: BLE001 - una clase entera no debe tumbar el script
            print(f"  ERROR FATAL procesando clase {class_id}: {exc} -- se salta esta clase, sigue con la siguiente.")
            summary.append((class_id, f"{row['denomination']},{row['currency']},{row['country']}", 0, 0))
            continue

        label = f"{row['denomination']},{row['currency']},{row['country']}"
        summary.append((class_id, label, saved, n_test))
        all_attribution_rows.extend(attribution_rows)
        if saved > 0:
            cat_to_name[str(class_id)] = label

        # Guardado incremental despues de CADA clase: si algo interrumpe el
        # script mas adelante (se cierra la laptop, se corta la luz, etc.),
        # las clases ya procesadas no se pierden ni hay que re-descargarlas.
        with open(CAT_TO_NAME_PATH, "w", encoding="utf-8") as f:
            json.dump(cat_to_name, f, ensure_ascii=False, indent=4, sort_keys=False)

    # Guardar log de atribucion (append si ya existe)
    if all_attribution_rows:
        write_header = not os.path.exists(ATTRIBUTION_CSV)
        with open(ATTRIBUTION_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_id", "denomination", "file", "commons_title", "source_url", "license", "artist"],
            )
            if write_header:
                writer.writeheader()
            writer.writerows(all_attribution_rows)
        print(f"\nAtribucion guardada en {ATTRIBUTION_CSV} ({len(all_attribution_rows)} imagenes nuevas).")

    print(f"\ncat_to_name.json actualizado: {len(cat_to_name)} clases en total.")
    print("\n=== Resumen ===")
    for class_id, label, saved, n_test in summary:
        estado = "" if saved > 0 else "  <-- SIN FOTOS, revisar el nombre/categoria en el manifiesto"
        print(f"  {class_id}: {label} -> {saved} imagenes bajadas ({n_test} para test){estado}")

    print(
        "\nListo. Ahora vuelve a correr el notebook desde la Fase 1 "
        "(celdas de 'Load Dataset' en adelante) para que regenere "
        "class_dictionary.csv, dataset_manifest.csv y label_mapping.json "
        "incluyendo estas clases nuevas junto con las que ya tenias."
    )


if __name__ == "__main__":
    main()
