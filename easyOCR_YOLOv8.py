from collections import defaultdict, deque
import cv2
import re
import os
import numpy as np
from ultralytics import YOLO
import easyocr

# --- I. Configuration Globale et Initialisation ---

model = YOLO("best.pt")
reader = easyocr.Reader(['en'], gpu=False)
plate_pattern = re.compile(r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$")

CONF_THRESH = 0.3

# Variables de stabilisation (pour vidéo)
plate_history = defaultdict(lambda: deque(maxlen=10)) 
plate_final = {}


# --- II. Fonctions Utilitaires ---

def get_file_type(file_path):
    """Détecte si c'est une image ou une vidéo."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in image_extensions:
        return "image"
    elif ext in video_extensions:
        return "video"
    else:
        return None


def get_box_id(x1, y1, x2, y2):
    """Génère un ID pseudo-unique basé sur les coordonnées."""
    return f"{int(x1/10)}_{int(y1/10)}_{int(x2/10)}_{int(y2/10)}"


def get_stable_plate(box_id, new_text):
    """Vote majoritaire sur les 10 dernières prédictions."""
    if new_text:
        plate_history[box_id].append(new_text)
        history_list = list(plate_history[box_id])
        if history_list:
            most_common = max(set(history_list), key=history_list.count)
            plate_final[box_id] = most_common
    return plate_final.get(box_id, "")


def correct_plate_format(ocr_text):
    """Corrige les erreurs OCR courantes."""
    mapping_num_to_alpha = {"0": "O", "1": "I", "5": "S", "8": "B"}
    mapping_alpha_to_num = {"O": "0", "I": "1", "Z": "2", "S": "5", "B": "8"}

    ocr_text = ocr_text.upper().replace(" ", "").replace("-", "")

    if len(ocr_text) != 7:
        return ""

    corrected = []
    for i, ch in enumerate(ocr_text):
        if i < 2 or i >= 4:
            if ch.isdigit() and ch in mapping_num_to_alpha:
                corrected.append(mapping_num_to_alpha[ch])
            elif ch.isalpha():
                corrected.append(ch)
            else:
                return ""
        else:
            if ch.isalpha() and ch in mapping_alpha_to_num:
                corrected.append(mapping_alpha_to_num[ch])
            elif ch.isdigit():
                corrected.append(ch)
            else:
                return ""

    return "".join(corrected)


def recognize_plate(plate_crop):
    """Prétraitement + OCR + validation."""
    if plate_crop is None or plate_crop.size == 0:
        return ""

    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        plate_resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        ocr_result = reader.readtext(
            plate_resized, 
            detail=0, 
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )

        if ocr_result:
            raw_text = "".join(ocr_result)
            candidate = correct_plate_format(raw_text)
            if candidate and plate_pattern.match(candidate):
                return candidate
    except:
        pass

    return ""


def process_frame(frame, use_stabilization=False):
    """
    Traite une frame (image ou frame vidéo).
    Retourne la frame annotée et les plaques détectées.
    """
    height, width = frame.shape[:2]
    detected_plates = []

    results = model(frame, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf.cpu().numpy())
            
            if conf < CONF_THRESH:
                continue
            
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            
            plate_crop = frame[y1:y2, x1:x2]
            text = recognize_plate(plate_crop)

            # Stabilisation (seulement pour vidéo)
            if use_stabilization:
                box_id = get_box_id(x1, y1, x2, y2)
                display_text = get_stable_plate(box_id, text)
            else:
                display_text = text

            if display_text:
                detected_plates.append({
                    "text": display_text,
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2)
                })

            # Dessiner rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Afficher confiance
            cv2.putText(frame, f"{conf:.0%}", (x1, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Overlay de la plaque
            if plate_crop.size > 0:
                try:
                    overlay_h, overlay_w = 80, 200
                    plate_resized = cv2.resize(plate_crop, (overlay_w, overlay_h))
                    
                    oy1 = max(0, y1 - overlay_h - 40)
                    ox1 = max(0, x1)
                    oy2 = oy1 + overlay_h
                    ox2 = ox1 + overlay_w

                    if oy2 <= height and ox2 <= width and oy1 >= 0:
                        frame[oy1:oy2, ox1:ox2] = plate_resized

                    if display_text:
                        text_y = max(25, oy1 - 10)
                        cv2.putText(frame, display_text, (ox1, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 5)
                        cv2.putText(frame, display_text, (ox1, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                except:
                    pass

    return frame, detected_plates


# --- III. Traitement Image ---

def process_image(input_path, output_path=None):
    """Traite une seule image."""
    print(f"Traitement de l'image: {input_path}")
    
    # Charger l'image
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"Erreur: Impossible de charger l'image: {input_path}")
        return None
    
    print(f"Dimensions: {image.shape[1]}x{image.shape[0]}")
    
    # Traiter l'image
    annotated_image, plates = process_frame(image, use_stabilization=False)
    
    # Générer le nom de sortie si non spécifié
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_detected{ext}"
    
    # Sauvegarder
    cv2.imwrite(output_path, annotated_image)
    
    # Afficher résultats
    print("=" * 50)
    print("Traitement terminé!")
    print(f"Image sauvegardée: {output_path}")
    print(f"Plaques détectées: {len(plates)}")
    
    for i, plate in enumerate(plates, 1):
        print(f"   Plaque {i}: {plate['text']} (confiance: {plate['confidence']:.0%})")
    
    return plates


# --- IV. Traitement Vidéo ---

def process_video(input_path, output_path=None):
    """Traite une vidéo complète."""
    global plate_history, plate_final
    plate_history = defaultdict(lambda: deque(maxlen=10))
    plate_final = {}
    
    print(f"🎥 Traitement de la vidéo: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo: {input_path}")
        return None
    
    # Propriétés vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Dimensions: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    # Générer le nom de sortie si non spécifié
    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_detected{ext}"
    
    # Créer le writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    print("Début du traitement...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Afficher progression
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"⏳ Progression: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Traiter la frame
        annotated_frame, _ = process_frame(frame, use_stabilization=True)
        out.write(annotated_frame)
    
    # Libérer ressources
    cap.release()
    out.release()
    
    # Afficher résultats
    print("=" * 50)
    print("Traitement terminé!")
    print(f"Vidéo sauvegardée: {output_path}")
    print(f"Plaques uniques détectées: {len(plate_final)}")
    
    for plate in plate_final.values():
        print(f"   {plate}")
    
    return list(plate_final.values())


# --- V. Fonction Principale Universelle ---

def process(input_path, output_path=None):
    """
    Fonction universelle qui détecte automatiquement 
    si c'est une image ou une vidéo et traite en conséquence.
    """
    if not os.path.exists(input_path):
        print(f"Erreur: Fichier non trouvé: {input_path}")
        return None
    
    file_type = get_file_type(input_path)
    
    if file_type == "image":
        return process_image(input_path, output_path)
    elif file_type == "video":
        return process_video(input_path, output_path)
    else:
        print(f"Erreur: Format non supporté: {input_path}")
        print("   Formats images: .jpg, .jpeg, .png, .bmp, .tiff, .webp")
        print("   Formats vidéos: .mp4, .avi, .mov, .mkv, .wmv, .flv")
        return None


# --- VI. Point d'Entrée ---

if __name__ == "__main__":
    
    input_file = "TestTwo.mp4"      # ← Mets une image ou vidéo ici
    output_file = None              # ← None = génère automatiquement
    
    process(input_file, output_file)