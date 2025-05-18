import cv2
import numpy as np
import os
import glob
import csv
from pathlib import Path
import pytesseract

# Configuração do Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'



def preprocess_image(img):
    """Realiza pré-processamento avançado da imagem"""
    # Converter para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Equalização de histograma adaptativa para melhorar contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Redução de ruído preservando bordas
    denoised = cv2.fastNlMeansDenoising(equalized, None, h=15, templateWindowSize=7, searchWindowSize=21)
    
    # Binarização adaptativa com correção de iluminação
    binary = cv2.adaptiveThreshold(denoised, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 31, 5)
    
    # Operações morfológicas para limpeza
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return cleaned

def find_bubbles(binary_img):
    """Encontra bolhas de resposta na imagem binária"""
    # Encontrar contornos
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for cnt in contours:
        # Aproximar contorno para eliminar pequenas irregularidades
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Calcular características geométricas
        area = cv2.contourArea(approx)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # Calcular circularidade
        perimeter = cv2.arcLength(approx, True)
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calcular solidez
        hull = cv2.convexHull(approx)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Filtros combinados para identificar bolhas
        if (150 < area < 2500 and 
            0.7 < aspect_ratio < 1.3 and 
            circularity > 0.6 and 
            solidity > 0.85 and
            w > 15 and h > 15):
            
            bubbles.append({
                'contour': approx,
                'x': x, 'y': y, 'w': w, 'h': h,
                'area': area,
                'center': (x + w//2, y + h//2)
            })
    
    return bubbles

def group_bubbles(bubbles):
    """Agrupa bolhas por linha e ordena"""
    if not bubbles:
        return []
    
    # Agrupar por posição vertical (mesma linha)
    bubbles.sort(key=lambda b: b['center'][1])
    groups = []
    current_group = [bubbles[0]]
    
    for bubble in bubbles[1:]:
        # Se estiver próximo o suficiente na vertical, é mesma linha
        if abs(bubble['center'][1] - current_group[-1]['center'][1]) < 20:
            current_group.append(bubble)
        else:
            groups.append(current_group)
            current_group = [bubble]
    
    if current_group:
        groups.append(current_group)
    
    # Ordenar cada grupo horizontalmente (esquerda para direita)
    for group in groups:
        group.sort(key=lambda b: b['center'][0])
    
    return groups

def detect_marked_bubble(bubble_group, binary_img):
    """Detecta qual bolha em um grupo está marcada pela posição (A-E)"""
    if len(bubble_group) != 5:
        return 'ND'  # Deve haver exatamente 5 bolhas

    # Ordenar da esquerda para direita
    bubble_group = sorted(bubble_group, key=lambda b: b['center'][0])

    fill_ratios = []
    for bubble in bubble_group:
        mask = np.zeros_like(binary_img)
        cv2.drawContours(mask, [bubble['contour']], -1, 255, -1)
        masked = cv2.bitwise_and(binary_img, binary_img, mask=mask)
        total_pixels = cv2.countNonZero(mask)
        marked_pixels = cv2.countNonZero(masked)
        fill_ratio = marked_pixels / total_pixels if total_pixels > 0 else 0
        fill_ratios.append(fill_ratio)

    # Considera marcada se o preenchimento for significativamente maior que os outros
    max_idx = np.argmax(fill_ratios)
    max_value = fill_ratios[max_idx]
    sorted_ratios = sorted(fill_ratios, reverse=True)
    if max_value < 0.3:  # Threshold para considerar marcado (ajuste conforme necessário)
        return 'ND'
    if len(sorted_ratios) > 1 and (sorted_ratios[0] - sorted_ratios[1]) < 0.15:
        return 'ND'

    return chr(65 + max_idx)  # A, B, C, D, E

def detectar_resposta(imagem_path, debug=False):
    img = cv2.imread(imagem_path)
    if img is None:
        print(f"Erro ao carregar imagem: {imagem_path}")
        return 'ND'
    binary = preprocess_image(img)
    bubbles = find_bubbles(binary)
    bubble_groups = group_bubbles(bubbles)
    for group in bubble_groups:
        if len(group) == 5:
            resposta = detect_marked_bubble(group, binary)
            if resposta != 'ND':
                return resposta
    # Se não encontrou resposta, tenta abordagem alternativa (quadrados)
    resposta_alt = alternative_detection(img)
    if resposta_alt:
        return resposta_alt
    # Só retorna ND se realmente não houver nada preenchido
    return 'ND'

def alternative_detection(img):
    """Detecção robusta para quadrados preenchidos, garantindo ordem correta"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]
    _, binary = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 100 < area < 3000 and 0.7 < w/h < 1.3:
            bubbles.append((x, y, w, h))
    if len(bubbles) < 5:
        return 'ND'
    # Agrupa por linha (Y próximo)
    bubbles.sort(key=lambda b: b[1])  # ordena por Y
    linha_central = np.median([b[1] + b[3]//2 for b in bubbles])
    bubbles_linha = [b for b in bubbles if abs((b[1] + b[3]//2) - linha_central) < 20]
    if len(bubbles_linha) < 5:
        return 'ND'
    # Ordena da esquerda para direita
    bubbles_linha.sort(key=lambda b: b[0])
    bubbles_linha = bubbles_linha[:5]
    options = []
    for i, (x, y, w, h) in enumerate(bubbles_linha):
        roi = binary[y:y+h, x:x+w]
        fill = cv2.countNonZero(roi) / (w * h)
        options.append((chr(65 + i), fill))
    options.sort(key=lambda x: x[1], reverse=True)
    if options[0][1] > 0.12:  # threshold baixo para considerar preenchido
        return options[0][0]
    return 'ND'

def processar_candidatos(pasta_base="processamento_de_cartoes_de_imagem/questoes"):
    """Processa todos os candidatos e gera arquivo CSV"""
    with open('respostas_candidatos.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        writer.writerow(['questao', 'resposta', 'candidato'])
        
        # Processar cada candidato
        candidatos_dirs = sorted(glob.glob(os.path.join(pasta_base, '*')))
        
        for candidato_dir in candidatos_dirs:
            candidato_id = os.path.basename(candidato_dir)
            print(f"Processando candidato {candidato_id}...")
            
            # Processar cada questão em ordem
            questao_paths = sorted(glob.glob(os.path.join(candidato_dir, 'Q*.png')))
            
            for questao_path in questao_paths:
                questao_num = os.path.basename(questao_path)[1:3]
                try:
                    questao_num = int(questao_num)
                except ValueError:
                    continue
                
                # Detectar resposta (com debug para questões problemáticas)
                debug_mode = questao_num in [18, 49, 50, 54, 59]  # Questões que costumam falhar
                resposta = detectar_resposta(questao_path, debug=debug_mode)
                
                if resposta:
                    writer.writerow([questao_num, resposta, candidato_id])
                else:
                    print(f"⚠ Não detectado: Q{questao_num:02d} - Candidato {candidato_id}")
                    writer.writerow([questao_num, 'ND', candidato_id])

if __name__ == "__main__":
    processar_candidatos()
    print("Processamento concluído. Resultados salvos em 'respostas_candidatos.csv'")