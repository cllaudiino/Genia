""" Esse por enquanto está deprecado, por eu ter tido uma epifania e um foco
    maior no desenvolvimento do algoritmo que treina o modelo, mas em breve
    será otimizado para conseguir encontrar as documentações que eu preciso
    e conseguir compreender os roadmaps para que a genia consiga direcionar
    utilizando toda a informação que puder ter"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re

# URL base da documentação
base_url = "https://devguide.python.org/"

# Nome da pasta para salvar os arquivos
folder_name = "python_dev_guide"

# Diretório para salvar os arquivos
output_dir = os.path.join('data', 'documentation', folder_name)

# Criar o diretório se não existir
os.makedirs(output_dir, exist_ok=True)

# Conjunto para armazenar URLs já visitadas
visited_urls = set()

def sanitize_filename(filename):
    # Remove caracteres indesejados e substitui espaços por underscores
    return re.sub(r'[\\/*?:"<>|]', "", filename).replace(" ", "_").strip()

def extract_and_save_content(url):
    parsed_url = urlparse(url)
    if url in visited_urls or parsed_url.fragment:
        return

    visited_urls.add(url)
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remover scripts e estilos para evitar conteúdo indesejado
        for script_or_style in soup(["script", "style"]):
            script_or_style.extract()
        
        # Extrair o conteúdo do body
        body_content = soup.find('body')
        if body_content:
            # Extrair apenas o texto do body
            text_content = body_content.get_text(separator='\n', strip=True)
            
            # Tentar obter o título da página
            title = soup.title.string if soup.title else "Sem título"
            
            # Criar nome do arquivo baseado no título
            file_name = sanitize_filename(title) + ".txt"
            full_path = os.path.join(output_dir, file_name)
            
            with open(full_path, "w", encoding='utf-8') as file:
                file.write(f"URL: {url}\n\n{text_content}")
            print(f"Arquivo salvo: {full_path}")
            
            # Encontrar e processar links internos apenas do body
            for a_tag in body_content.find_all('a', href=True):
                link = urljoin(base_url, a_tag['href'])
                # Garantir que o link seja interno e não uma âncora interna
                if link.startswith(base_url) and link not in visited_urls:
                    extract_and_save_content(link)
        else:
            print(f"Aviso: Não foi encontrado o conteúdo do body em {url}")
    except Exception as e:
        print(f"Erro ao processar {url}: {e}")

# Iniciar a extração a partir da URL base
extract_and_save_content(base_url)

print(f"\nExtração completa. Os arquivos foram salvos em '{output_dir}'.")

# Verificação final
if os.path.exists(output_dir):
    files = [f for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    if files:
        print(f"\nArquivos salvos (total: {len(files)}):")
        for file in files:
            print(f" - {file}")
    else:
        print(f"\nA pasta '{output_dir}' está vazia. Nenhum arquivo foi salvo.")
else:
    print(f"\nA pasta '{output_dir}' não foi criada.")