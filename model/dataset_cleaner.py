import codecs
import os
import logging

def clean_and_verify_dataset(input_file, output_file=None, clean=False):
    if output_file is None:
        output_file = input_file + '.clean'
    
    cleaning_outcome = {
        "success": False,
        "total_lines": 0,
        "problematic_lines": 0,
        "cleaned_lines": 0,
        "output_file": None,
        "error": None
    }
    
    logging.info(f"{'Limpando' if clean else 'Verificando'} o arquivo: {input_file}")
    
    try:
        with codecs.open(input_file, 'r', encoding='utf-8', errors='replace') as infile, \
             codecs.open(output_file, 'w', encoding='utf-8') as outfile:
            for line_number, line in enumerate(infile, 1):
                cleaning_outcome["total_lines"] += 1
                try:
                    if clean:
                        clean_line = ''.join(char for char in line if char.isprintable() or char.isspace()).strip()
                        if clean_line:
                            outfile.write(clean_line + '\n')
                            cleaning_outcome["cleaned_lines"] += 1
                    else:
                        outfile.write(line)
                except Exception as e:
                    cleaning_outcome["problematic_lines"] += 1
                    logging.warning(f"Erro na linha {line_number}: {e}")
                
                if cleaning_outcome["total_lines"] % 100000 == 0:
                    logging.info(f"Processadas {cleaning_outcome['total_lines']} linhas...")
    
        if cleaning_outcome["problematic_lines"] == 0:
            logging.info(f"Dataset {'limpo e' if clean else ''} verificado com sucesso!")
            if clean:
                os.replace(output_file, input_file)
                cleaning_outcome["output_file"] = input_file
            else:
                cleaning_outcome["output_file"] = output_file
            cleaning_outcome["success"] = True
        else:
            cleaning_outcome["output_file"] = output_file
            cleaning_outcome["success"] = True
            logging.info(f"Dataset {'limpo e' if clean else ''} salvo como: {output_file}")
    
    except Exception as e:
        logging.error(f"Erro ao processar o arquivo: {e}")
        cleaning_outcome["error"] = str(e)
        cleaning_outcome["success"] = False
    
    return cleaning_outcome