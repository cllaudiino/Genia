import numpy as np
from joblib import load
import logging
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.cluster import MiniBatchKMeans
from joblib import dump, load
import nltk
from nltk.tokenize import word_tokenize
import os
import mmap
import psutil
import time
import sqlite3
from .dataset_cleaner import clean_and_verify_dataset
from numba import jit

nltk.download('punkt', quiet=True)

@jit(nopython=True)
def fast_cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class ContentRanker:
    def __init__(self, n_clusters=300, batch_size=100000, checkpoint_interval=10800, max_checkpoints=3):
        self.vectorizer = HashingVectorizer(n_features=2**18, alternate_sign=False)
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters, batch_size=batch_size, max_iter=200)
        self.file_path = None
        self.file_size = 0
        self.processed_documents_count = 0
        self.checkpoint_interval = checkpoint_interval
        self.last_checkpoint_time = time.time()
        self.batch_size = batch_size
        self.start_time = None
        self.total_documents = 0
        self.checkpoint_count = 0
        self.max_checkpoints = max_checkpoints
        self.interrupted = False
        self.db_connection = sqlite3.connect('content_ranker.db')
        self.create_database()

    def create_database(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS documents
                          (id INTEGER PRIMARY KEY, content TEXT)''')
        self.db_connection.commit()

    def is_trained(self):
        return self.processed_documents_count > 0

    def check_dataset(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"O arquivo de dataset '{file_path}' não foi encontrado.")
        return True

    def has_checkpoint(self):
        return any(f.startswith('content_ranker_checkpoint_') and f.endswith('.joblib') for f in os.listdir('.'))

    def clear_memory(self):
        import gc
        gc.collect()

    def summarize(self):
        return f"ContentRanker: {'Treinado' if self.is_trained() else 'Não treinado'}, " \
               f"Documentos processados: {self.processed_documents_count}"

    def preprocess_text(self, text):
        return ' '.join(word.lower() for word in word_tokenize(text) if word.isalnum())

    def process_batch(self, batch):
        return [self.preprocess_text(line.strip()) for line in batch if line.strip()]

    def process_in_batches(self, mm, batch_size=100000):
        current_pos = 0
        try:
            while current_pos < self.file_size:
                mm.seek(current_pos)
                raw_batch = mm.read(batch_size * 100)
                try:
                    batch = raw_batch.decode('utf-8', errors='ignore').splitlines()
                    if batch:
                        yield batch
                        current_pos = mm.tell()
                    else:
                        break
                except Exception as e:
                    logging.error(f"Erro ao processar batch na posição {current_pos}: {str(e)}")
                    current_pos = mm.tell()
        except GeneratorExit:
            logging.info("Gerador fechado antecipadamente.")
        finally:
            logging.info("Finalizando processamento em lotes.")

    def check_resources(self):
        mem = psutil.virtual_memory()
        if mem.percent > 90:
            print("\nMemória muito alta (>90%). Pausando o treinamento por 60 segundos.")
            time.sleep(60)
            return False
        return True

    def train(self, file_path, clean=False):
        self.interrupted = False
        mm = None
        training_outcome = {
            "result": None,
            "documents_processed": 0,
            "total_documents": 0
        }
        
        try:
            logging.info(f"Iniciando treinamento com arquivo: {file_path}")
            self.check_dataset(file_path)
            clean_file_path = file_path + '.clean'

            if clean:
                logging.info("Limpando e verificando o dataset...")
                self.file_path = clean_and_verify_dataset(file_path, clean_file_path)
            else:
                if os.path.exists(clean_file_path):
                    logging.info("Usando dataset limpo existente...")
                    self.file_path = clean_file_path
                else:
                    logging.info("Dataset limpo não encontrado. Usando dataset original...")
                    self.file_path = file_path

            self.file_size = os.path.getsize(self.file_path)
            logging.info(f"Tamanho do arquivo: {self.file_size} bytes")
            
            self.total_documents = sum(1 for _ in open(self.file_path, 'r', encoding='utf-8'))
            training_outcome["total_documents"] = self.total_documents
            logging.info(f"Total de documentos: {self.total_documents}")

            checkpoint_loaded = self.load_checkpoint()
            if not checkpoint_loaded:
                logging.info("Iniciando novo treinamento")
                self.start_time = time.time()
                self.processed_documents_count = 0
            else:
                logging.info(f"Retomando treinamento. Documentos já processados: {self.processed_documents_count}")

            self.start_time = time.time()
            self.last_checkpoint_time = time.time()

            with open(self.file_path, 'rb') as file:
                mm = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
                for raw_batch in self.process_in_batches(mm, self.batch_size):
                    if self.interrupted:
                        logging.info("Treinamento interrompido pelo usuário.")
                        training_outcome["result"] = "interrupted"
                        break

                    if not self.check_resources():
                        continue

                    processed_batch = self.process_batch(raw_batch)
                    X = self.vectorizer.transform(processed_batch)
                    self.kmeans.partial_fit(X)
                    
                    self.save_documents_to_db(processed_batch)
                    self.processed_documents_count += len(processed_batch)
                    training_outcome["documents_processed"] = self.processed_documents_count
                    self.print_progress()

                    if time.time() - self.last_checkpoint_time >= self.checkpoint_interval:
                        self.save_checkpoint()

            if not self.interrupted:
                logging.info("Treinamento completo. Salvando modelo final...")
                self.save_model()
                training_outcome["result"] = "completed"
            
            self.clear_memory()
            logging.info(f"Treinamento concluído. Total de documentos processados: {self.processed_documents_count}")
        
        except Exception as e:
            logging.error(f"Erro durante o treinamento: {str(e)}")
            logging.error(f"Tipo de erro: {type(e)}")
            logging.exception("Traceback completo:")
            training_outcome["result"] = "error"
        finally:
            if mm:
                mm.close()

        return training_outcome

    def interrupt(self):
        self.interrupted = True
        logging.info("Sinal de interrupção recebido.")

    def save_documents_to_db(self, documents):
        cursor = self.db_connection.cursor()
        cursor.executemany("INSERT INTO documents (content) VALUES (?)", [(doc,) for doc in documents])
        self.db_connection.commit()

    def print_progress(self):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed_time = time.time() - self.start_time
        processed_documents = self.processed_documents_count

        if self.total_documents > 0:
            progress = (processed_documents / self.total_documents) * 100
            documents_per_second = processed_documents / elapsed_time if elapsed_time > 0 else 0
            estimated_total_time = self.total_documents / documents_per_second if documents_per_second > 0 else 0
            remaining_time = max(0, estimated_total_time - elapsed_time)

            hours, remainder = divmod(int(remaining_time), 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"\rProgresso: {progress:.2f}% ({processed_documents}/{self.total_documents} documentos). "
                f"Tempo restante estimado: {hours:02d}:{minutes:02d}:{seconds:02d}", end="", flush=True)
        else:
            print(f"\rProcessando documentos: {processed_documents}. Tempo decorrido: {elapsed_time:.2f}s", end="", flush=True)

    def save_checkpoint(self):
        self.checkpoint_count += 1
        checkpoint_filename = f'content_ranker_checkpoint_{self.checkpoint_count}.joblib'
        temp_filename = f'{checkpoint_filename}.temp'
        print(f"\nSalvando checkpoint {self.checkpoint_count}...")
        try:
            dump({
                'kmeans': self.kmeans,
                'vectorizer': self.vectorizer,
                'processed_documents_count': self.processed_documents_count,
                'batch_size': self.batch_size,
                'start_time': self.start_time,
                'total_documents': self.total_documents,
                'checkpoint_count': self.checkpoint_count
            }, temp_filename)
            os.replace(temp_filename, checkpoint_filename)
            print(f"Checkpoint {self.checkpoint_count} salvo. Total de documentos processados: {self.processed_documents_count}")
        except Exception as e:
            logging.error(f"Erro ao salvar checkpoint: {str(e)}")
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        finally:
            self.last_checkpoint_time = time.time()

    def load_checkpoint(self):
        checkpoints = sorted(
            [f for f in os.listdir('.') if f.startswith('content_ranker_checkpoint_') and f.endswith('.joblib')],
            key=lambda x: int(x.split('_')[-1].split('.')[0]),
            reverse=True
        )
        
        if not checkpoints:
            logging.info("Nenhum checkpoint encontrado.")
            return False

        for checkpoint_file in checkpoints:
            logging.info(f"Tentando carregar o checkpoint: {checkpoint_file}")
            
            # Verificar integridade do arquivo
            if not self.verify_file_integrity(checkpoint_file):
                logging.warning(f"Checkpoint {checkpoint_file} parece estar corrompido. Tentando o próximo.")
                continue

            try:
                with open(checkpoint_file, 'rb') as f:
                    checkpoint = load(f)
                
                self.kmeans = checkpoint['kmeans']
                self.vectorizer = checkpoint['vectorizer']
                self.processed_documents_count = checkpoint.get('processed_documents_count', 0)
                self.batch_size = checkpoint.get('batch_size', self.batch_size)
                self.start_time = checkpoint.get('start_time', time.time())
                self.total_documents = checkpoint.get('total_documents', self.processed_documents_count)
                self.checkpoint_count = checkpoint.get('checkpoint_count', 0)
                
                logging.info(f"Checkpoint {self.checkpoint_count} carregado com sucesso. "
                            f"Documentos processados: {self.processed_documents_count}")
                return True
            except EOFError:
                logging.error(f"EOFError ao carregar {checkpoint_file}. O arquivo pode estar corrompido.")
            except Exception as e:
                logging.error(f"Erro ao carregar checkpoint {checkpoint_file}: {str(e)}")
                logging.exception("Detalhes do erro:")

        logging.error("Todos os checkpoints falharam ao carregar. Iniciando novo treinamento.")
        return False
    
    def verify_file_integrity(self, file_path):
        try:
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logging.warning(f"Arquivo {file_path} está vazio.")
                return False
            
            with open(file_path, 'rb') as f:
                # Ler os primeiros e últimos bytes para verificar se o arquivo não está truncado
                f.seek(0)
                f.read(1)
                f.seek(-1, 2)
                f.read(1)
            return True
        except Exception as e:
            logging.error(f"Erro ao verificar integridade do arquivo {file_path}: {str(e)}")
            return False

    def load_documents_from_checkpoint(self, documents):
        batch_size = 10000
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            self.save_documents_to_db(batch)
            print(f"\rCarregado {i+len(batch)} documentos...", end="", flush=True)
        print("\nTodos os documentos foram carregados para o banco de dados.")

    def save_model(self):
        dump(self, 'content_ranker_model.joblib')

    def rank_content(self, query):
        query_vec = self.vectorizer.transform([self.preprocess_text(query)]).toarray().flatten()
        cluster = self.kmeans.predict(query_vec.reshape(1, -1))[0]
        
        cursor = self.db_connection.cursor()
        cursor.execute("SELECT id, content FROM documents ORDER BY RANDOM() LIMIT 1000")
        documents = cursor.fetchall()
        
        similarities = []
        for doc_id, doc_content in documents:
            doc_vec = self.vectorizer.transform([doc_content]).toarray().flatten()
            similarity = fast_cosine_similarity(query_vec, doc_vec)
            similarities.append((doc_id, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]

    def extract_relevant_info(self, query, ranked_indices):
        cursor = self.db_connection.cursor()
        relevant_info = []
        for doc_id, _ in ranked_indices:
            cursor.execute("SELECT content FROM documents WHERE id = ?", (doc_id,))
            result = cursor.fetchone()
            if result:
                relevant_info.append(result[0])
        return relevant_info

    def generate_response(self, query, relevant_info):
        if not relevant_info:
            return "Desculpe, não encontrei informações relevantes para sua pergunta."
        
        response = f"Com base na sua pergunta '{query}', encontrei as seguintes informações relevantes:\n\n"
        for info in relevant_info:
            response += f"- {info}\n"
        return response

    def answer_query(self, query):
        logging.info(f"Recebida query: {query}")
        if not self.is_trained():
            logging.warning("Modelo não treinado. Retornando mensagem de erro.")
            return "O modelo ainda não foi treinado. Por favor, conclua o treinamento antes de fazer perguntas."

        try:
            logging.info("Iniciando rank_content")
            ranked_indices = self.rank_content(query)
            logging.info(f"rank_content concluído. Resultados: {ranked_indices}")

            logging.info("Iniciando extract_relevant_info")
            relevant_info = self.extract_relevant_info(query, ranked_indices)
            logging.info(f"extract_relevant_info concluído. Informações relevantes: {relevant_info}")

            logging.info("Gerando resposta")
            response = self.generate_response(query, relevant_info)
            logging.info("Resposta gerada com sucesso")

            return response
        except Exception as e:
            logging.error(f"Erro ao processar query: {str(e)}")
            return f"Ocorreu um erro ao processar sua pergunta: {str(e)}"

    def increment_model(self, new_file_path):
        logging.info(f"Incrementando o modelo com novos dados de: {new_file_path}")
        increment_outcome = {
            "result": None,
            "new_documents_processed": 0
        }
        
        try:
            initial_count = self.processed_documents_count
            training_outcome = self.train(new_file_path, clean=True)
            
            if training_outcome["result"] == "completed":
                increment_outcome["new_documents_processed"] = training_outcome["documents_processed"] - initial_count
                increment_outcome["result"] = "completed"
            elif training_outcome["result"] == "interrupted":
                increment_outcome["new_documents_processed"] = training_outcome["documents_processed"] - initial_count
                increment_outcome["result"] = "interrupted"
            else:
                increment_outcome["result"] = "error"
                
            logging.info("Incremento concluído. O modelo foi atualizado com os novos dados.")
            return increment_outcome
        
        except Exception as e:
            logging.error(f"Erro durante o incremento do modelo: {str(e)}")
            increment_outcome["result"] = "error"
            return increment_outcome

    def __del__(self):
        if hasattr(self, 'db_connection'):
            self.db_connection.close()

def load_model(filename='content_ranker_model.joblib'):
    model = load(filename)
    model.db_connection = sqlite3.connect('content_ranker.db')
    return model