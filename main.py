from model.content_ranker import ContentRanker, load_model
import os
import logging
import signal

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training.log',
                    filemode='a')

# Variável global para controlar a interrupção
interrupted = False

def signal_handler(signum, frame):
    global interrupted
    interrupted = True
    print("\nOperação interrompida. Retornando ao menu principal...")

# Registrar o manipulador de sinal
signal.signal(signal.SIGINT, signal_handler)

def list_available_datasets():
    """Lista todos os datasets disponíveis no diretório data/npl_datasets/"""
    dataset_dir = os.path.join('data', 'npl_datasets')
    if not os.path.exists(dataset_dir):
        print(f"Diretório {dataset_dir} não encontrado.")
        return []
    
    datasets = [f for f in os.listdir(dataset_dir) if f.endswith('.txt')]
    return datasets

def select_dataset():
    """Permite ao usuário selecionar um dataset disponível ou especificar um novo caminho"""
    print("\n--- Seleção de Dataset ---")
    print("1. Usar dataset existente")
    print("2. Especificar novo caminho")
    
    choice = input("Escolha uma opção: ")
    
    if choice == '1':
        datasets = list_available_datasets()
        if not datasets:
            print("Nenhum dataset encontrado em data/npl_datasets/")
            return select_dataset()
        
        print("\nDatasets disponíveis:")
        for i, dataset in enumerate(datasets, 1):
            print(f"{i}. {dataset}")
        
        try:
            index = int(input("\nSelecione o número do dataset: ")) - 1
            if 0 <= index < len(datasets):
                return os.path.join('data', 'npl_datasets', datasets[index])
            else:
                print("Seleção inválida.")
                return select_dataset()
        except ValueError:
            print("Entrada inválida.")
            return select_dataset()
    
    elif choice == '2':
        path = input("Digite o caminho completo para o dataset: ")
        if os.path.exists(path):
            return path
        else:
            print(f"Arquivo não encontrado: {path}")
            return select_dataset()
    
    else:
        print("Opção inválida.")
        return select_dataset()

def print_menu():
    print("\n--- Menu Principal ---")
    print("1. Conversar com o modelo")
    print("2. Treinar o modelo")
    print("3. Salvar modelo atual")
    print("4. Incrementar modelo com novos dados")
    print("5. Resumo do modelo")
    print("6. Alterar dataset")
    print("7. Sair")
    return input("Escolha uma opção: ")

def chat_with_model(ranker):
    while not ranker.interrupted:
        query = input("\nDigite sua pergunta (ou 'voltar' para retornar ao menu principal): ")
        if query.lower() == 'voltar':
            break
        print("Processando sua pergunta. Isso pode levar alguns segundos...")
        try:
            response = ranker.answer_query(query)
            print("\nResposta:")
            print(response)
        except Exception as e:
            print(f"\nErro ao processar a pergunta: {e}")
            logging.error(f"Erro durante o chat: {e}")
    
    if ranker.interrupted:
        print("\nOperação de chat interrompida.")
        ranker.interrupted = False

def train_model(ranker, dataset_path=None):
    if dataset_path is None:
        dataset_path = select_dataset()
        if not dataset_path:
            print("Nenhum dataset selecionado.")
            return
    
    clean_dataset = input("Limpar o dataset antes do treinamento? (s/N): ").lower() == 's'
    try:
        print(f"Iniciando treinamento com dataset: {dataset_path}")
        print("Isso pode levar algum tempo...")
        training_outcome = ranker.train(dataset_path, clean=clean_dataset)
        
        if training_outcome["result"] == "completed":
            print(f"Treinamento concluído com sucesso!")
            print(f"Processados {training_outcome['documents_processed']} de {training_outcome['total_documents']} documentos.")
        elif training_outcome["result"] == "interrupted":
            print(f"Treinamento interrompido pelo usuário.")
            print(f"Processados {training_outcome['documents_processed']} documentos antes da interrupção.")
        else:
            print("Ocorreu um erro durante o treinamento. Verifique os logs para mais detalhes.")
        
        print(f"Progresso total: {(training_outcome['documents_processed'] / training_outcome['total_documents']) * 100:.2f}%")
    except Exception as e:
        logging.error(f"Erro durante o treinamento: {e}")
        print("Ocorreu um erro durante o treinamento. Verifique os logs para mais detalhes.")

def increment_model(ranker):
    new_dataset_path = select_dataset()
    if not new_dataset_path:
        print("Nenhum dataset selecionado para incremento.")
        return
    
    try:
        increment_outcome = ranker.increment_model(new_dataset_path)
        if increment_outcome["result"] == "completed":
            print(f"Incremento concluído com sucesso!")
            print(f"Novos documentos processados: {increment_outcome['new_documents_processed']}")
        elif increment_outcome["result"] == "interrupted":
            print(f"Incremento interrompido. Novos documentos processados antes da interrupção: {increment_outcome['new_documents_processed']}")
        else:
            print("Ocorreu um erro durante o incremento. Verifique os logs para mais detalhes.")
    except Exception as e:
        logging.error(f"Erro ao incrementar o modelo: {e}")
        print(f"Ocorreu um erro ao incrementar o modelo: {e}")
    finally:
        ranker.interrupted = False

def save_model(ranker):
    try:
        ranker.save_model()
        print("Modelo salvo com sucesso.")
    except Exception as e:
        logging.error(f"Erro ao salvar o modelo: {e}")
        print(f"Ocorreu um erro ao salvar o modelo: {e}")
    
    if ranker.interrupted:
        print("\nOperação de salvamento interrompida.")
        ranker.interrupted = False

def create_new_model():
    checkpoint_interval = int(input("Digite o intervalo de checkpoint em segundos (padrão 3600 (1h)): ") or 3600)
    max_checkpoints = int(input("Digite o número máximo de checkpoints a manter (padrão 3): ") or 3)
    ranker = ContentRanker(checkpoint_interval=checkpoint_interval, max_checkpoints=max_checkpoints)
    return ranker

def load_or_create_model():
    if ContentRanker().has_checkpoint():
        print("Checkpoints encontrados. Carregando o mais recente...")
        ranker = ContentRanker()
        ranker.load_checkpoint()
        return ranker
    else:
        print("Nenhum checkpoint encontrado. Criando novo modelo...")
        return create_new_model()

def main():
    # Configuração inicial
    ranker = load_or_create_model()
    current_dataset = None

    def signal_handler(signum, frame):
        print("\nOperação interrompida. Retornando ao menu principal...")
        if hasattr(ranker, 'interrupt'):
            ranker.interrupt()

    # Registrar o manipulador de sinal
    signal.signal(signal.SIGINT, signal_handler)

    # Verifica se o modelo está treinado
    if not ranker.is_trained():
        print("O modelo ainda não foi treinado.")
        train_now = input("Deseja treinar o modelo agora? (S/n): ").lower() != 'n'
        if train_now:
            current_dataset = select_dataset()
            if current_dataset:
                train_model(ranker, current_dataset)
        else:
            print("Aviso: O modelo não está treinado. Algumas funcionalidades podem não estar disponíveis.")

    while True:
        try:
            choice = print_menu()
            if choice == '1':
                if ranker.is_trained():
                    chat_with_model(ranker)
                else:
                    print("O modelo não está treinado. Por favor, treine o modelo primeiro.")
            elif choice == '2':
                current_dataset = select_dataset()
                if current_dataset:
                    train_model(ranker, current_dataset)
            elif choice == '3':
                save_model(ranker)
            elif choice == '4':
                increment_model(ranker)
            elif choice == '5':
                print(ranker.summarize())
            elif choice == '6':
                current_dataset = select_dataset()
                print(f"Dataset atual alterado para: {current_dataset}")
            elif choice == '7':
                print("Encerrando o programa.")
                break
            else:
                print("Opção inválida. Por favor, tente novamente.")
        except KeyboardInterrupt:
            print("\nOperação interrompida. Retornando ao menu principal...")
        finally:
            if hasattr(ranker, 'interrupted'):
                ranker.interrupted = False

if __name__ == "__main__":
    main()