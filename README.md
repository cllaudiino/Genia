# Gênia 🤖
> Assistente de Aprendizado Contínuo em Português

![Status do Projeto](https://img.shields.io/badge/status-em%20desenvolvimento-brightgreen)
![Versão](https://img.shields.io/badge/versão-0.1.0-blue)
![Licença](https://img.shields.io/badge/licença-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)

## 🌟 Visão Geral

A Gênia é uma assistente de IA projetada para processar e compreender documentação técnica em português, com foco especial em programação e tecnologia. Utilizando técnicas avançadas de processamento de linguagem natural, ela aprende continuamente através de diferentes fontes de conhecimento, desde datasets gerais até documentação técnica específica.

## 💡 Por que Gênia?

Em um cenário onde a maioria dos recursos e assistentes de IA são focados em inglês, a Gênia surge como uma alternativa em português, projetada especificamente para:

- Processar e compreender conteúdo técnico em português
- Aprender continuamente com novos materiais
- Fornecer respostas contextualizadas e precisas
- Auxiliar desenvolvedores e técnicos em suas consultas diárias

## 🏗️ Arquitetura

O projeto é estruturado em duas fases principais de desenvolvimento:

### Fase 1: Base de Conhecimento
- [x] Motor de processamento de linguagem natural
- [x] Sistema de treinamento incremental
- [x] Gerenciamento de checkpoints
- [x] Interface básica de consultas

### Fase 2: Especialização Técnica
- [ ] Processamento de PDFs e documentação
- [ ] Aprendizado contextual por domínio
- [ ] Sistema avançado de respostas
- [ ] Citações e referências

## 🛠️ Stack Tecnológica

### Core
- Python 3.12+
- Scikit-learn (MiniBatchKMeans, HashingVectorizer)
- NLTK & Spacy
- SQLite

### Otimização
- Numba (Aceleração de CPU)
- Processamento em lotes
- Gerenciamento de memória otimizado

### Ferramentas
- Poetry (Gerenciamento de dependências)
- Git (Controle de versão)

## 📂 Estrutura do Projeto

```bash
genia/
├── data/                   # Dados de treinamento
│   ├── books/             # PDFs e livros técnicos
│   ├── documentation/     # Docs processada
│   └── npl_datasets/      # Datasets de linguagem
├── model/                 # Core do sistema
│   ├── content_ranker.py  # Ranking e processamento
│   └── dataset_cleaner.py # Limpeza de dados
├── scraper/              # Coleta de dados
│   └── web_scraper.py    # Scraping de documentação
├── utils/                # Utilitários
│   └── pre_processing.py # Preprocessamento
└── main.py              # CLI do sistema
```

## ⚡ Funcionalidades

### Processamento
- Treinamento incremental com gestão de memória
- Checkpoints automáticos
- Validação e limpeza de datasets
- Otimização via processamento paralelo

### Consultas
- Busca semântica avançada
- Respostas contextualizadas
- Rastreamento de fontes
- CLI intuitiva

## 🚀 Começando

### Requisitos
- Python 3.12+
- Poetry

### Setup

```bash
# Clone o repositório
git clone https://github.com/cllaudiino/genia.git
cd genia

# Instale dependências
poetry install

# Ative o ambiente
poetry shell

# Execute
python main.py
```

### Configurando o Treinamento

1. Datasets
```bash
# Adicione seus datasets em
data/npl_datasets/

# Adicione documentação em
data/books/
```

2. Inicie o treinamento através do CLI
3. Monitore os checkpoints automáticos

## 📊 Status do Desenvolvimento

### Concluído ✅
- Sistema base de NLP
- Treinamento incremental
- Sistema de checkpoints
- Interface de consulta

### Em Progresso 🔄
- Processador de documentação técnica
- Sistema de especialização
- Melhorias na interface
- Sistema de citações

## 🔜 Roadmap

### Curto Prazo
- [ ] Processamento de PDFs técnicos
- [ ] Extração de conhecimento específico
- [ ] Sistema de respostas aprimorado

### Médio Prazo
- [ ] Interface web
- [ ] Suporte a múltiplos idiomas
- [ ] Visualização de conhecimento

### Longo Prazo
- [ ] API pública
- [ ] Plugins para IDEs
- [ ] Integração com ferramentas de desenvolvimento

## 🤝 Contribuindo

Contribuições são muito bem-vindas! Se você tem interesse em contribuir:

1. Faça um Fork do projeto
2. Crie sua Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a Branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📜 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## ✨ Agradecimentos

Agradecimento especial ao professor Marcotti pela inspiração no nome do projeto. Suas sugestões me fizeram chegar ao nome genia, o que capturou perfeitamente a essência do que esta assistente representa.

Agradeço também a todos que contribuíram com feedbacks e insights durante o desenvolvimento. Cada sugestão foi valiosa para moldar o projeto no que ele é hoje.

---

📫 **Contato e Suporte**
- [Issues](https://github.com/cllaudiino/genia/issues)
- [Discussions](https://github.com/cllaudiino/genia/discussions)