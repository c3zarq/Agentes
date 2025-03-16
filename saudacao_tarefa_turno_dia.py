from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import requests
from datetime import datetime

# Verifica se o Ollama está rodando
def check_ollama():
    try:
        # Usa o endpoint /api/tags para verificar se o Ollama está ativo
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Ollama está rodando!")
            return True
        else:
            print(f"Erro: Ollama retornou status {response.status_code}. Verifique o servidor.")
            return False
    except requests.ConnectionError as e:
        print(f"Erro de conexão com o Ollama: {str(e)}. Certifique-se de que 'ollama run llama3.1' está ativo em http://localhost:11434.")
        return False
    except requests.Timeout:
        print("Erro: Tempo limite atingido ao tentar conectar ao Ollama. O servidor pode estar sobrecarregado ou não responde.")
        return False

# Executa a verificação
if not check_ollama():
    print("Saindo devido a falha na conexão com o Ollama.")
    exit(1)

# Obtém a data e hora atuais
data_atual = datetime.now().strftime("%d/%m/%Y")
hora_atual = datetime.now().strftime("%H:%M:%S")

# Função para determinar a parte do dia (executada fora do LLM)
def obter_parte_do_dia(hora: str) -> str:
    try:
        hora_int = int(hora.split(":")[0])
        if 5 <= hora_int <12:
            return "manhã"
        elif 12 <= hora_int < 18:
            return "tarde"
        else:
            return "noite"
    except Exception:
        return "indefinido"

# Calcula a parte do dia
parte_do_dia = obter_parte_do_dia(hora_atual)

#Agente 1: Planejador de Tarefas
planejador = Agent(
    role="Planejador de Tarefas",
    goal="Criar uma lista de tarefas para o dia baseada na parte do dia",
    backstory="Você é um organizador eficiente que sugere tarefas adequadas ao horário.",
    verbose=True,
    llm=ChatOpenAI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434/v1",
        api_key="nada",
        temperature=0.7
    ),
    allow_delegation=False,
    max_iter=1
)

# Agente 2: Gerador da Saudação
saudador = Agent(
    role="Gerador da Saudação",
    goal="Criar uma saudação personalizada com base na parte do dia e nas tarefas",
    backstory="Você é um especialista em saudações motivacionais que incorporam o plano do dia.",
    verbose=True,
    llm=ChatOpenAI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434/v1",
        api_key="nada",
        temperature=0.7
    ),
    allow_delegation=False,
    max_iter=1
)

# Agente 3: Revisor de Plano
revisor = Agent(
    role="Revisor de plano",
    goal="Adicionar um comentário motivacional ao plano do dia",
    backstory="Você motiva as pessoas com comentários positivos sobre seus planos.",
    verbose=True,
    llm=ChatOpenAI(
        model_name="ollama/llama3.1",
        base_url="Http://localhost:11434/v1",
        api_key="nada",
        temperature=0.7
    ),
    allow_delegation=False,
    max_iter=1
)

# Tarefa 1: Planejar tarefas
tarefa_planejamento = Task(
    description="""
    ATENÇÃO: Crie RAPIDAMENTE uma lista de 3 tarefas em JSON para o dia em português:
    - Use a parte do dia '{parte_do_dia}' para sugerir-tasks adequadas (ex.: "Planejar o dia" para manhã, "Relaxar" para noite).
    - Retorne um JSON com:
      - "tarefas": uma lista de 3 strings (ex.: ["Tarefa 1", "Tarefa 2", "Tarefa 3"])
    - NÃO INVENTE horários, NÃO DELEGUE, NÃO PENSE MAIS APÓS GERAR O JSON.
    """,
    agent=planejador,
    expected_output="Um JSON com uma lista de 3 tarefas baseada na parte do dia."
)

# Tarefa 2: Criar saudação (ajustada para usar a saída da tarefa anterior)
tarefa_saudacao = Task(
    description="""
    ATENÇÃO: Crie RAPIDAMENTE uma saudação em JSON em português com base nos inputs
    e na saída do Planejador:
    - Nome: '{nome}'
    - Parte do dia: '{parte_do_dia}'
    - Data: '{data}'
    - Hora: '{hora}'
    - Tarefas: use a lista de tarefas gerada pelo Planejador (disponível como contexto
    da tarefa anterior)
    - Retorne um JSON com:
      - "nome": '{nome}'
      - "mensagem": uma saudação como "Boa {parte_do_dia}, {nome}! Hoje você pode: [lista de tarefas separada por vírgulas]"
      - "tarefas": a lista de tarefas recebida do Planejador
      - "data": '{data}'
      - "hora": '{hora}'
    - NÃO INVENTE, NÃO DELEGUE, NÃO PENSE MAIS APÓS GERAR O JSON.
    """,
    agent=saudador,
    expected_output="Um JSON com nome, mensagem, tarefas, data e hora."
)

# Tarefa 3: Adicionar comentário (ajustada para usar a saída da tarefa anterior)
tarefa_revisao = Task(
    description="""
    ATENÇÃO: Adicione RAPIDAMENTE um campo 'comentario' em português ao JSON recebido
    do Saudador:
    - Baseie-se no horário '{hora}' e nas tarefas recebidas do Saudador (disponíveis
    como contexto da tarefa anterior)
    - Exemplo: "Ótimo plano para a noite!" para tarefas noturnas
    - NÃO ALTERE os campos existentes. NÃO DELEGUE. NÃO PENSE MAIS APÓS ADICIONAR
    O COMENTÁRIO.
    - Retorne o JSON atualizado.
    """,
    agent=revisor,
    expected_output="Um JSON com nome, mensagem, tarefas, data, hora e comentario."
)

# Criação do Crew
equipe = Crew(
    agents=[planejador, saudador, revisor],
    tasks=[tarefa_planejamento, tarefa_saudacao, tarefa_revisao],
    verbose=True
)

# Executa o Crew
resultado = equipe.kickoff(
    inputs={
        "nome": "insiraonome",
        "data": data_atual,
        "hora": hora_atual,
        "parte_do_dia": parte_do_dia
    }
)

print("Resultado final:")
print(resultado)