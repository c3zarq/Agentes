from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import requests

# Verifica se o Ollama está rodando
def check_ollama():
    try:
        # Usa o endpoint /api/tags para verificar se o Ollama está ativo
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            print("Olá miserável! :)\n O Ollama está rodando! Táca-le pau!")
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

# Inputs iniciais
topico = input(str("Tópico: "))
papel = input(str("Papel e Contexto: "))
objetivo = input(str("Objetivo: "))
tarefa_input = input(str("Tarefa: "))
especif = input(str("Especificações: "))
saida = input(str("Saída esperada: "))
restricoes = input(str("Restrições: "))
complexidade = input(str("Nível de complexidade de 1 a 10: "))
temper = input(str("Temperatura: "))

#Agente 1: Planejador de Tarefas
montador = Agent(
    role="Montador de Prompt",
    goal="Um prompt estruturado, baseado nos inputs fornecidos",
    backstory="Você é um montador de prompts, que monta prompts estruturados com os strings fornecidos nos inputs",
    verbose=True,
    llm=ChatOpenAI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434/v1",
        api_key="nada",
        temperature=0.1
    ),
    allow_delegation=False,
    max_iter=1
)

# Agente 2: Gerador da Saudação
melhorador = Agent(
    role="Melhorador de prompt",
    goal="efinar o prompt inicial para atingir nível 8 de complexidade intelectual",
    backstory="Você é um especialista em engenharia de prompt, aplicando técnicas avançadas para criar prompts precisos e intelectualmente rigorosos.",
    verbose=True,
    llm=ChatOpenAI(
        model_name="ollama/llama3.1",
        base_url="http://localhost:11434/v1",
        api_key="nada",
        temperature=0.8
    ),
    allow_delegation=False,
    max_iter=3
)

# Tarefa 1: Montar o Prompt inicial
tarefa_montagem = Task(
    description="""
    AATENÇÃO: Crie RAPIDAMENTE um prompt estruturado em português usando os valores exatos fornecidos nos inputs:
    - Tópico: '{topico}'
    - Papel e Contexto: '{papel}'
    - Objetivo: '{objetivo}'
    - Tarefa: '{tarefa_input}'
    - Especificações: '{especif}'
    - Saída esperada: '{saida}'
    - Restrições: '{restricoes}'
    - Nível de Complexidade: '{complexidade}'
    - Temperatura: '{temper}'
    Substitua os placeholders pelos valores reais dos inputs e retorne o prompt como uma string no formato:
    "Tópico: [valor]\nPapel e Contexto: [valor]\nObjetivo: [valor]\nTarefa: [valor]\nEspecificações: [valor]\nSaída esperada: [valor]\nRestrições: [valor]\nNível de Complexidade: [valor]\nTemperatura: [valor]"
    NÃO INVENTE, NÃO DELEGUE, NÃO PENSE MAIS APÓS GERAR O PROMPT.    
    """,
    agent=montador,
    expected_output="O prompt deve ser um texto estruturado no formato string."
)

# Tarefa 2: Melhorar o prompt anterior
tarefa_aperfeicoamento = Task(
    description="""
    ATENÇÃO: Refine RAPIDAMENTE o prompt gerado na tarefa anterior para atingir um nível 8 de complexidade intelectual em português:
    - Receba o prompt inicial da tarefa anterior como contexto.
    - Considere uma escala de 1 a 10, onde 1 é um texto rudimentar (ex.: descrição factual simples) e 10 é um texto de complexidade superior (ex.: ensaio multidisciplinar com argumentos originais). O objetivo é nível 8: elevado rigor intelectual, argumentos articulados e insights profundos.
    - Preserve todos os elementos do prompt inicial (Tópico, Papel e Contexto, Objetivo, Tarefa, Especificações, Saída Esperada, Restrições, Nível de Complexidade, Temperatura), mas reformule cada item com maior detalhamento e sofisticação, mantendo os valores originais dos inputs.
    - Estruture o prompt em formato de texto corrido, começando com: "Você é um [Papel e Contexto refinado]. Sua tarefa é..."
    - NÃO INVENTE novos valores além dos fornecidos, NÃO DELEGUE, NÃO PENSE MAIS APÓS GERAR O PROMPT.
    """,
    agent=melhorador,
    expected_output="Uma string contendo o prompt refinado com os valores dos inputs, em nível 8 de complexidade intelectual."
)

# Criação do Crew
equipe = Crew(
    agents=[montador, melhorador],
    tasks=[tarefa_montagem, tarefa_aperfeicoamento],
    verbose=True
)

# Executa o Crew
resultado = equipe.kickoff(
    inputs={
        "topico": topico,
        "papel": papel,
        "objetivo": objetivo,
        "tarefa_input": tarefa_input,
        "especif": especif,
        "saida": saida,
        "restricoes": restricoes,
        "complexidade": complexidade,
        "temper": temper
    }
)


print("Resultado final:")
print(resultado)