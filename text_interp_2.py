import fitz  # PyMuPDF
from crewai import Agent, Task, Crew
from langchain_community.llms import Ollama # Modificado para usar diretamente o Ollama.
from langchain.embeddings import OllamaEmbeddings # Importando o OllamaEmbeddings.
from crewai.tools import BaseTool
import os
import numpy as np

# Ferramenta para Extrair Texto e Gerar Embeddings de PDF
class ExtractTextWithEmbeddingsTool(BaseTool):
    name: str = "Extract Text With Embeddings From PDF"
    description: str = "Extracts text from a PDF file and generates embeddings using nomic-embed-text"

    def _run(self, pdf_path: str) -> dict:
        """Extrai o texto de um arquivo PDF e gera embeddings.

        Args:
            pdf_path (str): O caminho completo para o arquivo PDF.

        Returns:
            dict: Contém o texto extraído, parágrafos e embeddings.
        """
        pdf_file_path = "arquivodotexto.pdf"  # caminho do arquivo.
        if not os.path.exists(pdf_file_path):
            return {"text": f"Erro: O arquivo '{pdf_file_path}' não foi encontrado.", "paragraphs": [], "embeddings": []}
        try:
            doc = fitz.open(pdf_file_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()

            # Dividir texto em parágrafos
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Gerar embeddings com nomic-embed-text via Ollama
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            embeddings_list = []
            for para in paragraphs:
                embeddings_list.append(embeddings.embed_query(para))
            embeddings = np.array(embeddings_list)

            return {"text": text, "paragraphs": paragraphs, "embeddings": embeddings}
        except Exception as e:
            return {"text": f"Erro ao processar o arquivo PDF: {e}", "paragraphs": [], "embeddings": []}

# Configuração do LLM leve - OLLAMA Direto
llm_config = Ollama(
    model="qwen2.5:0.5b",  # Modelo leve especificado
    timeout=40,  # Timeout ajustado para consistência
)

# Agente 1: Extrator de Texto
extractor_agent = Agent(
    role="PDF Text Extractor",
    goal="Extract the text content from a given PDF file and generate embeddings.",
    backstory="You are an expert in document processing, skilled in retrieving text from PDF files and generating embeddings.",
    verbose=True,
    llm=llm_config,
    tools=[ExtractTextWithEmbeddingsTool()],
    allow_delegation=False,
)

# Agente 2: Interpretador de Texto
resume_agent = Agent(
    role="Text Interpreter",
    goal="Present a summary and a list of topics from the extracted text.",
    backstory="You are a master at summarizing texts, extracting the essence of the text, without neglecting the most important arguments.",
    verbose=True,
    llm=llm_config,
    allow_delegation=False,
)

# Tarefa 1: Extrair Texto
extract_task = Task(
    description="""
    Use the tool 'Extract Text With Embeddings From PDF' to extract the text from the PDF.
    Return ONLY the text from the PDF.
    """,
    agent=extractor_agent,
    expected_output="Return the text from the PDF.",
)

# Tarefa 2: Resumir Texto
resume_task = Task(
    description="""
    1. Analyze the text extracted in the previous task.
    2. Present a summary of the analyzed text, synthesizing the essence of the text, without failing to mention each important argument.
    3. Elaborate a list of the 4 most important arguments contained in the text, in bullet point form.
    Restrictions: We do not want information about the author of the text, nor do we want to know about the bibliographic sources. We are only interested in the main subject of the text.
    """,
    agent=resume_agent,
    expected_output="Return the text formatted in string, divided into 2 parts, in an explanatory academic tone."
                    "Part 1 - A summary of the text, with about 100 words;"
                    "Part 2 - A list in bullet point format with the 4 main arguments of the text",
    input_from_tasks=[extract_task]
)

# Criação da equipe
crew = Crew(
    agents=[extractor_agent, resume_agent],
    tasks=[extract_task, resume_task],
    verbose=True,
)

# Executa o Crew
try:
    result = crew.kickoff()
    print("\nFinal Result:")
    print(result)
except Exception as e:
    print(f"Error during execution: {str(e)}")