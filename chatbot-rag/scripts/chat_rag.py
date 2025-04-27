import json
import requests
import re
import string
import difflib
from collections import deque
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- Configurações Iniciais ---
with open("pokedex_gen1.json", encoding="utf-8") as f:
    cache_local = json.load(f)

NOMES_VALIDOS = [p["nome"].lower() for p in cache_local]
CORRECOES_API = {
    "nidoran(f)": "nidoran-f",
    "nidoran(m)": "nidoran-m",
    "mr. mime": "mr-mime",
    "farfetch'd": "farfetchd"
}

# Inicializa vetor de embeddings
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="db", embedding_function=embedding)

# --- Memória de Conversa ---
class MemoriaConversa:
    def __init__(self):
        self.historico = deque(maxlen=3)
        self.ultimo_pokemon = None
    
    def adicionar(self, pergunta: str, resposta: str):
        self.historico.append((pergunta, resposta))
    
    def contexto(self) -> str:
        return "\n".join([f"Q: {q}\nA: {a}" for q, a in self.historico])

memoria = MemoriaConversa()

# --- Funções Utilitárias ---
def corrigir_capitalizacao(texto):
    return ' '.join([palavra.capitalize() for palavra in texto.split()])

def limpar_nome(nome):
    return nome.translate(str.maketrans('', '', string.punctuation)).strip()

def corrigir_nome_pokemon(nome_digitado):
    sugestao = difflib.get_close_matches(nome_digitado.lower(), NOMES_VALIDOS, n=1)
    return sugestao[0] if sugestao else nome_digitado.lower()

def normalizar_nome_para_api(nome):
    nome_limpo = nome.lower().strip()
    nome_limpo = CORRECOES_API.get(nome_limpo, nome_limpo)
    return nome_limpo.replace(" ", "-").replace(".", "").replace("'", "")

def extrair_nomes_de_pokemon(texto):
    palavras = texto.split()
    nomes_detectados = []
    for palavra in palavras:
        palavra_limpa = limpar_nome(palavra).lower()
        if palavra_limpa in NOMES_VALIDOS:
            nomes_detectados.append(palavra.capitalize())
    return nomes_detectados

def detectar_intencao(pergunta: str) -> str:
    pergunta = pergunta.lower()
    if re.search(r'\b(ataque|poder|força|dano)\b', pergunta):
        return 'ataque'
    elif re.search(r'\b(defesa|proteção|resistência)\b', pergunta):
        return 'defesa'
    elif re.search(r'\b(tipos?|elemento)\b', pergunta):
        return 'tipo'
    elif re.search(r'\b(habilidades?|poderes?)\b', pergunta):
        return 'habilidade'
    elif re.search(r'\b(evolução|evolui)\b', pergunta):
        return 'evolucao'
    elif re.search(r'\b(fraquezas?|vulnerabilidades?|contra|fracos?)\b', pergunta):
        return 'fraqueza'
    elif re.search(r'\b(local|encontrar|onde acha|habitat)\b', pergunta):
        return 'localizacao'
    elif re.search(r'\b(comparar|vs|versus|mais forte|quem ganha)\b', pergunta):
        return 'comparar'
    else:
        return 'geral'

# --- Sistema de cálculo de fraquezas ---
def calcular_fraquezas(tipos: list) -> dict:
    # Mapeamento simplificado de fraquezas (pode ser expandido)
    tabela_tipos = {
        'Fire': {'Water', 'Rock', 'Ground'},
        'Water': {'Electric', 'Grass'},
        'Electric': {'Ground'},
        'Grass': {'Fire', 'Ice', 'Poison', 'Flying', 'Bug'},
        'Ice': {'Fire', 'Fighting', 'Rock', 'Steel'},
        'Fighting': {'Flying', 'Psychic', 'Fairy'},
        'Poison': {'Ground', 'Psychic'},
        'Ground': {'Water', 'Grass', 'Ice'},
        'Flying': {'Electric', 'Ice', 'Rock'},
        'Psychic': {'Bug', 'Ghost', 'Dark'},
        'Bug': {'Fire', 'Flying', 'Rock'},
        'Rock': {'Water', 'Grass', 'Fighting', 'Ground', 'Steel'},
        'Ghost': {'Ghost', 'Dark'},
        'Dragon': {'Ice', 'Dragon', 'Fairy'},
        'Dark': {'Fighting', 'Bug', 'Fairy'},
        'Steel': {'Fire', 'Fighting', 'Ground'},
        'Fairy': {'Poison', 'Steel'}
    }
    
    fraquezas = set()
    for tipo in tipos:
        if tipo in tabela_tipos:
            fraquezas.update(tabela_tipos[tipo])
    
    return sorted(fraquezas)

# --- Sistema de Busca (RAG) ---
def buscar_na_pokeapi(nome) -> dict:
    try:
        nome_limpo = limpar_nome(nome)
        nome_corrigido = corrigir_nome_pokemon(nome_limpo)
        nome_api = normalizar_nome_para_api(nome_corrigido)

        poke_url = f"https://pokeapi.co/api/v2/pokemon/{nome_api}"
        species_url = f"https://pokeapi.co/api/v2/pokemon-species/{nome_api}"

        poke_response = requests.get(poke_url, timeout=5)
        if poke_response.status_code != 200:
            return None

        poke_data = poke_response.json()
        
        # Processar dados básicos
        stats = {stat["stat"]["name"]: stat["base_stat"] for stat in poke_data.get("stats", [])}
        
        # Processar descrição
        descricao = "Descrição não disponível."
        species_response = requests.get(species_url, timeout=5)
        if species_response.status_code == 200:
            species_data = species_response.json()
            descricao = next(
                (entry["flavor_text"].replace('\n', ' ') 
                for entry in species_data["flavor_text_entries"] 
                if entry["language"]["name"] == "en"),
                "Descrição não encontrada."
            )
            
            # Processar evolução
            evolucao = ["Não evolui"]
            if "evolution_chain" in species_data:
                chain_url = species_data["evolution_chain"]["url"]
                chain_data = requests.get(chain_url, timeout=5).json()
                if chain_data:
                    evolucoes = []
                    def extrair_evolucoes(chain):
                        nome_evo = chain["species"]["name"].capitalize()
                        if nome_evo.lower() != nome_corrigido:
                            evolucoes.append(nome_evo)
                        for evo in chain.get("evolves_to", []):
                            extrair_evolucoes(evo)
                    extrair_evolucoes(chain_data["chain"])
                    if evolucoes:
                        evolucao = evolucoes

        return {
            "nome": nome_corrigido.capitalize(),
            "tipos": [t["type"]["name"].capitalize() for t in poke_data.get("types", [])],
            "habilidades": [h["ability"]["name"].capitalize() for h in poke_data.get("abilities", [])],
            "altura": poke_data.get("height", 0) / 10,
            "peso": poke_data.get("weight", 0) / 10,
            "stats": stats,
            "descricao": descricao,
            "evolucao": evolucao,
            "fonte": "PokéAPI"
        }

    except Exception:
        return None

def buscar_no_json(nome) -> dict:
    nome_corrigido = corrigir_nome_pokemon(limpar_nome(nome))
    for poke in cache_local:
        if poke.get("nome", "").lower() == nome_corrigido:
            return {
                "nome": poke["nome"],
                "tipos": poke.get("tipos", []),
                "habilidades": poke.get("habilidades", []),
                "altura": poke.get("altura", "N/A"),
                "peso": poke.get("peso", "N/A"),
                "stats": {
                    "hp": poke["stats"].get("hp"),
                    "attack": poke["stats"].get("ataque"),
                    "defense": poke["stats"].get("defesa"),
                    "special-attack": poke["stats"].get("ataque_especial"),
                    "special-defense": poke["stats"].get("defesa_especial"),
                    "speed": poke["stats"].get("velocidade")
                },
                "descricao": poke.get("descricao", "Descrição não disponível."),
                "evolucao": poke.get("evolucao", ["Não evolui"]),
                "fonte": "Cache Local"
            }
    return None

# --- Gerador de Respostas ---
def gerar_resposta(intencao: str, dados: dict) -> str:
    nome = dados["nome"]
    
    if intencao == 'ataque':
        return f"⚔️ {nome} tem {dados['stats']['attack']} de ataque base!"
    
    elif intencao == 'defesa':
        return f"🛡️ {nome} tem {dados['stats']['defense']} de defesa base!"
    
    elif intencao == 'tipo':
        return f"🌿 {nome} é do tipo: {', '.join(dados['tipos'])}"
    
    elif intencao == 'habilidade':
        return f"✨ Habilidades de {nome}: {', '.join(dados['habilidades'])}"
    
    elif intencao == 'evolucao':
        return f"🔮 {nome} evolui para: {', '.join(dados['evolucao'])}"
    
    elif intencao == 'fraqueza':
        fraquezas = calcular_fraquezas(dados['tipos'])
        return f"⚠️ {nome} é fraco contra: {', '.join(fraquezas) if fraquezas else 'Nenhum tipo em especial'}"
    
    elif intencao == 'localizacao':
        # Implementação simplificada - pode ser expandida com dados reais
        locais = {
            'Pikachu': 'Floresta de Viridian',
            'Charizard': 'Montanha da Liga Pokémon',
            'Blastoise': 'Lagos do Vale Celeste'
            # Em breve adicionar mais locais com base no quanto o projeto avançar
        }
        local = locais.get(nome, "Localização desconhecida na primeira geração")
        return f"🗺️ {nome} pode ser encontrado em: {local}"
    
    elif intencao == 'comparar':
        contexto = memoria.contexto()
        pokemons = re.findall(r'[A-Z][a-z]+', contexto)
        
        if len(pokemons) >= 2:
            poke1, poke2 = pokemons[:2]
            dados1 = buscar_na_pokeapi(poke1) or buscar_no_json(poke1)
            dados2 = buscar_na_pokeapi(poke2) or buscar_no_json(poke2)
            
            if dados1 and dados2:
                comparacao = []
                for stat in ['attack', 'defense', 'hp', 'speed']:
                    val1 = dados1['stats'].get(stat, 0)
                    val2 = dados2['stats'].get(stat, 0)
                    if val1 > val2:
                        comparacao.append(f"{poke1} tem mais {stat} ({val1} vs {val2})")
                    elif val2 > val1:
                        comparacao.append(f"{poke2} tem mais {stat} ({val2} vs {val1})")
                    else:
                        comparacao.append(f"Empate em {stat} ({val1})")
                
                return (
                    f"⚖️ Comparação entre {poke1} e {poke2}:\n"
                    f"🔥 Ataque: {dados1['stats']['attack']} vs {dados2['stats']['attack']}\n"
                    f"🛡️ Defesa: {dados1['stats']['defense']} vs {dados2['stats']['defense']}\n"
                    f"❤️ HP: {dados1['stats']['hp']} vs {dados2['stats']['hp']}\n"
                    f"⚡ Velocidade: {dados1['stats']['speed']} vs {dados2['stats']['speed']}\n"
                    f"\n🔍 Análise:\n" + "\n".join(comparacao[:3])
                )
        
        return "Por favor, pergunte algo como 'Quem é mais forte: Charizard ou Blastoise?'"
    
    else:  # Resposta completa
        return (
            f"📘 {nome} ({dados['fonte']})\n"
            f"🌿 Tipos: {', '.join(dados['tipos'])}\n"
            f"✨ Habilidades: {', '.join(dados['habilidades'])}\n"
            f"📖 Descrição: {dados['descricao']}\n"
            f"📏 Altura: {dados['altura']}m | Peso: {dados['peso']}kg\n"
            f"⚔️ Stats:\n"
            f"  - HP: {dados['stats']['hp']}\n"
            f"  - Ataque: {dados['stats']['attack']}\n"
            f"  - Defesa: {dados['stats']['defense']}\n"
            f"  - Velocidade: {dados['stats']['speed']}\n"
            f"🔮 Evolução: {', '.join(dados['evolucao'])}"
        )

# --- Função Principal ---
def responder(pergunta):
    # Detecção de intenção
    intencao = detectar_intencao(pergunta)
    
    # Extrair nomes de Pokémon
    nomes_detectados = extrair_nomes_de_pokemon(pergunta)
    
    # Se não detectar nomes, verifica no histórico
    if not nomes_detectados and memoria.ultimo_pokemon:
        nomes_detectados = [memoria.ultimo_pokemon]

    # Confirmação para nomes ambíguos
    if len(nomes_detectados) > 1:
        confirmacao = input(f"Você quis dizer {nomes_detectados[0]} ou {nomes_detectados[1]}? (1/2): ")
        nomes_detectados = [nomes_detectados[0] if confirmacao == '1' else nomes_detectados[1]]

    if not nomes_detectados:
        return "❓ Não identifiquei um Pokémon na sua pergunta. Poderia ser mais específico?"
    
    # Busca dados (RAG)
    dados = None
    for nome in nomes_detectados:
        dados = buscar_na_pokeapi(nome)
        if not dados:
            dados = buscar_no_json(nome)
        if dados:
            memoria.ultimo_pokemon = dados["nome"]
            break
    
    if not dados:
        return f"❌ Não encontrei dados sobre {', '.join(nomes_detectados)}"
    
    # Gera resposta adaptativa
    resposta = gerar_resposta(intencao, dados)
    memoria.adicionar(pergunta, resposta)
    
    return resposta

# --- Interface Gradio ---
if __name__ == "__main__":
    import gradio as gr
    
    def run_chatbot(pergunta, historico):
        resposta = responder(pergunta)
        return [(pergunta, resposta)]
    
    with gr.Blocks(title="🤖 PokéChat - Chatbot Pokémon Inteligente") as app:
        gr.Markdown("# 🤖 PokéChat - Chatbot Pokémon com RAG")
        
        chatbot = gr.Chatbot(height=500)
        msg = gr.Textbox(label="Pergunte sobre um Pokémon")
        clear = gr.Button("Limpar")
        
        def respond(pergunta, chat_historico):
            resposta = responder(pergunta)
            chat_historico.append((pergunta, resposta))
            return "", chat_historico
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    app.launch()