from fpdf import FPDF
import json
from typing import Dict, List

class PokedexPDF(FPDF):
    def header(self):
        self.image('logo_pokemon.png', 10, 8, 33)
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'POKÉDEX - PRIMEIRA GERAÇÃO', 0, 1, 'C')
        self.ln(10)
    
    def add_pokemon_page(self, pokemon: Dict):
        self.add_page()
        
        # Cabeçalho
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, f"#{pokemon['id']:03} {pokemon['nome']}", 0, 1)
        self.ln(5)
        
        # Imagem (opcional - requer internet)
        # self.image(pokemon['imagem'], x=10, y=30, w=50)
        
        # Dados básicos
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 7, f"Tipos: {', '.join(pokemon['tipos'])}")
        self.multi_cell(0, 7, f"Altura: {pokemon['altura']} m | Peso: {pokemon['peso']} kg")
        self.multi_cell(0, 7, f"Descrição: {pokemon['descricao']}")
        self.ln(5)
        
        # Estatísticas
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "Estatísticas de Batalha:", 0, 1)
        self.set_font('Arial', '', 10)
        stats = pokemon['stats']
        self.cell(0, 6, f"HP: {stats['hp']} | Ataque: {stats['ataque']} | Defesa: {stats['defesa']}", 0, 1)
        self.cell(0, 6, f"Ataque Especial: {stats['ataque_especial']} | Defesa Especial: {stats['defesa_especial']}", 0, 1)
        self.cell(0, 6, f"Velocidade: {stats['velocidade']}", 0, 1)
        self.ln(5)
        
        # Fraquezas
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f"Fraquezas: {', '.join(pokemon['fraquezas'])}", 0, 1)
        
        # Evolução
        evolucao = " → ".join(pokemon['evolucao']) if pokemon['evolucao'] else "Não evolui"
        self.cell(0, 10, f"Linha evolutiva: {evolucao}", 0, 1)

def gerar_pdf_completo():
    # Carrega os dados
    with open('pokedex_gen1.json', 'r', encoding='utf-8') as f:
        pokemons = json.load(f)
    
    # Cria PDF
    pdf = PokedexPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Adiciona índice por tipo
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, "Índice por Tipo", 0, 1)
    pdf.ln(10)
    
    tipos = sorted(list({tipo for p in pokemons for tipo in p['tipos']}))
    for tipo in tipos:
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, tipo, 0, 1)
        pdf.set_font('Arial', '', 10)
        for p in [p for p in pokemons if tipo in p['tipos']]:
            pdf.cell(0, 6, f"#{p['id']:03} {p['nome']}", 0, 1)
        pdf.ln(5)
    
    # Adiciona páginas individuais
    for pokemon in sorted(pokemons, key=lambda x: x['id']):
        pdf.add_pokemon_page(pokemon)
    
    pdf.output('pokedex_completa.pdf')
    print("PDF gerado com sucesso!")

if __name__ == "__main__":
    gerar_pdf_completo()