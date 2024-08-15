import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import plotly.express as px
from datetime import datetime
import locale

# Configuração da página
st.set_page_config(page_title="Análise Comercial Completa", layout="wide")

# Configurar o locale para pt_BR
locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')

# Funções de formatação
def formatar_moeda(valor):
    return locale.currency(valor, grouping=True, symbol='R$')

def formatar_numero(valor):
    return locale.format_string('%.2f', valor, grouping=True)

# Função para gerar dados fictícios
@st.cache_data
def gerar_dados():
    np.random.seed(42)
    datas = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    vendas_base = np.random.poisson(lam=200, size=len(datas))
    tendencia = np.linspace(100, 500, len(datas))
    sazonalidade = 50 * np.sin(np.linspace(0, 4 * np.pi, len(datas)))
    vendas = vendas_base + tendencia + sazonalidade
    quantidades = np.random.randint(1, 100, size=len(datas))
    precos = np.random.uniform(50, 500, size=len(datas))
    receita = vendas * precos
    custo = receita * np.random.uniform(0.3, 0.7, size=len(datas))
    lucro = receita - custo
    paises = ["Estados Unidos", "Brasil", "Canadá", "México", "Argentina"]

    dados = pd.DataFrame({
        "Data": datas,
        "Vendas": vendas,
        "Quantidade": quantidades,
        "Preço": precos,
        "Receita": receita,
        "Custo": custo,
        "Lucro": lucro,
        "País": np.random.choice(paises, size=len(datas)),
        "Estado": np.random.choice(["Califórnia", "Texas", "Flórida", "Nova York", "São Paulo", "Ontário"], size=len(datas)),
        "Produto": np.random.choice(["Produto A", "Produto B", "Produto C"], size=len(datas)),
        "Filial": np.random.choice(["Filial 1", "Filial 2", "Filial 3"], size=len(datas)),
        "Vendedor": np.random.choice(["Alice", "Bob", "Charlie", "David", "Eve"], size=len(datas)),
        "Cliente": np.random.choice(["Cliente 1", "Cliente 2", "Cliente 3", "Cliente 4", "Cliente 5"], size=len(datas))
    })

    return dados

# Função para previsão de vendas usando LinearRegression
@st.cache_data
def prever_vendas(dados):
    dados['Dia do Ano'] = dados['Data'].dt.dayofyear
    X = dados[['Dia do Ano']]
    y = dados['Vendas']
    modelo = LinearRegression()
    scores = cross_val_score(modelo, X, y, cv=5)
    modelo.fit(X, y)
    
    datas_futuras = pd.date_range(start="2024-01-01", end="2024-12-31", freq="D")
    dias_futuros = datas_futuras.dayofyear.values.reshape(-1, 1)
    vendas_futuras = modelo.predict(dias_futuros)
    
    dados_futuros = pd.DataFrame({
        "Data": datas_futuras,
        "Vendas": vendas_futuras
    })

    return dados_futuros, np.mean(scores)

# Função genérica para criar gráficos de linha
def criar_grafico_linha(dados, x, y, titulo, is_currency=False):
    fig = px.line(dados, x=x, y=y, title=titulo, markers=True)
    fig.update_traces(mode='lines+markers')
    fig.update_layout(xaxis_title=x, yaxis_title=y, hovermode="x unified")
    
    if is_currency:
        fig.update_traces(hovertemplate="%{x}: " + dados[y].apply(lambda v: formatar_moeda(v)).astype(str))
    else:
        fig.update_traces(hovertemplate="%{x}: " + dados[y].apply(lambda v: formatar_numero(v)).astype(str))

    return fig

# Função genérica para criar gráficos de barra
def criar_grafico_barra(dados, x, y, titulo, is_currency=False):
    fig = px.bar(dados, x=x, y=y, title=titulo)
    if is_currency:
        fig.update_traces(texttemplate=dados[y].apply(lambda v: formatar_moeda(v)), textposition='outside')
    else:
        fig.update_traces(texttemplate=dados[y].apply(lambda v: formatar_numero(v)), textposition='outside')
    fig.update_layout(xaxis_title=x, yaxis_title=y, hovermode="x unified")
    
    return fig

# Função para criar mapa de vendas por país
def criar_mapa_vendas(dados):
    vendas_por_pais = dados.groupby('País').agg({'Receita': 'sum', 'Lucro': 'sum'}).reset_index()
    fig = px.choropleth(
        vendas_por_pais,
        locations='País',
        locationmode='country names',
        color='Receita',
        hover_name='País',
        color_continuous_scale='Blues',
        labels={'Receita': 'Receita Total'},
        title='Receita por País',
        scope="world"
    )
    
    fig.update_layout(geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'))
    fig.update_traces(hovertemplate="%{hovertext}: " + vendas_por_pais['Receita'].apply(lambda v: formatar_moeda(v)).astype(str))
    
    return fig

# Função de análise de produtos
def analise_produtos(dados):
    vendas_por_produto = dados.groupby('Produto').agg({'Vendas': 'sum', 'Quantidade': 'sum', 'Receita': 'sum', 'Lucro': 'sum'}).reset_index()
    fig_vendas_produto = criar_grafico_barra(vendas_por_produto, 'Produto', 'Vendas', 'Vendas por Produto')
    fig_lucro_produto = criar_grafico_barra(vendas_por_produto, 'Produto', 'Lucro', 'Lucro por Produto', is_currency=True)
    
    return fig_vendas_produto, fig_lucro_produto

# Função de análise de clientes
def analise_clientes(dados):
    receita_por_cliente = dados.groupby('Cliente').agg({'Receita': 'sum', 'Lucro': 'sum'}).reset_index()
    fig_receita_cliente = criar_grafico_barra(receita_por_cliente, 'Cliente', 'Receita', 'Receita por Cliente', is_currency=True)
    fig_lucro_cliente = criar_grafico_barra(receita_por_cliente, 'Cliente', 'Lucro', 'Lucro por Cliente', is_currency=True)
    
    return fig_receita_cliente, fig_lucro_cliente

# Função de análise temporal
def analise_temporal(dados):
    dados['Mês'] = dados['Data'].dt.to_period('M').astype(str)
    desempenho_mensal = dados.groupby('Mês').agg({'Receita': 'sum', 'Lucro': 'sum'}).reset_index()
    fig_receita_mensal = criar_grafico_linha(desempenho_mensal, 'Mês', 'Receita', 'Receita Mensal', is_currency=True)
    fig_lucro_mensal = criar_grafico_linha(desempenho_mensal, 'Mês', 'Lucro', 'Lucro Mensal', is_currency=True)
    
    return fig_receita_mensal, fig_lucro_mensal

# Função para análise de correlação
def analise_correlacao(dados):
    corr = dados[['Vendas', 'Quantidade', 'Preço', 'Receita', 'Custo', 'Lucro']].corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação")
    
    fig.update_traces(hovertemplate="%{x} vs %{y}: Correlação: %{z:.2f}", texttemplate="%{z:.2f}")
    
    return fig

# Função principal para exibir o dashboard no Streamlit
def main():
    st.title("Análise Comercial Completa")

    # Gerar dados fictícios
    dados_vendas = gerar_dados()

    # Sidebar com filtros
    st.sidebar.header('Filtros')
    data_inicio = st.sidebar.date_input("Data de Início", dados_vendas['Data'].min())
    data_fim = st.sidebar.date_input("Data de Fim", dados_vendas['Data'].max())
    pais_selecionado = st.sidebar.multiselect('Selecione o País', dados_vendas['País'].unique())
    produto_selecionado = st.sidebar.multiselect('Selecione o Produto', dados_vendas['Produto'].unique())
    vendedor_selecionado = st.sidebar.multiselect('Selecione o Vendedor', dados_vendas['Vendedor'].unique())

    # Aplicar filtros
    dados_filtrados = dados_vendas[
        (dados_vendas['Data'] >= pd.Timestamp(data_inicio)) &
        (dados_vendas['Data'] <= pd.Timestamp(data_fim))
    ]

    if pais_selecionado:
        dados_filtrados = dados_filtrados[dados_filtrados['País'].isin(pais_selecionado)]
    if produto_selecionado:
        dados_filtrados = dados_filtrados[dados_filtrados['Produto'].isin(produto_selecionado)]
    if vendedor_selecionado:
        dados_filtrados = dados_filtrados[dados_filtrados['Vendedor'].isin(vendedor_selecionado)]

    # Prever as vendas futuras
    dados_futuros, score_modelo = prever_vendas(dados_filtrados)

    # Criar gráficos
    fig_vendas = criar_grafico_linha(dados_filtrados, 'Data', 'Vendas', 'Vendas Diárias')
    fig_previsao = criar_grafico_linha(dados_futuros, 'Data', 'Vendas', 'Previsão de Vendas')
    fig_mapa = criar_mapa_vendas(dados_filtrados)
    fig_vendas_produto, fig_lucro_produto = analise_produtos(dados_filtrados)
    fig_receita_cliente, fig_lucro_cliente = analise_clientes(dados_filtrados)
    fig_receita_mensal, fig_lucro_mensal = analise_temporal(dados_filtrados)
    fig_correlacao = analise_correlacao(dados_filtrados)

    # Exibir gráficos no Streamlit
    st.header("Vendas Diárias e Previsão")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_vendas, use_container_width=True)
    with col2:
        st.plotly_chart(fig_previsao, use_container_width=True)

    st.metric("Precisão do Modelo de Previsão", formatar_numero(score_modelo))

    st.header("Mapa de Vendas por País")
    st.plotly_chart(fig_mapa, use_container_width=True)

    st.header("Análise de Produtos")
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(fig_vendas_produto, use_container_width=True)
    with col4:
        st.plotly_chart(fig_lucro_produto, use_container_width=True)

    st.header("Análise de Clientes")
    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(fig_receita_cliente, use_container_width=True)
    with col6:
        st.plotly_chart(fig_lucro_cliente, use_container_width=True)

    st.header("Análise Temporal")
    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(fig_receita_mensal, use_container_width=True)
    with col8:
        st.plotly_chart(fig_lucro_mensal, use_container_width=True)

    st.header("Análise de Correlação")
    st.plotly_chart(fig_correlacao, use_container_width=True)

    # Opção para download dos dados
    st.download_button(
        label="Baixar dados filtrados",
        data=dados_filtrados.to_csv(index=False).encode('utf-8'),
        file_name='dados_filtrados.csv',
        mime='text/csv',
    )

if __name__ == "__main__":
    main()