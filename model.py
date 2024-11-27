import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Carregar modelos treinados para prever o preço do ouro
model_rf = joblib.load('model_random_forest.pkl')
model_dt = joblib.load('model_decision_tree.pkl')
model_xgb = joblib.load('model_xgb.pkl')
modelos = {'Random Forest': model_rf, 'Decision Tree': model_dt, 'XGB Regressor': model_xgb}

# Título e menu de navegação
st.title('Dashboard de Previsão do Preço do Ouro')
st.sidebar.title('Navegação')
selecao = st.sidebar.radio('Escolha a Seção', [
    'Visão Geral',
    'Previsão do Preço',
    'Gráficos Interativos',
    'Estatísticas Descritivas'
])

df_csv = pd.read_csv('gld_price_data.csv')
df_csv['Date'] = pd.to_datetime(df_csv['Date'], format='%m/%d/%Y')
df_csv['Ano'] = df_csv['Date'].dt.year
df_csv['Mes'] = df_csv['Date'].dt.month


# Seção de Visão Geral
if selecao == 'Visão Geral':
    st.header('Visão Geral')
    st.write('Este dashboard permite explorar diferentes aspectos do modelo de previsão do preço do ouro.')
    st.write('Escolha um modelo para fazer previsões, visualize gráficos interativos e veja estatísticas descritivas.')

    # Classificação em Lote com upload de CSV
    st.subheader('Classificação em Lote com Arquivo CSV')
    uploaded_file = st.file_uploader('Faça o upload de um arquivo CSV com dados dos parâmetros para que o modelo faça a previsão do valor do ouro', type='csv')

    if uploaded_file is not None:
        df_csv = pd.read_csv(uploaded_file)
        st.write('Amostras carregadas:')
        st.write(df_csv.head())
        
        # Escolher o modelo para a previsão
        modelo_selecionado = st.selectbox('Escolha o modelo para previsão em lote', list(modelos.keys()))
        model = modelos[modelo_selecionado]
        
        # Fazer a previsão
        predictions_csv = model.predict(df_csv)
        df_csv['Previsão'] = predictions_csv

        st.subheader('Previsões Geradas')
        st.write(df_csv)

        # Botão para download do CSV com as previsões
        csv = df_csv.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Baixar Previsões em CSV",
            data=csv,
            file_name='previsoes_ouro.csv',
            mime='text/csv'
        )

# Seção de Previsão do Preço
elif selecao == 'Previsão do Preço':
    st.header('Previsão do Preço do Ouro')
    st.write('Escolha um modelo de previsão, insira os dados e visualize o preço do ouro estimado.')

    # Seleção do modelo e entrada de dados para previsão
    modelo_selecionado = st.selectbox('Escolha o modelo para previsão do preço do ouro', list(modelos.keys()))
    model = modelos[modelo_selecionado]

    # Sidebar para entrada de dados (exemplo: parâmetros como taxa de juros, volume de produção, etc.)
    eur_usd = st.sidebar.number_input('EUR/USD', min_value=0.0, value=1.5)
    slv = st.sidebar.number_input('SLV', min_value=0.0, value=15.0)
    spx = st.sidebar.number_input('SPX', min_value=0.0, value=1450.0)
    uso = st.sidebar.number_input('USO', min_value=0.0, value=80.0)

    if st.button('Prever o Preço do Ouro'):
        # Criar DataFrame com os valores inseridos
        data = [[eur_usd, slv, spx, uso]]
        df = pd.DataFrame(data, columns=['SPX', 'USO', 'SLV', 'EUR/USD'])

        # Fazer a previsão
        prediction = model.predict(df)
        
        # Exibir o preço estimado do ouro
        st.write(f'**Modelo Selecionado**: {modelo_selecionado}')
        st.write(f'**Preço Previsto do Ouro**: {prediction[0]:.2f} USD/Ounce')

# Seção de Gráficos Interativos
elif selecao == 'Gráficos Interativos':
    st.header('Gráficos Interativos')
    plot_tipo = st.selectbox('Escolha o tipo de gráfico', ['Gráfico de Linhas', 'Gráfico de Barras', 'Histograma', 'Boxplot'])

    

    if plot_tipo == 'Gráfico de Linhas':
        # Gráfico de linhas da evolução do preço do ouro ao longo do tempo
        st.write('Gráfico de Linhas: Evolução do Preço do Ouro')
        fig, ax = plt.subplots()
        sns.lineplot(x='Date', y='GLD', data=df_csv, ax=ax)  # Usando a coluna GLD (preço do ouro)
        ax.set_title('Evolução do Preço do Ouro')
        st.pyplot(fig)

    elif plot_tipo == 'Gráfico de Barras':
        # Gráfico de barras da média do preço do ouro por ano
        st.write('Gráfico de Barras: Média do Preço do Ouro')
        df_grouped = df_csv.groupby('Ano').mean()  # Agrupar por ano
        fig, ax = plt.subplots()
        df_grouped['GLD'].plot(kind='bar', ax=ax)
        ax.set_ylabel('Preço Médio (USD/Onça)')
        ax.set_title('Preço Médio do Ouro por Ano')
        st.pyplot(fig)

    elif plot_tipo == 'Histograma':
        # Histograma do preço do ouro
        st.write('Histograma: Distribuição do Preço do Ouro')
        fig, ax = plt.subplots()
        sns.histplot(df_csv['GLD'], kde=True, ax=ax)
        ax.set_title('Distribuição do Preço do Ouro')
        st.pyplot(fig)

    elif plot_tipo == 'Boxplot':
        # Boxplot para verificar a variação do preço do ouro
        st.write('Boxplot: Variação do Preço do Ouro')
        fig, ax = plt.subplots()
        sns.boxplot(data=df_csv, x='Ano', y='GLD', ax=ax)  # Agrupar por Ano
        ax.set_title('Variação do Preço do Ouro por Ano')
        st.pyplot(fig)

# Seção de Estatísticas Descritivas
elif selecao == 'Estatísticas Descritivas':
    st.header('Estatísticas Descritivas')
    
    # Exibir estatísticas descritivas dos dados do ouro
    st.subheader('Estatísticas Descritivas do Preço do Ouro')
    df_no_date = df_csv.drop(['Date'], axis=1)
    st.write(df_no_date.describe())

    st.write('As métricas principais incluem média, desvio padrão, valor mínimo e máximo do preço do ouro.')
