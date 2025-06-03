import pandas as pd
import matplotlib.pyplot as plt
import math
import re

# === Leitura e pré-processamento dos dados ===
def carregar_dados(path):
    df = pd.read_csv(path, encoding='latin-1')
    colunas_remover = ['Product Image', 'Product Description', 'Order Zipcode', 'Customer Password', 'Customer Email']
    df.drop(columns=colunas_remover, inplace=True)
    df['order date (DateOrders)'] = pd.to_datetime(df['order date (DateOrders)'], errors='coerce')
    df['Date'] = df['order date (DateOrders)'].dt.date
    return df

# === Algoritmo de Bellman-Ford ===
def bellman_ford(N, A, c, s):
    d = {j: float('inf') for j in N}
    p = {j: None for j in N}
    d[s] = 0
    p[s] = 0

    for _ in range(len(N) - 1):
        for (i, j) in A:
            if d[i] != float('inf') and d[j] > d[i] + c[(i, j)]:
                d[j] = d[i] + c[(i, j)]
                p[j] = i

    # Verificação de ciclo negativo
    for (i, j) in A:
        if d[i] != float('inf') and d[j] > d[i] + c[(i, j)]:
            raise ValueError("Ciclo negativo detectado")

    return d, p


# === Utilitários para pesos e mapeamentos ===
def add_weight_mean(df, group_col, new_col):
    df[new_col] = df.groupby(group_col)['Weight'].transform('mean')

def create_mapping(unique_values, prefix='', start=1):
    return {val: f"{prefix}{i+start}" if prefix else i+start for i, val in enumerate(unique_values)}

def create_weight_dict(df, keys_cols, values_col, mapping_dicts=None):
    if mapping_dicts:
        keys = [df[col].map(mapping_dicts[i]) if mapping_dicts[i] else df[col] for i, col in enumerate(keys_cols)]
    else:
        keys = [df[col] for col in keys_cols]
    return dict(zip(zip(*keys), df[values_col]))

def reconstruir_caminho(p, destino):
    caminho = []
    atual = destino
    while atual in p and p[atual] is not None:
        caminho.append(atual)
        atual = p[atual]
    caminho.append(atual)
    caminho.reverse()
    return caminho

def calcular_custo_caminho(caminho, pesos):
    return sum(pesos.get((caminho[i], caminho[i+1]), 0) for i in range(len(caminho) - 1))

# === Processamento principal ===
def processar(df, date_range):
    resultados = {}

    for date in date_range:
        pedidos_dia = df[df['Date'] == date.date()].copy()
        usa_df = pedidos_dia

        unique_city_state = pd.unique(usa_df[['Customer City', 'Customer State']].values.ravel())
        city_mapping = {city: f'C{i+1}' for i, city in enumerate(unique_city_state)}
        df_cities = pd.DataFrame(list(city_mapping.items()), columns=['Cidade', 'Código'])

        # Cálculo de peso
        pedidos_dia['Weight'] = (
            pedidos_dia['Order Item Quantity'] * pedidos_dia['Order Profit Per Order']
        ) / (
            pedidos_dia['Days for shipping (real)'] + 
            pedidos_dia['Late_delivery_risk'] * pedidos_dia['Days for shipment (scheduled)']
        )

        # Médias de peso por agrupamento
        agrupamentos = {
            'Customer City': 'W_ord_city_cus',
            'Customer State': 'W_city_state_cus',
            'Customer Country': 'W_state_country_cus',
            'Order Country': 'W_country_country',
            'Order State': 'W_country_state_ord',
            'Order City': 'W_state_city_ord'
        }

        for grupo, nova_coluna in agrupamentos.items():
            add_weight_mean(pedidos_dia, grupo, nova_coluna)

        # Mapeamentos
        countries = pd.unique(pedidos_dia[['Customer Country', 'Order Country']].values.ravel())
        country_map = create_mapping(countries)

        city_state = pd.unique(pedidos_dia[['Customer City', 'Customer State']].values.ravel())
        cust_city_map = create_mapping(city_state, prefix='C')

        order_city_state = pd.unique(usa_df[['Order City', 'Order State']].values.ravel())
        ord_city_map = create_mapping(order_city_state, prefix='C_d')

        # Criação dos arcos
        arcs = {}
        arcs |= create_weight_dict(pedidos_dia, ['Customer Country', 'Order Country'], 'W_country_country', [country_map]*2)
        arcs |= create_weight_dict(pedidos_dia, ['Customer State', 'Customer Country'], 'W_state_country_cus', [None, country_map])
        arcs |= create_weight_dict(pedidos_dia, ['Customer City', 'Customer State'], 'W_city_state_cus', [cust_city_map, None])
        arcs |= create_weight_dict(pedidos_dia, ['Order Country', 'Order State'], 'W_country_state_ord', [country_map, None])
        arcs |= create_weight_dict(pedidos_dia, ['Order State', 'Order City'], 'W_state_city_ord', [None, ord_city_map])

        arcs_neg = {k: -abs(v) for k, v in arcs.items()}
        arestas = [(u, v) for (u, v) in arcs_neg if pd.notna(u) and pd.notna(v)]
        nos = set(u for u, v in arestas) | set(v for u, v in arestas)

        melhores_caminhos = {}
        for cidade in df_cities['Código']:
            d, p = bellman_ford(nos, arestas, arcs_neg, cidade)
            destinos_validos = [n for n in p if isinstance(n, str) and re.fullmatch(r'C_d\d+', n)]

            melhor_caminho, melhor_custo = [], float('inf')
            for destino in destinos_validos:
                caminho = reconstruir_caminho(p, destino)
                custo = calcular_custo_caminho(caminho, arcs_neg)
                if custo < melhor_custo:
                    melhor_caminho, melhor_custo = caminho, custo

            melhores_caminhos[cidade] = [melhor_caminho, melhor_custo]

        # Filtra caminhos cujo custo é finito
        melhores_validos = {k: v for k, v in melhores_caminhos.items() if math.isfinite(v[1])}

        # Verifica se há algum válido
        if melhores_validos:
            cidade_otima = min(melhores_validos, key=lambda k: melhores_validos[k][1])
        else:
            cidade_otima = None  # Ou alguma outra lógica para lidar com isso

        # cidade_otima = min(melhores_caminhos, key=lambda k: melhores_caminhos[k][1])
        resultados[date] = melhores_caminhos[cidade_otima]

    return resultados

# === Execução ===
df = carregar_dados('dataco.csv')
intervalo_datas = pd.date_range(start='2016-01-01', periods=300)
resultados = processar(df, intervalo_datas)

melhor_dia = min(resultados, key=lambda k: resultados[k][1])
print(melhor_dia)
print(resultados[melhor_dia])
