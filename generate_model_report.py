import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from jinja2 import Template
import datetime

# Função principal para gerar o relatório HTML
def generate_model_report(df, output_html='model_report.html'):
    # --- 1. Pré-processamento ---
    # Ajuste de tipos
    x_int = ['flag_previdencia','flag_estabilidade','flag_promocao_merito_engaj_ult_12m','nota_performance_ult_ano','nota_performance_perfil_ano','flag_pool']
    x_float = ['idade','tempo_empresa','nota_performance_ult_ano','nota_performance_perfil_ano','qtd_horas_extras_pagas_ma12',
               'qtd_horas_extras_pagas_ma3','qtd_sobrecarga_pago_ma12','qtd_sobrecarga_pago_ma3']
    x_str = ['estado_residencia','estado_civil','subgrupo_piramide_atual','estado_civil_alteracao_ult_12m']
    y_pred = 'pred_flag_desligamento_voluntario'
    y_real = 'flag_desligamento_voluntario'
    data_col = 'dt_yearmonth'

    for col in x_int:
        if col in df.columns:
            df[col] = df[col].dropna().astype('int64')
    for col in x_float:
        if col in df.columns:
            df[col] = df[col].astype('float')
    for col in x_str:
        if col in df.columns:
            df[col] = df[col].astype(str)
    if data_col in df.columns:
        df[data_col] = pd.to_datetime(df[data_col], errors='coerce')

    # --- 2. Filtros de data ---
    min_date = df[data_col].min()
    max_date = df[data_col].max()
    # Pré-configurar para o mês mais atual
    default_month = max_date.strftime('%Y-%m') if pd.notnull(max_date) else ''
    # Filtro: por padrão, usa tudo
    df_filtered = df.copy()

    # --- 3. Métricas globais ---
    y_true = df_filtered[y_real]
    y_pred_ = df_filtered[y_pred]
    acc = accuracy_score(y_true, y_pred_)
    prec = precision_score(y_true, y_pred_)
    rec = recall_score(y_true, y_pred_)
    f1 = f1_score(y_true, y_pred_)

    # --- 4. Gráfico de métricas ---
    metrics_fig = go.Figure(data=[
        go.Bar(x=['Acurácia', 'Precisão', 'Recall', 'F1'], y=[acc, prec, rec, f1], marker_color=['#636EFA','#EF553B','#00CC96','#AB63FA'])
    ])
    metrics_fig.update_layout(title='Métricas Globais do Modelo', yaxis=dict(range=[0,1]))
    metrics_html = pio.to_html(metrics_fig, full_html=False, include_plotlyjs='cdn')

    # --- 4.1 Matriz de Confusão ---
    cm = confusion_matrix(y_true, y_pred_)
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Negativo', 'Positivo'],
        y=['Negativo', 'Positivo'],
        colorscale='Blues',
        showscale=True,
        text=cm,
        texttemplate="%{text}"
    ))
    cm_fig.update_layout(title='Matriz de Confusão', xaxis_title='Predito', yaxis_title='Real')
    cm_html = pio.to_html(cm_fig, full_html=False, include_plotlyjs=False)

    # --- KPIs ---
    kpi_cards = f'''
    <div class="kpi-cards">
        <div class="kpi-card"><div class="kpi-label">Acurácia</div><div class="kpi-value">{acc:.2%}</div></div>
        <div class="kpi-card"><div class="kpi-label">Precisão</div><div class="kpi-value">{prec:.2%}</div></div>
        <div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value">{rec:.2%}</div></div>
        <div class="kpi-card"><div class="kpi-label">F1</div><div class="kpi-value">{f1:.2%}</div></div>
    </div>'''

    # --- 5. Data Drift (média móvel do target ou variável escolhida ao longo do tempo) ---
    # Por padrão, usar y_real, mas o HTML terá um dropdown para escolher a variável
    drift_default_var = y_real
    drift_vars = [y_real] + x_float + x_int + x_str
    drift_data = {}
    for var in drift_vars:
        if var in df_filtered.columns:
            if var in x_float or var in x_int:
                drift = df_filtered.groupby(data_col)[var].mean().reset_index()
            else:
                # Para categóricas, pegar a moda (valor mais frequente)
                drift = df_filtered.groupby(data_col)[var].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index()
            drift_data[var] = drift
    # Gera gráfico apenas para o default, os outros serão trocados via JS
    drift_fig = go.Figure()
    drift_fig.add_trace(go.Scatter(x=drift_data[drift_default_var][data_col], y=drift_data[drift_default_var][drift_default_var], mode='lines+markers', name=drift_default_var))
    drift_fig.update_layout(title=f'Data Drift: {drift_default_var} ao Longo do Tempo', xaxis_title='Data', yaxis_title=drift_default_var)
    drift_html = pio.to_html(drift_fig, full_html=False, include_plotlyjs=False, div_id='drift_plot')

    # --- 5. Data Drift: preparar boxplots e proporções ---
    drift_box_html = ''
    drift_prop_html = ''
    # Exemplo: para a variável default
    if drift_default_var in x_float + x_int:
        # Boxplot mensal
        box_df = df_filtered[[data_col, drift_default_var]].dropna()
        if not box_df.empty:
            box_df['mes'] = box_df[data_col].dt.strftime('%Y-%m')
            fig = go.Figure()
            for m in sorted(box_df['mes'].unique()):
                fig.add_trace(go.Box(y=box_df[box_df['mes']==m][drift_default_var], name=m, boxmean=True))
            fig.update_layout(title=f'Boxplot Mensal de {drift_default_var}', xaxis_title='Mês', yaxis_title=drift_default_var)
            drift_box_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)
    elif drift_default_var in x_str:
        # Proporção de categorias por mês
        prop_df = df_filtered[[data_col, drift_default_var]].dropna()
        if not prop_df.empty:
            prop_df['mes'] = prop_df[data_col].dt.strftime('%Y-%m')
            prop = prop_df.groupby(['mes', drift_default_var]).size().reset_index(name='count')
            total = prop.groupby('mes')['count'].transform('sum')
            prop['proportion'] = prop['count'] / total
            fig = go.Figure()
            for cat in prop[drift_default_var].unique():
                fig.add_trace(go.Bar(x=prop[prop[drift_default_var]==cat]['mes'], y=prop[prop[drift_default_var]==cat]['proportion'], name=str(cat)))
            fig.update_layout(barmode='stack', title=f'Proporção de {drift_default_var} por Mês', xaxis_title='Mês', yaxis_title='Proporção')
            drift_prop_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # --- 6. Performance por variável (histograma + linha de densidade) ---
    perf_var_default = x_float[0] if x_float else x_int[0] if x_int else x_str[0]
    perf_var_html = ''
    import scipy.stats as stats
    if perf_var_default in df_filtered.columns:
        vals_0 = df_filtered[df_filtered[y_real] == 0][perf_var_default].dropna().astype(float)
        vals_1 = df_filtered[df_filtered[y_real] == 1][perf_var_default].dropna().astype(float)
        fig = go.Figure()
        # Histograma para 0
        if not vals_0.empty:
            fig.add_trace(go.Histogram(
                x=vals_0,
                name=f'{perf_var_default} | {y_real}=0',
                opacity=0.5,
                marker_color='#636EFA',
                histnorm=None
            ))
            # Linha de densidade para 0
            kde_x = np.linspace(vals_0.min(), vals_0.max(), 200)
            kde_y = stats.gaussian_kde(vals_0)(kde_x)
            kde_y_scaled = kde_y * len(vals_0) * (vals_0.max()-vals_0.min())/30  # escala para sobrepor
            fig.add_trace(go.Scatter(x=kde_x, y=kde_y_scaled, mode='lines', name=f'Densidade {y_real}=0', line=dict(color='#636EFA', width=2)))
        # Histograma para 1
        if not vals_1.empty:
            fig.add_trace(go.Histogram(
                x=vals_1,
                name=f'{perf_var_default} | {y_real}=1',
                opacity=0.5,
                marker_color='#EF553B',
                histnorm=None
            ))
            # Linha de densidade para 1
            kde_x1 = np.linspace(vals_1.min(), vals_1.max(), 200)
            kde_y1 = stats.gaussian_kde(vals_1)(kde_x1)
            kde_y1_scaled = kde_y1 * len(vals_1) * (vals_1.max()-vals_1.min())/30
            fig.add_trace(go.Scatter(x=kde_x1, y=kde_y1_scaled, mode='lines', name=f'Densidade {y_real}=1', line=dict(color='#EF553B', width=2)))
        fig.update_layout(
            barmode='overlay',
            title=f'Distribuição e Densidade de {perf_var_default} por {y_real}',
            xaxis_title=perf_var_default,
            yaxis_title='Contagem',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        perf_var_html = pio.to_html(fig, full_html=False, include_plotlyjs=False)

    # --- 7. Template HTML ---
    html_template = Template('''
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <title>Dashboard de Performance do Modelo de Desligamento Voluntário</title>
        <link href="https://fonts.googleapis.com/css?family=Roboto:400,700&display=swap" rel="stylesheet">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: 'Roboto', sans-serif; background: #f4f6fa; margin: 0; padding: 0; }
            .container { max-width: 1400px; margin: 40px auto; background: #fff; border-radius: 14px; box-shadow: 0 2px 16px #0002; padding: 40px; }
            h1 { color: #222; text-align: center; margin-bottom: 10px; }
            .section { margin-bottom: 40px; }
            .bloco { background: #f7f7fb; border-radius: 10px; box-shadow: 0 1px 4px #0001; padding: 24px; margin-bottom: 32px; }
            .bloco-title { font-size: 1.3em; font-weight: bold; margin-bottom: 18px; text-align: center; }
            .kpi-cards { display: flex; gap: 24px; justify-content: center; margin-bottom: 32px; flex-wrap: wrap; }
            .kpi-card { background: #fff; border-radius: 10px; box-shadow: 0 1px 4px #0001; padding: 24px 32px; text-align: center; min-width: 120px; }
            .kpi-label { color: #888; font-size: 1.1em; margin-bottom: 6px; }
            .kpi-value { color: #222; font-size: 2.1em; font-weight: bold; }
            .filtros { display: flex; gap: 20px; align-items: center; margin-bottom: 20px; flex-wrap: wrap; justify-content: center; }
            .filtros label { font-weight: bold; }
            .filtros select, .filtros input { padding: 4px 8px; border-radius: 4px; border: 1px solid #ccc; }
            @media (max-width: 900px) {
                .kpi-cards { flex-direction: column; align-items: center; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dashboard de Performance do Modelo de Desligamento Voluntário</h1>
            <!-- Bloco 1: Big Numbers e Matriz de Confusão -->
            <div class="bloco">
                <div class="bloco-title">Indicadores Gerais</div>
                <div class="filtros">
                    <label for="kpi_data_inicio">Data início:</label>
                    <input type="month" id="kpi_data_inicio" value="{{ default_month }}">
                    <label for="kpi_data_fim">Data fim:</label>
                    <input type="month" id="kpi_data_fim" value="{{ default_month }}">
                </div>
                <div class="kpi-cards" id="kpi_cards_container">{{ kpi_cards|safe }}</div>
                <div style="max-width:400px; margin:0 auto;" id="cm_html_container">{{ cm_html|safe }}</div>
            </div>
            <!-- Bloco 2: Evolução Temporal da Métrica -->
            <div class="bloco">
                <div class="bloco-title">Evolução Temporal da Métrica</div>
                <div class="filtros">
                    <label for="metrica_data_inicio">Data início:</label>
                    <input type="month" id="metrica_data_inicio" value="{{ min_date }}">
                    <label for="metrica_data_fim">Data fim:</label>
                    <input type="month" id="metrica_data_fim" value="{{ default_month }}">
                    <label for="metrica">Métrica:</label>
                    <select id="metrica">
                        <option value="Acurácia">Acurácia</option>
                        <option value="Precisão">Precisão</option>
                        <option value="Recall">Recall</option>
                        <option value="F1">F1</option>
                    </select>
                </div>
                <div id="metrica_mensal_plot"></div>
            </div>
            <!-- Bloco 3: Data Drift -->
            <div class="bloco">
                <div class="bloco-title">Data Drift</div>
                <div class="filtros">
                    <label for="drift_data_inicio">Data início:</label>
                    <input type="month" id="drift_data_inicio" value="{{ min_date }}">
                    <label for="drift_data_fim">Data fim:</label>
                    <input type="month" id="drift_data_fim" value="{{ default_month }}">
                    <label for="drift_var">Variável:</label>
                    <select id="drift_var">
                        {% for var in drift_vars %}
                            <option value="{{ var }}">{{ var }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div id="drift_plot_container">{{ drift_html|safe }}</div>
                <div id="drift_box_container">{{ drift_box_html|safe }}</div>
                <div id="drift_prop_container">{{ drift_prop_html|safe }}</div>
            </div>
            <!-- Bloco 4: Performance por Variável -->
            <div class="bloco">
                <div class="bloco-title">Performance por Variável</div>
                <div class="filtros">
                    <label for="perf_data_inicio">Data início:</label>
                    <input type="month" id="perf_data_inicio" value="{{ min_date }}">
                    <label for="perf_data_fim">Data fim:</label>
                    <input type="month" id="perf_data_fim" value="{{ default_month }}">
                    <label for="perf_var">Variável:</label>
                    <select id="perf_var">
                        {% for var in x_float + x_int + x_str %}
                            <option value="{{ var }}" {% if var == perf_var_default %}selected{% endif %}>{{ var }}</option>
                        {% endfor %}
                    </select>
                </div>
                <div id="perf_var_plot">{{ perf_var_html|safe }}</div>
            </div>
        </div>
        <script>
            // --- Dados para os filtros ---
            const df = {{ df_filtered.to_json(orient='records', date_format='iso')|safe }};
            const data_col = '{{ data_col }}';
            const y_real = '{{ y_real }}';
            const y_pred = '{{ y_pred }}';
            const drift_vars = {{ drift_vars|tojson }};
            const x_float = {{ x_float|tojson }};
            const x_int = {{ x_int|tojson }};
            const x_str = {{ x_str|tojson }};
            const default_month = '{{ default_month }}';

            // --- Função para atualizar KPIs e Matriz de Confusão ---
            function updateKPIBlock() {
                const dataInicio = document.getElementById('kpi_data_inicio').value;
                const dataFim = document.getElementById('kpi_data_fim').value;
                let dados = df.filter(row => {
                    const dt = row[data_col]?.slice(0,7);
                    return (!dataInicio || dt >= dataInicio) && (!dataFim || dt <= dataFim);
                });
                let y_true = dados.map(r => r[y_real]);
                let y_pred_ = dados.map(r => r[y_pred]);
                let acc = accuracy(y_true, y_pred_);
                let prec = precision(y_true, y_pred_);
                let rec = recall(y_true, y_pred_);
                let f1s = f1(y_true, y_pred_);
                document.getElementById('kpi_cards_container').innerHTML = `
                    <div class="kpi-card"><div class="kpi-label">Acurácia</div><div class="kpi-value">${(acc*100).toFixed(2)}%</div></div>
                    <div class="kpi-card"><div class="kpi-label">Precisão</div><div class="kpi-value">${(prec*100).toFixed(2)}%</div></div>
                    <div class="kpi-card"><div class="kpi-label">Recall</div><div class="kpi-value">${(rec*100).toFixed(2)}%</div></div>
                    <div class="kpi-card"><div class="kpi-label">F1</div><div class="kpi-value">${(f1s*100).toFixed(2)}%</div></div>
                `;
                // Matriz de confusão
                let cm = confusionMatrix(y_true, y_pred_);
                let cm_html = `<table style='width:100%;text-align:center;'><tr><th></th><th>Predito 0</th><th>Predito 1</th></tr><tr><th>Real 0</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr><tr><th>Real 1</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr></table>`;
                document.getElementById('cm_html_container').innerHTML = cm_html;
            }
            document.getElementById('kpi_data_inicio').addEventListener('change', updateKPIBlock);
            document.getElementById('kpi_data_fim').addEventListener('change', updateKPIBlock);
            // --- Funções de métricas JS ---
            function accuracy(y_true, y_pred) {
                let correct = 0;
                for (let i=0; i<y_true.length; i++) if (y_true[i] === y_pred[i]) correct++;
                return y_true.length ? correct/y_true.length : 0;
            }
            function precision(y_true, y_pred) {
                let tp=0, fp=0;
                for (let i=0; i<y_true.length; i++) {
                    if (y_pred[i]===1) {
                        if (y_true[i]===1) tp++; else fp++;
                    }
                }
                return (tp+fp) ? tp/(tp+fp) : 0;
            }
            function recall(y_true, y_pred) {
                let tp=0, fn=0;
                for (let i=0; i<y_true.length; i++) {
                    if (y_true[i]===1) {
                        if (y_pred[i]===1) tp++; else fn++;
                    }
                }
                return (tp+fn) ? tp/(tp+fn) : 0;
            }
            function f1(y_true, y_pred) {
                let p = precision(y_true, y_pred);
                let r = recall(y_true, y_pred);
                return (p && r && (p+r)) ? 2*p*r/(p+r) : 0;
            }
            function confusionMatrix(y_true, y_pred) {
                let cm = [[0,0],[0,0]];
                for (let i=0; i<y_true.length; i++) {
                    if (y_true[i]===0 && y_pred[i]===0) cm[0][0]++;
                    if (y_true[i]===0 && y_pred[i]===1) cm[0][1]++;
                    if (y_true[i]===1 && y_pred[i]===0) cm[1][0]++;
                    if (y_true[i]===1 && y_pred[i]===1) cm[1][1]++;
                }
                return cm;
            }
            // Inicializar KPIs para mês mais atual
            updateKPIBlock();

            // --- Função para atualizar gráfico de métrica mensal (linha contínua) ---
            function updateMetricaMensal() {
                const metrica = document.getElementById('metrica').value;
                const dataInicio = document.getElementById('metrica_data_inicio').value;
                const dataFim = document.getElementById('metrica_data_fim').value;
                let dados = df.filter(row => {
                    const dt = row[data_col]?.slice(0,7);
                    return (!dataInicio || dt >= dataInicio) && (!dataFim || dt <= dataFim);
                });
                // Agrupar por mês
                let grupos = {};
                dados.forEach(row => {
                    const mes = row[data_col]?.slice(0,7);
                    if (!grupos[mes]) grupos[mes] = [];
                    grupos[mes].push(row);
                });
                let meses = Object.keys(grupos).sort();
                let valores = meses.map(mes => {
                    let y_true = grupos[mes].map(r => r[y_real]);
                    let y_pred_ = grupos[mes].map(r => r[y_pred]);
                    if (metrica === 'Acurácia') return accuracy(y_true, y_pred_);
                    if (metrica === 'Precisão') return precision(y_true, y_pred_);
                    if (metrica === 'Recall') return recall(y_true, y_pred_);
                    if (metrica === 'F1') return f1(y_true, y_pred_);
                });
                Plotly.newPlot('metrica_mensal_plot', [{
                    x: meses,
                    y: valores,
                    mode: 'lines+markers',
                    type: 'scatter',
                    marker: {color: '#636EFA'}
                }], {title: `Métrica Mensal: ${metrica}`, yaxis: {range: [0,1]}});
            }
            document.getElementById('metrica').addEventListener('change', updateMetricaMensal);
            document.getElementById('metrica_data_inicio').addEventListener('change', updateMetricaMensal);
            document.getElementById('metrica_data_fim').addEventListener('change', updateMetricaMensal);
            updateMetricaMensal();

            // --- Data Drift ---
            function updateDriftPlot() {
                const varDrift = document.getElementById('drift_var').value;
                const dataInicio = document.getElementById('drift_data_inicio').value;
                const dataFim = document.getElementById('drift_data_fim').value;
                let dados = df.filter(row => {
                    const dt = row[data_col]?.slice(0,7);
                    return (!dataInicio || dt >= dataInicio) && (!dataFim || dt <= dataFim);
                });
                // Agrupar por mês
                let grupos = {};
                dados.forEach(row => {
                    const mes = row[data_col]?.slice(0,7);
                    if (!grupos[mes]) grupos[mes] = [];
                    grupos[mes].push(row);
                });
                let meses = Object.keys(grupos).sort();
                let valores;
                if (x_float.includes(varDrift) || x_int.includes(varDrift)) {
                    valores = meses.map(mes => {
                        let vals = grupos[mes].map(r => r[varDrift]);
                        let sum = vals.reduce((a,b) => a+parseFloat(b), 0);
                        return vals.length ? sum/vals.length : null;
                    });
                } else if (x_str.includes(varDrift)) {
                    valores = meses.map(mes => {
                        let vals = grupos[mes].map(r => r[varDrift]);
                        if (!vals.length) return null;
                        let freq = {};
                        vals.forEach(v => { freq[v] = (freq[v]||0)+1; });
                        return Object.keys(freq).reduce((a,b) => freq[a]>freq[b]?a:b);
                    });
                } else {
                    valores = meses.map(mes => {
                        let vals = grupos[mes].map(r => r[varDrift]);
                        let sum = vals.reduce((a,b) => a+parseFloat(b), 0);
                        return vals.length ? sum/vals.length : null;
                    });
                }
                let trace;
                if (x_str.includes(varDrift)) {
                    trace = {x: meses, y: valores, mode: 'lines+markers', type: 'scatter'};
                } else {
                    trace = {x: meses, y: valores, mode: 'lines+markers', type: 'scatter'};
                }
                Plotly.newPlot('drift_plot_container', [trace], {title: `Data Drift: ${varDrift} ao Longo do Tempo`, xaxis: {title: 'Data'}, yaxis: {title: varDrift}});
            }
            document.getElementById('drift_var').addEventListener('change', updateDriftPlot);
            document.getElementById('drift_data_inicio').addEventListener('change', updateDriftPlot);
            document.getElementById('drift_data_fim').addEventListener('change', updateDriftPlot);
            updateDriftPlot();

            // --- Performance por Variável (KDE) ---
            function updatePerfVarPlot() {
                const perfVar = document.getElementById('perf_var').value;
                const dataInicio = document.getElementById('perf_data_inicio').value;
                const dataFim = document.getElementById('perf_data_fim').value;
                let dados = df.filter(row => {
                    const dt = row[data_col]?.slice(0,7);
                    return (!dataInicio || dt >= dataInicio) && (!dataFim || dt <= dataFim);
                });
                let grupos = {0: [], 1: []};
                dados.forEach(row => {
                    if (row[perfVar] !== undefined && row[y_real] !== undefined) {
                        if (row[y_real] === 0) grupos[0].push(parseFloat(row[perfVar]));
                        if (row[y_real] === 1) grupos[1].push(parseFloat(row[perfVar]));
                    }
                });
                let data = [];
                if (grupos[0].length > 1) {
                    data.push({
                        type: 'histogram',
                        x: grupos[0],
                        opacity: 0.5,
                        name: perfVar + ' | ' + y_real + '=0',
                        histnorm: 'probability density',
                        marker: {color: '#636EFA'}
                    });
                }
                if (grupos[1].length > 1) {
                    data.push({
                        type: 'histogram',
                        x: grupos[1],
                        opacity: 0.5,
                        name: perfVar + ' | ' + y_real + '=1',
                        histnorm: 'probability density',
                        marker: {color: '#EF553B'}
                    });
                }
                Plotly.newPlot('perf_var_plot', data, {barmode: 'overlay', title: `Distribuição de ${perfVar} por ${y_real}`, xaxis: {title: perfVar}, yaxis: {title: 'Densidade'} });
            }
            document.getElementById('perf_var').addEventListener('change', updatePerfVarPlot);
            document.getElementById('perf_data_inicio').addEventListener('change', updatePerfVarPlot);
            document.getElementById('perf_data_fim').addEventListener('change', updatePerfVarPlot);
            updatePerfVarPlot();
        </script>
    </body>
    </html>
    ''')

    html = html_template.render(
        min_date=min_date.strftime('%Y-%m'),
        max_date=max_date.strftime('%Y-%m'),
        default_month=default_month,
        metrics_html=metrics_html,
        cm_html=cm_html,
        drift_html=drift_html,
        drift_box_html=drift_box_html,
        drift_prop_html=drift_prop_html,
        kpi_cards=kpi_cards,
        perf_var_html=perf_var_html,
        perf_var_default=perf_var_default,
        df_filtered=df_filtered,
        data_col=data_col,
        y_real=y_real,
        y_pred=y_pred,
        drift_vars=drift_vars,
        x_float=x_float,
        x_int=x_int,
        x_str=x_str
    )

    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Relatório salvo em: {output_html}')

# Exemplo de uso:
# import pandas as pd
# df = pd.read_csv('seuarquivo.csv')
# generate_model_report(df)
