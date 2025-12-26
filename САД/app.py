"""
Это веб-приложение предоставляет интерфейс для выполнения
методов анализа данных, включая сравнение групп, корреляционный анализ,
регрессионный анализ и анализ распределений. Для каждого метода реализованы
соответствующие веб-страницы с возможностью ввода данных и визуализации результатов.

Группы методов:
- Сравнение групп (Group Comparison)
- Корреляция (Correlation)
- Регрессия (Regression)
- Описательная статистика / Распределения (Descriptive Stats / Distributions)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from flask import Flask, render_template, request
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import (
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import io
import base64
import matplotlib.pyplot as plt
app = Flask(__name__)

# --- Вспомогательные функции ---
def parse_input_data(input_str):
    """Разбор строки с разделителями-запятыми в числовой массив"""
    try:
        return [float(x.strip()) for x in input_str.split(",") if x.strip()]
    except ValueError:
        return []

app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size


def process_uploaded_file(file):
    try:
        filename = file.filename.lower()

        if filename.endswith(".csv"):
            # Попробуем разные кодировки для CSV
            try:
                df = pd.read_csv(file, encoding="utf-8")
            except UnicodeDecodeError:
                file.seek(0)  # Вернуться к началу файла
                try:
                    df = pd.read_csv(file, encoding="cp1251")
                except UnicodeDecodeError:
                    file.seek(0)
                    df = pd.read_csv(file, encoding="latin-1")

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            return None, "Неподдерживаемый формат файла. Используйте CSV, XLS или XLSX."

        if df.empty:
            return None, "Файл пуст или не содержит данных."

        return df, None

    except Exception as e:
        return None, f"Ошибка при чтении файла: {str(e)}"


def extract_data_from_dataframe(df, columns=None):
    try:
        if columns is None:
            # Используем все числовые колонки
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                return None, "В файле не найдено числовых колонок."
            columns = numeric_columns

        # Извлекаем данные и удаляем NaN значения
        data = {}
        for col in columns:
            if col in df.columns:
                values = df[col].dropna().tolist()
                if values:  # Проверяем, что есть данные после удаления NaN
                    data[col] = values

        if not data:
            return None, "Не удалось извлечь числовые данные из файла."

        return data, None

    except Exception as e:
        return None, f"Ошибка при извлечении данных: {str(e)}"

def create_anova_plots(df, labels):
    """Создание графиков визуализации ANOVA"""
    try:
        if df.empty or not labels:
            empty_html = "<div>Нет данных для построения графика</div>"
            return {k: empty_html for k in ["summary_plot", "boxplot", "distribution_plot"]}

        # 1. Сводный график
        summary_fig = make_subplots(rows=1, cols=2, subplot_titles=("Ящичковая диаграмма", "Средние ± Станд. отклонение"))
        for i, group in enumerate(labels):
            group_data = df[df["Group"] == group]["Value"].dropna()
            if not group_data.empty:
                summary_fig.add_trace(go.Box(
                    y=group_data, name=group, boxpoints="all", jitter=0.3, pointpos=-1.8,
                    marker_color=f"hsl({i * 40}, 50%, 50%)", line_color=f"hsl({i * 40}, 30%, 30%)",
                ), row=1, col=1)
                mean_val = np.mean(group_data)
                std_val = np.std(group_data)
                summary_fig.add_trace(go.Scatter(
                    x=[group], y=[mean_val], mode="markers",
                    marker=dict(size=10, color=f"hsl({i * 40}, 50%, 50%)"),
                    error_y=dict(type="data", array=[std_val], visible=True),
                    showlegend=False, name=f"Среднее {group}"
                ), row=1, col=2)
        summary_fig.update_layout(height=400, title_text="Сводная статистика")

        # 2. Ящичковая диаграмма
        boxplot = go.Figure()
        for i, group in enumerate(labels):
            group_data = df[df["Group"] == group]["Value"].dropna()
            if not group_data.empty:
                boxplot.add_trace(go.Box(
                    y=group_data, name=group, boxpoints="all", jitter=0.3, pointpos=-1.8,
                    marker_color=f"hsl({i * 40}, 50%, 50%)", line_color=f"hsl({i * 40}, 30%, 30%)",
                ))
        boxplot.update_layout(title="Сравнение ящичковых диаграмм", height=400)

        # 3. График распределения
        dist_fig = go.Figure()
        for i, group in enumerate(labels):
            group_data = df[df["Group"] == group]["Value"].dropna()
            if not group_data.empty:
                dist_fig.add_trace(go.Histogram(
                    x=group_data, name=group, opacity=0.6,
                    marker_color=f"hsl({i * 40}, 50%, 50%)",
                ))
        dist_fig.update_layout(
            title="Распределение по рядам", barmode="overlay", height=400,
            xaxis_title="Значения", yaxis_title="Количество",
        )

        return {
            "summary_plot": summary_fig.to_html(full_html=False, include_plotlyjs="cdn"),
            "boxplot": boxplot.to_html(full_html=False, include_plotlyjs=False),
            "distribution_plot": dist_fig.to_html(full_html=False, include_plotlyjs=False),
        }
    except Exception as e:
        error_html = f"<div>Ошибка при генерации графиков: {str(e)}</div>"
        return {k: error_html for k in ["summary_plot", "boxplot", "distribution_plot"]}


def create_shapiro_plots(data):
    """Создание графиков для теста нормальности Шапиро-Уилка"""
    try:
        # Гистограмма
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=data, nbinsx=20, name="Данные", opacity=0.7))
        hist.update_layout(title="Распределение данных", xaxis_title="Значение", yaxis_title="Частота", height=300)
        # Q-Q график
        sorted_data = np.sort(data)
        n = len(sorted_data)
        quantiles = [(i - 0.5) / n for i in range(1, n + 1)]
        theoretical_quantiles = stats.norm.ppf(quantiles, loc=np.mean(data), scale=np.std(data))
        qq_fig = go.Figure()
        qq_fig.add_trace(go.Scatter(x=theoretical_quantiles, y=sorted_data, mode="markers", name="Точки Q-Q"))
        min_val = min(min(theoretical_quantiles), min(sorted_data))
        max_val = max(max(theoretical_quantiles), max(sorted_data))
        qq_fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode="lines", name="y=x", line=dict(color="red")))
        qq_fig.update_layout(title="Сравнение с нормальным распределением", xaxis_title="Теоретические квантили", yaxis_title="Выборочные квантили", height=400)
        return {
            "distplot": hist.to_html(full_html=False, include_plotlyjs=False),
            "qqplot": qq_fig.to_html(full_html=False, include_plotlyjs=False),
        }
    except Exception as e:
        error_html = f"<div>Ошибка при генерации графиков: {str(e)}</div>"
        return {k: error_html for k in ["distplot", "qqplot"]}

def create_ttest_plots(data1, data2):
    """Создание графиков для сравнения t-теста"""
    try:
        # Сравнение ящичковых диаграмм
        boxplot = go.Figure()
        boxplot.add_trace(go.Box(y=data1, name="Ряд 1", boxpoints="all", marker_color="rgba(55, 128, 191, 0.7)"))
        boxplot.add_trace(go.Box(y=data2, name="Ряд 2", boxpoints="all", marker_color="rgba(214, 39, 40, 0.7)"))
        boxplot.update_layout(title="Сравнение ящичковых диаграмм", height=400)
        # Сравнение гистограмм
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=data1, name="Ряд 1", opacity=0.7, marker_color="rgba(55, 128, 191, 0.7)"))
        hist.add_trace(go.Histogram(x=data2, name="Ряд 2", opacity=0.7, marker_color="rgba(214, 39, 40, 0.7)"))
        hist.update_layout(title="Сравнение распределений", barmode="overlay", xaxis_title="Значение", yaxis_title="Частота", height=400)
        return {
            "boxplot": boxplot.to_html(full_html=False, include_plotlyjs=False),
            "histogram": hist.to_html(full_html=False, include_plotlyjs=False),
        }
    except Exception as e:
        error_html = f"<div>Ошибка при генерации графиков: {str(e)}</div>"
        return {k: error_html for k in ["boxplot", "histogram"]}

def create_mannwhitney_plots(data1, data2):
    """Создание графиков для U-теста Манна-Уитни"""
    try:
        # Сравнение ящичковых диаграмм (такой же, как для t-теста)
        boxplot = go.Figure()
        boxplot.add_trace(go.Box(y=data1, name="Ряд 1", boxpoints="all", marker_color="rgba(55, 128, 191, 0.7)"))
        boxplot.add_trace(go.Box(y=data2, name="Ряд 2", boxpoints="all", marker_color="rgba(214, 39, 40, 0.7)"))
        boxplot.update_layout(title="Сравнение ящичковых диаграмм", height=400)
        # Гистограммы (такие же, как для t-теста)
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=data1, name="Ряд 1", opacity=0.7, marker_color="rgba(55, 128, 191, 0.7)"))
        hist.add_trace(go.Histogram(x=data2, name="Ряд 2", opacity=0.7, marker_color="rgba(214, 39, 40, 0.7)"))
        hist.update_layout(title="Сравнение распределений", barmode="overlay", xaxis_title="Значение", yaxis_title="Частота", height=400)
        return {
            "boxplot": boxplot.to_html(full_html=False, include_plotlyjs=False),
            "histogram": hist.to_html(full_html=False, include_plotlyjs=False),
        }
    except Exception as e:
        error_html = f"<div>Ошибка при генерации графиков: {str(e)}</div>"
        return {k: error_html for k in ["boxplot", "histogram"]}

def create_correlation_plots(x, y, xlabel, ylabel):

    try:
        # Преобразование в массивы numpy и удаление NaN/inf
        x = np.array(x, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        # Расчет надежной корреляции
        r_value, p_value = stats.pearsonr(x, y)
        # --- Точечная диаграмма с выделением выбросов ---
        scatter = go.Figure()
        # Расчет IQR для обнаружения выбросов
        q1_x, q3_x = np.percentile(x, [25, 75])
        iqr_x = q3_x - q1_x
        q1_y, q3_y = np.percentile(y, [25, 75])
        iqr_y = q3_y - q1_y
        # Отделение выбросов
        normal_mask = ((x >= q1_x - 1.5*iqr_x) & (x <= q3_x + 1.5*iqr_x) &
                      (y >= q1_y - 1.5*iqr_y) & (y <= q3_y + 1.5*iqr_y))
        # Добавление обычных точек
        scatter.add_trace(go.Scatter(
            x=x[normal_mask],
            y=y[normal_mask],
            mode="markers",
            name="Обычные данные",
            marker=dict(
                color="rgba(55, 128, 191, 0.7)",
                size=10,
                line=dict(width=1, color="rgba(0, 0, 0, 0.8)")
            ),
            hovertemplate=f"{xlabel}: %{{x}}<br>{ylabel}: %{{y}}<extra></extra>"
        ))
        # Добавление выбросов, если они существуют (исправленный маркер)
        if not all(normal_mask):
            scatter.add_trace(go.Scatter(
                x=x[~normal_mask],
                y=y[~normal_mask],
                mode="markers",
                name="Выбросы",
                marker=dict(
                    color="red",
                    size=12,
                    symbol="x",  # Исправлено с 'x-thick' на допустимое значение 'x'
                    line=dict(width=2)
                ),
                hovertemplate=f"Выброс: {xlabel}=%{{x}} {ylabel}=%{{y}}<extra></extra>"
            ))
        # Добавление линии регрессии
        coeffs = np.polyfit(x[normal_mask], y[normal_mask], 1)
        reg_line = np.poly1d(coeffs)(np.linspace(min(x), max(x), 100))
        scatter.add_trace(go.Scatter(
            x=np.linspace(min(x), max(x), 100),
            y=reg_line,
            mode="lines",
            name=f"Регрессия (r={r_value:.2f})",
            line=dict(color="blue", width=2),
            hovertemplate=f"Регрессия: y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}<extra></extra>"
        ))
        scatter.update_layout(
            title="Точечный график с линией тренда",
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            hovermode="closest",
            showlegend=True,
            height=500,
            margin=dict(l=50, r=50, b=50, t=50, pad=4)
        )
        # --- Гистограммы с бинами выбросов ---
        def create_hist(data, title):
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=data,
                name=title,
                xbins=dict(
                    start=min(data),
                    end=max(data),
                    size=(max(data)-min(data))/20
                ),
                marker_color="rgba(55, 128, 191, 0.7)",
                hovertemplate="Значение: %{x}<br>Количество: %{y}<extra></extra>"
            ))
            fig.update_layout(
                title=" ",
                xaxis_title="Значение",
                yaxis_title="Частота",
                bargap=0.05,
                height=400,
                margin=dict(l=50, r=50, b=50, t=50, pad=4)
            )
            return fig
        hist1 = create_hist(x, xlabel)
        hist2 = create_hist(y, ylabel)
        return {
            "scatter": scatter.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False
                }
            ),
            "hist1": hist1.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False
                }
            ),
            "hist2": hist2.to_html(
                full_html=False,
                include_plotlyjs=False,
                config={
                    "displayModeBar": True,
                    "responsive": True,
                    "displaylogo": False
                }
            ),
            "stats": {
                "r_value": r_value,
                "p_value": p_value,
                "outliers_count": len(x) - sum(normal_mask)
            }
        }
    except Exception as e:
        import traceback
        return {
            "error": f"Ошибка при генерации графиков: {str(e)}",
            "traceback": traceback.format_exc()
        }

def create_regression_plot(x, y, x_range, y_range, title):
    """Создание графика для линейной/полиномиальной регрессии"""
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="Точки данных"))
        fig.add_trace(go.Scatter(x=x_range, y=y_range, mode="lines", name="Линия регрессии"))
        fig.update_layout(title=title, xaxis_title="X", yaxis_title="Y", height=500)
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        print(f"Ошибка при создании графика: {str(e)}")
        return None

def create_logistic_plot(x, y, model):
    """Создание графика для логистической регрессии"""
    try:
        fig = go.Figure()
        # Точечная диаграмма для точек данных
        fig.add_trace(go.Scatter(
            x=x.flatten(),
            y=y,
            mode="markers",
            marker=dict(
                color=y,
                colorscale="Viridis",
                showscale=False
            ),
            name="Точки данных",
        ))
        # Граница решения (предполагая 1D вход для простоты визуализации)
        x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_proba = model.predict_proba(x_range)[:, 1]
        fig.add_trace(go.Scatter(
            x=x_range.flatten(),
            y=y_proba,
            mode="lines",
            name="Граница решения",
            line=dict(color="red")
        ))
        fig.update_layout(
            title="Граница решения логистической регрессии (1D проекция)",
            xaxis_title="Признак 1",
            yaxis_title="Вероятность",
            showlegend=True,
            autosize=True,
            height=500
        )
        return fig.to_html(full_html=False, include_plotlyjs="cdn")
    except Exception as e:
        print(f"Ошибка при создании графика: {str(e)}")
        return None

def create_levene_plots(df, labels):
    """Создание графиков для теста Левена"""
    return create_anova_plots(df, labels)

def create_fisher_plot(a, b, c, d):
    """Создание графика для результатов точного теста Фишера"""
    labels = ["Группа 1", "Группа 2"]
    success = [a, c]
    failure = [b, d]
    fig = go.Figure(data=[
        go.Bar(name="Успех", x=labels, y=success, marker_color="rgba(55, 128, 191, 0.7)"),
        go.Bar(name="Неудача", x=labels, y=failure, marker_color="rgba(214, 39, 40, 0.7)"),
    ])
    fig.update_layout(title="Визуализация таблицы сопряжённости", barmode="group", yaxis_title="Количество", height=400)
    return {"barplot": fig.to_html(full_html=False, include_plotlyjs=False)}

def create_tukey_plot(df, labels):
    """Создание ящичковой диаграммы для результатов Tukey HSD"""
    try:
        fig = go.Figure()
        if df.empty or not labels:
            return {"tukeyplot": "<div>Нет данных для построения графика</div>"}
        for i, group in enumerate(labels):
            group_data = df[df["Group"] == group]["Value"]
            if not group_data.empty:
                fig.add_trace(go.Box(
                    y=group_data,
                    name=group,
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=-1.8,
                    marker_color=f"hsl({i * 40}, 50%, 50%)",
                    line_color=f"hsl({i * 40}, 30%, 30%)",
                ))
        fig.update_layout(
            title="Ящичковая диаграмма для теста Тьюки",
            height=400,
            xaxis_title="Ряды",
            yaxis_title="Значения",
            showlegend=False
        )
        # Первый график должен включать библиотеку Plotly
        if len(labels) > 0:
            return {"tukeyplot": fig.to_html(full_html=False, include_plotlyjs="cdn")}
        else:
            return {"tukeyplot": fig.to_html(full_html=False, include_plotlyjs=False)}
    except Exception as e:
        error_html = f"<div>Ошибка при генерации графика: {str(e)}</div>"
        return {"tukeyplot": error_html}

def create_iqr_plots(data):
    """Создание графиков для анализа межквартильного размаха"""
    try:
        # Box Plot
        boxplot = go.Figure()
        boxplot.add_trace(go.Box(y=data, name="Данные", boxpoints="all"))
        boxplot.update_layout(title="Ящичковая диаграмма", height=400)

        # Violin Plot
        violin = go.Figure()
        violin.add_trace(go.Violin(y=data, name="Данные", box_visible=True, meanline_visible=True))
        violin.update_layout(title="Скрипичная диаграмма", height=400)

        # Histogram
        hist = go.Figure()
        hist.add_trace(go.Histogram(x=data, name="Данные", opacity=0.7))
        hist.update_layout(title="Распределение", xaxis_title="Значение", yaxis_title="Частота", height=400)

        return {
            "boxplot": boxplot.to_html(full_html=False, include_plotlyjs=False),
            "violin": violin.to_html(full_html=False, include_plotlyjs=False),
            "histogram": hist.to_html(full_html=False, include_plotlyjs=False),
        }

    except Exception as e:
        error_html = f"<div>Ошибка при генерации графиков: {str(e)}</div>"
        # Для отладки, можете напечатать ошибку в консоль
        print(f"Error creating plots: {e}")
        return {k: error_html for k in ["boxplot", "violin", "histogram"]}

# --- Маршруты Flask ---

@app.route("/")
def home():
    return render_template("index.html", show_results=False)

# --- Сравнение групп (Group Comparison) ---

@app.route("/anova", methods=["GET", "POST"])
def anova():
    initial_groups = ["1, 2, 3", "4, 5, 6", "7, 8, 9"]
    anova_result = None
    plots = None
    conclusion = ""
    loaded_groups = []

    if request.method == "POST":
        # Проверяем, загружается ли файл
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data_dict, error = extract_data_from_dataframe(df, df.columns.tolist())
                if data_dict is not None:
                    for col in data_dict:
                        if data_dict[col]:
                            group_values = ",".join(map(str, data_dict[col]))
                            loaded_groups.append(group_values)
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    return render_template("group_comparison/anova.html", error=error)
            else:
                # Обработка ошибки process_uploaded_file
                return render_template("group_comparison/anova.html", error="Ошибка загрузки файла.")
        else:
            # Обрабатываем введенные данные через форму
            i = 1
            while f"group{i}" in request.form:
                group_value = request.form[f"group{i}"]
                loaded_groups.append(group_value)
                i += 1

        # Выполняем ANOVA для собранных данных
        if len(loaded_groups) >= 2:
            data = []
            labels = []
            for idx, group_str in enumerate(loaded_groups):
                group_data = parse_input_data(group_str)
                data.extend(group_data)
                labels.extend([f"{idx+1} ряд"] * len(group_data))
            
            if len(set(labels)) >= 2 and len(data) == len(labels):
                df_anova = pd.DataFrame({"Group": labels, "Value": data})
                unique_groups = sorted(set(labels))
                
                if len(unique_groups) >= 2:
                    group_data_list = [df_anova[df_anova["Group"] == g]["Value"].dropna() for g in unique_groups]
                    if all(len(g_data) > 0 for g_data in group_data_list):
                        stat, p = stats.f_oneway(*group_data_list)
                        anova_result = {"statistic": stat, "pvalue": p}
                        conclusion = (
                            "найдены значимые различия между рядами (отвергаем H0)"
                            if p < 0.05
                            else "значимых различий между рядами не найдено (не отвергаем H0)"
                        )
                        plots = create_anova_plots(df_anova, unique_groups)

    # Возвращаем данные в шаблон
    return render_template(
        "group_comparison/anova.html",
        active_page="anova",
        initial_groups=loaded_groups if loaded_groups else initial_groups,
        anova_result=anova_result,
        anova_conclusion=conclusion,
        plots=plots,
    )

@app.route("/tukey", methods=["GET", "POST"])
def tukey():
    initial_groups = ["1, 2, 3", "4, 5, 6", "7, 8, 9"]
    result = None
    plots = None
    conclusion = ""
    groups_data = []

    if request.method == "POST":
        # Проверяем, загружается ли файл с данными
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data_dict, error = extract_data_from_dataframe(df, df.columns.tolist())
                if data_dict is not None:
                    for col in data_dict:
                        if data_dict[col]:
                            group_values = ",".join(map(str, data_dict[col]))
                            groups_data.append(group_values)
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    return render_template("group_comparison/tukey.html", error=error)
            else:
                # Обработка ошибки process_uploaded_file
                return render_template("group_comparison/tukey.html", error="Ошибка загрузки файла.")
        else:
            # Обрабатываем введенные данные через форму
            i = 1
            while f"group{i}" in request.form:
                group_value = request.form[f"group{i}"]
                groups_data.append(group_value)  # Сохраняем введенные данные как строки
                i += 1

        # Передаем собранные данные в шаблон как initial_groups
        initial_groups = groups_data

        # Остальная логика обработки (Tukey HSD, графики) использует groups_data
        if len(groups_data) >= 2:  # Используем groups_data
            data = []
            labels = []
            for idx, group_str in enumerate(groups_data):  # Используем groups_data
                group_data = parse_input_data(group_str)
                data.extend(group_data)
                labels.extend([f"Ряд {idx+1}"] * len(group_data))

            if len(set(labels)) >= 2 and len(data) == len(labels):
                df = pd.DataFrame({"Group": labels, "Value": data})
                unique_groups = sorted(set(labels))
                if len(unique_groups) >= 2:
                    group_data_list = [df[df["Group"] == g]["Value"].dropna() for g in unique_groups]
                    if all(len(g_data) > 0 for g_data in group_data_list):
                        # Выполнение Tukey HSD
                        mc = pairwise_tukeyhsd(endog=df["Value"], groups=df["Group"], alpha=0.05)

                        # Переименовываем столбцы в результатах Tukey HSD
                        result_table = pd.DataFrame(data=mc.summary().data[1:], columns=mc.summary().data[0])
                        result_table = result_table.rename(columns={
                            "group1": "Первая выборка",
                            "group2": "Вторая выборка",
                            "meandiff": "Разница средних значений",
                            "p-adj": "P-значение",
                            "lower": "Нижняя граница",
                            "upper": "Верхняя граница",
                            "reject": "Отвергаем H0"
                        })

                        result = result_table.to_html(index=False)  # Преобразуем в HTML

                        # Общий вывод на основе результатов Тьюки
                        significant_differences = any(row[3] < 0.05 for row in mc.summary().data[1:])

                        if significant_differences:
                            conclusion = "средние значения чрезвычайно различаются (отвергаем H0).  "
                            conclusion += "Обнаружены значимые различия\n"
                            for row in mc.summary().data[1:]:
                                if row[3] < 0.05:
                                    group1_num = row[0].split()[1]
                                    group2_num = row[1].split()[1]
                                    p_adj = row[3]
                                    meandiff = row[2]
                                    conclusion += f" рядом {group1_num} и рядом {group2_num} (p-value={p_adj:.3f}, разница средних значений={meandiff:.2f})\n"
                        else:
                            conclusion = "значимой разницы в средних значениях нет (не отвергаем H0)."
                        plots = create_tukey_plot(df, unique_groups)

    # Передаем (возможно, обновленный) initial_groups в шаблон
    return render_template(
        "group_comparison/tukey.html",
        active_page="tukey",
        initial_groups=initial_groups,  # Передаем обновленный список
        tukey_result=result,
        tukey_conclusion=conclusion,
        plots=plots,
    )

@app.route("/ttest", methods=["GET", "POST"])
def ttest():
    group1 = "1, 2, 3, 4, 5"
    group2 = "2, 3, 4, 5, 6"
    equal_var = True
    result = None
    plots = None
    conclusion = ""
    if request.method == "POST":
        equal_var = "equal_var" in request.form
        
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df)
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        data1 = data[columns[0]]
                        data2 = data[columns[1]]
                        group1 = ', '.join(map(str, data1[:20]))  # Заполняем поля ввода первыми 5 значениями
                        group2 = ', '.join(map(str, data2[:20]))
                        stat, p = stats.ttest_ind(data1, data2, equal_var=equal_var)
                        result = {"statistic": stat, "pvalue": p}
                        conclusion = (
                            "средние значения значительно различаются (отвергаем H0)"
                            if p < 0.05
                            else "значимой разницы в средних значениях нет (не отвергаем H0)"
                        )
                        plots = create_ttest_plots(data1, data2)
                    else:
                        error = "Недостаточно колонок для сравнения."
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            group1 = request.form.get("group1", group1)
            group2 = request.form.get("group2", group2)
            data1 = parse_input_data(group1)
            data2 = parse_input_data(group2)
            if len(data1) > 1 and len(data2) > 1:
                stat, p = stats.ttest_ind(data1, data2, equal_var=equal_var)
                result = {"statistic": stat, "pvalue": p}
                conclusion = (
                    "средние значения значительно различаются (отвергаем H0)"
                    if p < 0.05
                    else "значимой разницы в средних значениях нет (не отвергаем H0)"
                )
                plots = create_ttest_plots(data1, data2)
    return render_template(
        "group_comparison/ttest.html",
        active_page="ttest",
        group1=group1,
        group2=group2,
        equal_var=equal_var,
        result=result,
        conclusion=conclusion,
        plots=plots,
    )


@app.route("/mannwhitney", methods=["GET", "POST"])
def mannwhitney():
    group1 = "1, 2, 3, 4, 5"
    group2 = "6, 7, 8, 9, 10"
    use_continuity = True
    alternative = "two-sided"
    result = None
    plots = None
    conclusion = ""
    file_error = None

    if request.method == "POST":
        # Обработка ручного ввода данных
        group1 = request.form.get("group1", group1)
        group2 = request.form.get("group2", group2)
        use_continuity = "use_continuity" in request.form
        alternative = request.form.get("alternative", alternative)

        # Обработка загрузки файла
        if 'data_file' in request.files:
            file = request.files['data_file']
            if file.filename != '':
                df, file_error = process_uploaded_file(file)
                if df is not None:
                    data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                    if data:
                        # Используем первые две колонки для анализа
                        columns = list(data.keys())
                        if len(columns) >= 2:
                            group1 = ', '.join(map(str, data[columns[0]]))
                            group2 = ', '.join(map(str, data[columns[1]]))
                        else:
                            file_error = "Недостаточно колонок для анализа."
                    else:
                        file_error = error if error else "Ошибка при извлечении данных."
                else:
                    file_error = file_error if file_error else "Ошибка при обработке файла."

        # Анализ данных
        data1 = parse_input_data(group1)
        data2 = parse_input_data(group2)
        if len(data1) > 0 and len(data2) > 0:
            stat, p = stats.mannwhitneyu(data1, data2, use_continuity=use_continuity, alternative=alternative)
            result = {"statistic": stat, "pvalue": p}
            conclusion = (
                "распределения значительно различаются (отвергаем H0)"
                if p < 0.05
                else "значимой разницы в распределениях нет (не отвергаем H0)"
            )
            plots = create_mannwhitney_plots(data1, data2)

    return render_template(
        "group_comparison/mannwhitney.html",
        active_page="mannwhitney",
        group1=group1,
        group2=group2,
        use_continuity=use_continuity,
        alternative=alternative,
        result=result,
        conclusion=conclusion,
        plots=plots,
        file_error=file_error
    )

@app.route("/levene", methods=["GET", "POST"])
def levene():
    # Инициализируем initial_groups стандартными значениями по умолчанию
    initial_groups = ["1, 2, 3", "4, 5, 6", "7, 8, 9"]
    result = None
    plots = None
    conclusion = ""
    groups_data = []

    if request.method == "POST":
        # Проверяем, загружается ли файл с данными
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data_dict, error = extract_data_from_dataframe(df, df.columns.tolist())
                if data_dict is not None:
                    for col in data_dict:
                        if data_dict[col]:
                            group_values = ",".join(map(str, data_dict[col]))
                            groups_data.append(group_values)
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    return render_template("group_comparison/levene.html", error=error)
            else:
                # Обработка ошибки process_uploaded_file
                return render_template("group_comparison/levene.html", error="Ошибка загрузки файла.")
        else:
            # Обрабатываем введенные данные через форму
            i = 1
            while f"group{i}" in request.form:
                group_value = request.form[f"group{i}"]
                groups_data.append(group_value)  # Сохраняем введенные данные как строки
                i += 1

        # Передаем собранные данные в шаблон как initial_groups
        initial_groups = groups_data

        # Остальная логика обработки (Levene's test, графики) остается такой же,
        # но теперь использует groups_data, полученные выше
        if len(groups_data) >= 2:  # Используем groups_data
            data = []
            labels = []
            for idx, group_str in enumerate(groups_data):  # Используем groups_data
                group_data = parse_input_data(group_str)
                data.extend(group_data)
                labels.extend([f"Ряд {idx+1}"] * len(group_data))

            if len(set(labels)) >= 2 and len(data) == len(labels):
                df = pd.DataFrame({"Group": labels, "Value": data})
                unique_groups = sorted(set(labels))
                if len(unique_groups) >= 2:
                    group_data_list = [df[df["Group"] == g]["Value"].dropna() for g in unique_groups]
                    if all(len(g_data) > 1 for g_data in group_data_list):
                        stat, p = stats.levene(*group_data_list)
                        result = {"statistic": stat, "pvalue": p}
                        conclusion = (
                            "дисперсии значительно различаются (отвергаем H0)"
                            if p < 0.05
                            else "значимой разницы в дисперсиях нет (не отвергаем H0)"
                        )
                        plots = create_levene_plots(df, unique_groups)

    # Передаем (возможно, обновленный) initial_groups в шаблон
    return render_template(
        "group_comparison/levene.html",
        active_page="levene",
        initial_groups=initial_groups,  # Передаем обновленный список
        result=result,
        conclusion=conclusion,
        plots=plots,
    )

@app.route("/fisher", methods=["GET", "POST"])
def fisher():
    a_default, b_default, c_default, d_default = "10", "5", "3", "12"
    alpha_default = "0.05"  # Добавляем уровень значимости по умолчанию
    result = None
    plots = None
    conclusion = ""

    if request.method == "POST":
        try:
            a = int(request.form.get("a", a_default))
            b = int(request.form.get("b", b_default))
            c = int(request.form.get("c", c_default))
            d = int(request.form.get("d", d_default))
            alpha = float(request.form.get("alpha", alpha_default))  # Получаем уровень значимости

            if any(x < 0 for x in [a, b, c, d]):
                raise ValueError("Значения в таблице сопряженности должны быть неотрицательными.")

            table = np.array([[a, b], [c, d]])
            odds_ratio, p_value = stats.fisher_exact(table)
            result = {"odds_ratio": odds_ratio, "pvalue": p_value}
            conclusion = (
                "ассоциация значима (отвергаем H0)"
                if p_value < alpha
                else "значимой ассоциации нет (не отвергаем H0)"
            )
            try:
                plots = create_fisher_plot(a, b, c, d)
            except Exception as e:
                msg = f"Ошибка при генерации графика: {e}"
                app.logger.exception(msg) # Записываем ошибку в лог
                plots = {"barplot": f"<div>{msg}</div>"}


        except (ValueError, TypeError) as e:
            msg = f"Неверный ввод: {e}. Пожалуйста, введите целые числа."
            result = {"error": msg}
            conclusion = "Ошибка в данных ввода."
            app.logger.warning(msg)  # Записываем предупреждение в лог
            plots = {"barplot": "<div>Ошибка при генерации графика: Неверный ввод</div>"}

    return render_template(
        "group_comparison/fisher.html",
        active_page="fisher",
        a=a_default,
        b=b_default,
        c=c_default,
        d=d_default,
        alpha = alpha_default, # Передаем значение alpha в шаблон
        result=result,
        conclusion=conclusion,
        plots=plots,
    )

# --- Корреляция (Correlation) ---

@app.route("/pearson", methods=["GET", "POST"])
def pearson():
    return correlation_analysis("pearson", "Пирсон")

@app.route("/spearman", methods=["GET", "POST"])
def spearman():
    return correlation_analysis("spearman", "Спирмен")

@app.route("/kendall", methods=["GET", "POST"])
def kendall():
    return correlation_analysis("kendall", "Кендалл")

def correlation_analysis(method, title):
    var1 = "1, 2, 3, 4, 5"
    var2 = "2, 4, 6, 8, 10"
    result = None
    plots = None
    conclusion = ""
    title = "Корреляционный анализ"

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        data1 = data[columns[0]]
                        data2 = data[columns[1]]
                        var1 = ', '.join(map(str, data1[:20]))  
                        var2 = ', '.join(map(str, data2[:20]))
                        x = data1
                        y = data2
                        if method == "pearson":
                            stat, p = stats.pearsonr(x, y)
                        elif method == "spearman":
                            stat, p = stats.spearmanr(x, y)
                        elif method == "kendall":
                            stat, p = stats.kendalltau(x, y)
                        result = {"statistic": stat, "pvalue": p}
                        conclusion = (
                            f"найдена значимая корреляция (p < 0.05, r={stat:.4f})"
                            if p < 0.05
                            else f"значимой корреляции нет (p ≥ 0.05, r={stat:.4f})"
                        )
                        plots = create_correlation_plots(np.array(x), np.array(y), "Первый ряд", "Второй ряд")
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            var1 = request.form.get("var1", var1)
            var2 = request.form.get("var2", var2)
            x = parse_input_data(var1)
            y = parse_input_data(var2)
            if len(x) > 1 and len(y) > 1 and len(x) == len(y):
                if method == "pearson":
                    stat, p = stats.pearsonr(x, y)
                elif method == "spearman":
                    stat, p = stats.spearmanr(x, y)
                elif method == "kendall":
                    stat, p = stats.kendalltau(x, y)
                result = {"statistic": stat, "pvalue": p}
                conclusion = (
                    f"найдена значимая корреляция (p < 0.05, r={stat:.4f})"
                    if p < 0.05
                    else f"значимой корреляции нет (p ≥ 0.05, r={stat:.4f})"
                )
                plots = create_correlation_plots(np.array(x), np.array(y), "Первый ряд", "Второй ряд")

    return render_template(
        f"correlation/{method}.html",
        title=title,
        active_page=method.lower(),
        var1=var1,
        var2=var2,
        result=result,
        conclusion=conclusion,
        plots=plots,
    )

# --- Регрессия (Regression) ---

@app.route("/linear_regression", methods=["GET", "POST"])
def linear_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        model = LinearRegression()
                        model.fit(x, y)
                        y_pred = model.predict(x)
                        mse = mean_squared_error(y, y_pred)
                        r2 = r2_score(y, y_pred)
                        result = {"mse": mse, "r2": r2, "intercept": model.intercept_, "slope": model.coef_[0]}
                        x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                        y_range = model.predict(x_range)
                        plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, "Линейная регрессия")}
                        if result:
                            conclusion = ""
                            if result["slope"] > 0:
                                conclusion += "При увеличении независимой переменной целевая переменная в среднем увеличивается. "
                            elif result["slope"] < 0:
                                conclusion += "При увеличении независимой переменной целевая переменная в среднем уменьшается. "
                            else:
                                conclusion += "Целевая переменная не зависит от независимой переменной. "
                            
                            conclusion += f"При изменении независимой переменной на единицу целевая переменная изменяется в среднем на {result['slope']:.2f} единиц. "
                            conclusion += f"Свободный член модели составляет {result['intercept']:.2f}, что представляет собой прогнозируемое значение целевой переменной, когда независимая переменная равна нулю. "
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            if len(x) == len(y) and len(x) > 1:
                model = LinearRegression()
                model.fit(x, y)
                y_pred = model.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                result = {"mse": mse, "r2": r2, "intercept": model.intercept_, "slope": model.coef_[0]}
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, "Линейная регрессия")}
                if result:
                    conclusion = ""
                    if result["slope"] > 0:
                        conclusion += "При увеличении независимой переменной целевая переменная в среднем увеличивается. "
                    elif result["slope"] < 0:
                        conclusion += "При увеличении независимой переменной целевая переменная в среднем уменьшается. "
                    else:
                        conclusion += "Целевая переменная не зависит от независимой переменной. "
                    
                    conclusion += f"При изменении независимой переменной на единицу целевая переменная изменяется в среднем на {result['slope']:.2f} единиц. "
                    conclusion += f"Свободный член модели составляет {result['intercept']:.2f}, что представляет собой прогнозируемое значение целевой переменной, когда независимая переменная равна нулю. "

    return render_template(
        "regression/linear.html",
        active_page="linear_regression",
        title="Линейная регрессия",
        regression_type="linear",
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots,
        conclusion=conclusion
    )


@app.route("/polynomial_regression", methods=["GET", "POST"])
def polynomial_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "1, 4, 9, 16, 25"
    degree = "2"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        degree = request.form.get("degree", degree)
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        degree = int(degree)
                        if len(x) == len(y) and len(x) > 1:
                            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                            poly = PolynomialFeatures(degree=degree)
                            X_train_poly = poly.fit_transform(X_train)
                            X_test_poly = poly.transform(X_test)
                            model = LinearRegression()
                            model.fit(X_train_poly, y_train)
                            y_train_pred = model.predict(X_train_poly)
                            y_test_pred = model.predict(X_test_poly)
                            train_mse = mean_squared_error(y_train, y_train_pred)
                            test_mse = mean_squared_error(y_test, y_test_pred)
                            train_r2 = r2_score(y_train, y_train_pred)
                            test_r2 = r2_score(y_test, y_test_pred)
                            result = {
                                "train_mse": train_mse,
                                "test_mse": test_mse,
                                "train_r2": train_r2,
                                "test_r2": test_r2,
                                "degree": degree,
                            }
                            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                            x_range_poly = poly.transform(x_range)
                            y_range = model.predict(x_range_poly)
                            plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"Полиномиальная регрессия (степень {degree})")}
                            if result:
                                conclusion = f"полиномиальная регрессия описывает зависимость между переменными. "
                                if degree == 1:
                                    conclusion += "Линейная зависимость указывает на равномерное изменение целевой переменной при изменении независимой переменной. "
                                elif degree == 2:
                                    conclusion += "Квадратичная зависимость указывает на наличие экстремума в целевой переменной. "
                                else:
                                    conclusion += f"Полином описывает сложную нелинейную зависимость между переменными. "
                                
                                if result["test_r2"] > 0:
                                    conclusion += "Модель показывает положительную зависимость между переменными. "
                                else:
                                    conclusion += "Модель показывает отрицательную зависимость между переменными. "
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            degree = request.form.get("degree", degree)
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            degree = int(degree)
            if len(x) == len(y) and len(x) > 1:
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train)
                X_test_poly = poly.transform(X_test)
                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_train_pred = model.predict(X_train_poly)
                y_test_pred = model.predict(X_test_poly)
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                train_r2 = r2_score(y_train, y_train_pred)
                test_r2 = r2_score(y_test, y_test_pred)
                result = {
                    "train_mse": train_mse,
                    "test_mse": test_mse,
                    "train_r2": train_r2,
                    "test_r2": test_r2,
                    "degree": degree,
                }
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                x_range_poly = poly.transform(x_range)
                y_range = model.predict(x_range_poly)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"Полиномиальная регрессия (степень {degree})")}
                if result:
                    conclusion = f"полиномиальная регрессия описывает зависимость между переменными. "
                    if degree == 1:
                        conclusion += "Линейная зависимость указывает на равномерное изменение целевой переменной при изменении независимой переменной. "
                    elif degree == 2:
                        conclusion += "Квадратичная зависимость указывает на наличие экстремума в целевой переменной. "
                    else:
                        conclusion += f"Полином описывает сложную нелинейную зависимость между переменными. "
                    
                    if result["test_r2"] > 0:
                        conclusion += "Модель показывает положительную зависимость между переменными. "
                    else:
                        conclusion += "Модель показывает отрицательную зависимость между переменными. "

    return render_template(
        "regression/polynomial.html",
        active_page="polynomial_regression",
        title="Полиномиальная регрессия",
        regression_type="polynomial",
        x_values=x_values,
        y_values=y_values,
        degree=degree,
        result=result,
        plots=plots,
        conclusion=conclusion
    )


def prepare_data_from_dataframe(df):
    """Подготовка данных для логистической регрессии из DataFrame"""
    try:
        # Извлекаем данные из первой и второй колонки
        x_column = df.iloc[:, 0]
        y_column = df.iloc[:, 1]

        # Формируем x_values как строку с запятыми
        x_values = ', '.join(map(str, x_column))

        # Проверяем уникальные значения во второй колонке
        unique_y = y_column.unique()

        if len(unique_y) > 2:
            return None, "Во второй колонке должно быть не более двух уникальных значений."
        
        # Если уникальные значения числовые
        if np.issubdtype(y_column.dtype, np.number):
            y_values = ', '.join(map(str, unique_y))
        else:
            # Если уникальные значения не числовые, присваиваем 0 и 1
            if len(unique_y) <= 2:
                mapping = {val: idx for idx, val in enumerate(unique_y)}
                y_values = ', '.join(map(str, [mapping[val] for val in y_column]))
            else:
                return None, "Во второй колонке обнаружено больше двух уникальных слов или букв."

        return x_values, y_values

    except Exception as e:
        return None, f"Ошибка при подготовке данных: {str(e)}"

@app.route("/logistic_regression", methods=["GET", "POST"])
def logistic_regression():
    result = None
    plots = None
    conclusion = None
    x_values = ""  # Инициализация переменной
    y_values = ""  # Инициализация переменной

    if request.method == "POST":
        # Получаем данные из формы
        x_values = request.form.get("x_values", "")
        y_values = request.form.get("y_values", "")
        uploaded_file = request.files.get("data_file")

        if uploaded_file:
            df, error = process_uploaded_file(uploaded_file)
            if error:
                return render_template(
                    "regression/logistic.html",
                    active_page="logistic_regression",
                    title="Логистическая регрессия",
                    regression_type="logistic",
                    x_values=x_values,
                    y_values=y_values,
                    result=result,
                    plots=plots,
                    conclusion=conclusion,
                    error=error
                )

            # Подготовка данных из загруженного файла
            x_values, y_values = prepare_data_from_dataframe(df)
            if not x_values or not y_values:
                return render_template(
                    "regression/logistic.html",
                    active_page="logistic_regression",
                    title="Логистическая регрессия",
                    regression_type="logistic",
                    x_values=x_values,
                    y_values=y_values,
                    result=result,
                    plots=plots,
                    conclusion=conclusion,
                    error="Ошибка при подготовке данных."
                )
        else:
            # Если файл не загружен, используем вводимые вручную данные
            if x_values and y_values:
                # Преобразуем вводимые данные в массивы
                x_values = x_values.split(",")
                y_values = y_values.split(",")

                # Проверяем, что y_values содержит только 0 и 1
                y_values_set = set(map(str.strip, y_values))
                if not y_values_set.issubset({"0", "1"}):
                    return render_template(
                        "regression/logistic.html",
                        active_page="logistic_regression",
                        title="Логистическая регрессия",
                        regression_type="logistic",
                        x_values=x_values,
                        y_values=y_values,
                        result=result,
                        plots=plots,
                        conclusion=conclusion,
                        error="Во второй колонке должны быть только значения 0 или 1."
                    )

                # Преобразуем x_values в массив
                x = np.array([float(val.strip()) for val in x_values]).reshape(-1, 1)
                y = np.array([int(val.strip()) for val in y_values])

                # Проверяем условия для обучения модели
                if len(x) == len(y) and len(x) > 1:
                    model = LogisticRegression()
                    model.fit(x, y)
                    y_pred = model.predict(x)
                    accuracy = accuracy_score(y, y_pred)
                    result = {"accuracy": accuracy, "intercept": model.intercept_[0], "coef": model.coef_[0][0]}
                    plots = {"regression_plot": create_logistic_plot(x, y, model)}
                    
                    if accuracy < 0.5:
                        conclusion = "модель имеет низкую точность."
                    elif 0.5 <= accuracy < 0.75:
                        conclusion = "модель имеет среднюю точность."
                    else:
                        conclusion = "модель имеет высокую точность."

                    # Дополнительные выводы на основе коэффициента
                    if model.coef_[0][0] > 0:
                        conclusion += " Положительный коэффициент указывает на возрастание вероятности положительного исхода."
                    else:
                        conclusion += " Отрицательный коэффициент указывает на возрастание вероятности отрицательного исхода."
                else:
                    return render_template(
                        "regression/logistic.html",
                        active_page="logistic_regression",
                        title="Логистическая регрессия",
                        regression_type="logistic",
                        x_values=x_values,
                        y_values=y_values,
                        result=result,
                        plots=plots,
                        conclusion=conclusion,
                        error="Количество значений x и y должно совпадать и быть больше 1."
                    )

    return render_template(
        "regression/logistic.html",
        active_page="logistic_regression",
        title="Логистическая регрессия",
        regression_type="logistic",
        x_values=x_values,
        y_values=y_values,
        result=result,
        plots=plots,
        conclusion=conclusion
    )


@app.route("/ridge_regression", methods=["GET", "POST"])
def ridge_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    alpha = "1.0"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        alpha = request.form.get("alpha", alpha)
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        alpha = float(alpha)
                        if len(x) == len(y) and len(x) > 1:
                            model = Ridge(alpha=alpha)
                            model.fit(x, y)
                            y_pred = model.predict(x)
                            mse = mean_squared_error(y, y_pred)
                            r2 = r2_score(y, y_pred)
                            result = {
                                "mse": mse,
                                "r2": r2,
                                "intercept": model.intercept_,
                                "slope": model.coef_[0],
                                "alpha": alpha
                            }
                            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                            y_range = model.predict(x_range)
                            plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"α = {alpha} гребневой регрессии")}
                            if r2 > 0.7:
                                conclusion = f"модель хорошо описывает данные (R^2 = {r2:.2f}). Наблюдается сильная линейная зависимость между X и Y."
                            elif r2 > 0.5:
                                conclusion = f"модель умеренно описывает данные (R^2 = {r2:.2f}). Линейная зависимость между X и Y прослеживается, но есть значительная дисперсия."
                            else:
                                conclusion = f"модель плохо описывает данные (R^2 = {r2:.2f}). Линейная зависимость между X и Y слабая или отсутствует. "

                            # Интерпретация коэффициентов
                            conclusion += f" С увеличением X на единицу, Y изменяется примерно на {result['slope']:.2f}."
                            conclusion += f" При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}."

                            # Интерпретация влияния alpha
                            if alpha > 10:  
                                conclusion += " Значение alpha велико, что указывает на сильную регуляризацию. Это помогает избежать переобучения, особенно при наличии мультиколлинеарности признаков."
                            else:
                                conclusion += " Значение alpha невелико, регуляризация умеренная."
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            alpha = float(request.form.get("alpha", alpha))
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            if len(x) == len(y) and len(x) > 1:
                model = Ridge(alpha=alpha)
                model.fit(x, y)
                y_pred = model.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                result = {
                    "mse": mse,
                    "r2": r2,
                    "intercept": model.intercept_,
                    "slope": model.coef_[0],
                    "alpha": alpha
                }
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"α = {alpha} гребневой регрессии")}
                if r2 > 0.7:
                    conclusion = f"модель хорошо описывает данные (R^2 = {r2:.2f}). Наблюдается сильная линейная зависимость между X и Y."
                elif r2 > 0.5:
                    conclusion = f"модель умеренно описывает данные (R^2 = {r2:.2f}). Линейная зависимость между X и Y прослеживается, но есть значительная дисперсия."
                else:
                    conclusion = f"модель плохо описывает данные (R^2 = {r2:.2f}). Линейная зависимость между X и Y слабая или отсутствует. "

                # Интерпретация коэффициентов
                conclusion += f" С увеличением X на единицу, Y изменяется примерно на {result['slope']:.2f}."
                conclusion += f" При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}."

                # Интерпретация влияния alpha
                if alpha > 10:  
                    conclusion += " Значение alpha велико, что указывает на сильную регуляризацию. Это помогает избежать переобучения, особенно при наличии мультиколлинеарности признаков."
                else:
                    conclusion += " Значение alpha невелико, регуляризация умеренная."

    return render_template(
        "regression/ridge.html",
        active_page="ridge_regression",
        title="Ридж-регрессия",
        regression_type="ridge",
        x_values=x_values,
        y_values=y_values,
        alpha=alpha,
        result=result,
        plots=plots,
        conclusion=conclusion
    )


@app.route("/lasso_regression", methods=["GET", "POST"])
def lasso_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    alpha = "1.0"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        alpha = request.form.get("alpha", alpha)
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        alpha = float(alpha)
                        if len(x) == len(y) and len(x) > 1:
                            model = Lasso(alpha=alpha)
                            model.fit(x, y)
                            y_pred = model.predict(x)
                            mse = mean_squared_error(y, y_pred)
                            r2 = r2_score(y, y_pred)
                            result = {
                                "mse": mse,
                                "r2": r2,
                                "intercept": model.intercept_,
                                "slope": model.coef_[0],
                                "alpha": alpha
                            }
                            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                            y_range = model.predict(x_range)
                            plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"α = {alpha} регрессии Лассо")}
                            if r2 > 0.7:
                                conclusion = f"модель хорошо описывает данные (R^2 = {r2:.2f}). "
                            elif r2 > 0.5:
                                conclusion = f"модель умеренно описывает данные (R^2 = {r2:.2f})."
                            else:
                                conclusion = f"модель плохо описывает данные (R^2 = {r2:.2f}).  Линейная зависимость между X и Y слабая или отсутствует. "

                            # Интерпретация коэффициентов
                            if abs(result["slope"]) < 0.01:  
                                conclusion += " Коэффициент наклона очень мал, что указывает на слабую зависимость Y от X."
                            else:
                                conclusion += f" С увеличением X на единицу, Y изменяется примерно на {result['slope']:.2f}."

                            # Пример предсказания
                            conclusion += f" При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}."

                            # Интерпретация влияния alpha
                            if alpha > 0.5:  
                                 conclusion += " Значение alpha велико, что указывает на сильную регуляризацию. Это может приводить к упрощению модели и занулению некоторых коэффициентов."
                            else:
                                 conclusion += " Значение alpha невелико, регуляризация умеренная."
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            alpha = float(request.form.get("alpha", alpha))
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            if len(x) == len(y) and len(x) > 1:
                model = Lasso(alpha=alpha)
                model.fit(x, y)
                y_pred = model.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                result = {
                    "mse": mse,
                    "r2": r2,
                    "intercept": model.intercept_,
                    "slope": model.coef_[0],
                    "alpha": alpha
                }
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"α = {alpha} регрессии Лассо")}
                if r2 > 0.7:
                    conclusion = f"модель хорошо описывает данные (R^2 = {r2:.2f}). "
                elif r2 > 0.5:
                    conclusion = f"модель умеренно описывает данные (R^2 = {r2:.2f})."
                else:
                    conclusion = f"модель плохо описывает данные (R^2 = {r2:.2f}).  Линейная зависимость между X и Y слабая или отсутствует. "

                # Интерпретация коэффициентов
                if abs(result["slope"]) < 0.01:  
                    conclusion += " Коэффициент наклона очень мал, что указывает на слабую зависимость Y от X."
                else:
                    conclusion += f" С увеличением X на единицу, Y изменяется примерно на {result['slope']:.2f}."

                # Пример предсказания
                conclusion += f" При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}."

                # Интерпретация влияния alpha
                if alpha > 0.5:  
                     conclusion += " Значение alpha велико, что указывает на сильную регуляризацию. Это может приводить к упрощению модели и занулению некоторых коэффициентов."
                else:
                     conclusion += " Значение alpha невелико, регуляризация умеренная."

    return render_template(
        "regression/lasso.html",
        active_page="lasso_regression",
        title="Lasso-регрессия",
        regression_type="lasso",
        x_values=x_values,
        y_values=y_values,
        alpha=alpha,
        result=result,
        plots=plots,
        conclusion=conclusion
    )

@app.route("/svr_regression", methods=["GET", "POST"])
def svr_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    z = ""  
    kernel = "rbf"
    result = None
    plots = None
    conclusion = None
    prediction = None  

    kernel_translations = {
        'rbf': 'Ядро радиально-базисных функций',
        'linear': 'Линейное ядро',
        'poly': 'Полиномиальное ядро',
        'sigmoid': 'Сигмоидное ядро',
    }

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        model = SVR(kernel=kernel)
                        model.fit(x, y)
                        y_pred = model.predict(x)
                        mse = mean_squared_error(y, y_pred)
                        r2 = r2_score(y, y_pred)

                        translated_kernel = kernel_translations.get(kernel, 'Неизвестное ядро')

                        result = {
                            "mse": mse,
                            "r2": r2,
                            "kernel": translated_kernel,
                            "cv_score": np.mean(cross_val_score(model, x, y, cv=3))
                        }
                        x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                        y_range = model.predict(x_range)
                        plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"SVR ({translated_kernel})")}
                        conclusion = ""
                        if r2 > 0.7:
                            conclusion += f"модель SVR показала высокое качество подгонки под данные (R² = {r2:.2f}). "
                        elif r2 > 0.4:
                            conclusion += f"модель SVR показала среднее качество подгонки под данные (R² = {r2:.2f}). "
                        else:
                            conclusion += f"модель SVR показала низкое качество подгонки под данные (R² = {r2:.2f}). "

                        try:
                            z_value = float(z)
                            prediction = model.predict([[z_value]])
                            conclusion += f" Предсказанное значение целевой переменной для z = {z}: {prediction[0]:.2f}."
                        except ValueError:
                            conclusion += " Не удалось выполнить предсказание. Пожалуйста, введите корректное значение для z."
                    else:
                        # Обработка ошибки: недостаточно колонок
                        conclusion = "Ошибка: недостаточно колонок в загруженном файле."
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    conclusion = "Ошибка: не удалось извлечь данные из загруженного файла."
            else:
                # Обработка ошибки process_uploaded_file
                conclusion = "Ошибка: не удалось обработать загруженный файл."
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            z = request.form.get("z", z)  
            kernel = request.form.get("kernel", kernel)
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            if len(x) == len(y) and len(x) > 1:
                model = SVR(kernel=kernel)
                model.fit(x, y)
                y_pred = model.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                translated_kernel = kernel_translations.get(kernel, 'Неизвестное ядро')

                result = {
                    "mse": mse,
                    "r2": r2,
                    "kernel": translated_kernel,
                    "cv_score": np.mean(cross_val_score(model, x, y, cv=3))
                }
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"SVR ({translated_kernel})")}
                conclusion = ""
                if r2 > 0.7:
                    conclusion += f"модель SVR показала высокое качество подгонки под данные (R² = {r2:.2f}). "
                elif r2 > 0.4:
                    conclusion += f"модель SVR показала среднее качество подгонки под данные (R² = {r2:.2f}). "
                else:
                    conclusion += f"модель SVR показала низкое качество подгонки под данные (R² = {r2:.2f}). "

                try:
                    z_value = float(z)
                    prediction = model.predict([[z_value]])
                    conclusion += f" Предсказанное значение целевой переменной для z = {z}: {prediction[0]:.2f}."
                except ValueError:
                    conclusion += " Не удалось выполнить предсказание. Пожалуйста, введите корректное значение для z."

    return render_template(
        "regression/svr.html",
        active_page="svr_regression",
        title="Регрессия опорных векторов",
        regression_type="svr",
        x_values=x_values,
        y_values=y_values,
        z=z,
        kernel=kernel,
        result=result,
        plots=plots,
        conclusion=conclusion
    )



@app.route("/random_forest_regression", methods=["GET", "POST"])
def random_forest_regression():
    x_values = "1, 2, 3, 4, 5"
    y_values = "2, 4, 5, 4, 5"
    n_estimators = "100"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, df.columns[:2].tolist())
                if data is not None:
                    columns = list(data.keys())
                    if len(columns) >= 2:
                        x = data[columns[0]]
                        y = data[columns[1]]
                        x_values = ', '.join(map(str, x[:20]))  
                        y_values = ', '.join(map(str, y[:20]))
                        n_estimators = int(request.form.get("n_estimators", n_estimators))
                        x = np.array(x).reshape(-1, 1)
                        y = np.array(y)
                        if len(x) == len(y) and len(x) > 1:
                            model = RandomForestRegressor(n_estimators=n_estimators)
                            model.fit(x, y)
                            y_pred = model.predict(x)
                            mse = mean_squared_error(y, y_pred)
                            r2 = r2_score(y, y_pred)
                            result = {
                                "mse": mse,
                                "r2": r2,
                                "n_estimators": n_estimators,
                                "feature_importance": model.feature_importances_[0]
                            }
                            x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                            y_range = model.predict(x_range)
                            plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"Случайный лес из {n_estimators}")}
                            if r2 > 0.7:
                                conclusion = "модель хорошо описывает данные. Наблюдается сильная зависимость между X и Y. " \
                                             f"При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}." # Пример предсказания
                            elif r2 > 0.5:
                                conclusion = "модель умеренно описывает данные. Зависимость между X и Y прослеживается, но есть значительная дисперсия. " \
                                             f"При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}." # Пример предсказания
                            else:
                                conclusion = "модель плохо описывает данные. Зависимость между X и Y слабая или отсутствует."

                            conclusion += f" Важность признака: {result['feature_importance']:.2f}. " # Добавляем важность признака к выводу
                    else:
                        # Обработка ошибки: недостаточно колонок
                        pass
                else:
                    # Обработка ошибки extract_data_from_dataframe
                    pass
            else:
                # Обработка ошибки process_uploaded_file
                pass
        else:
            x_values = request.form.get("x_values", x_values)
            y_values = request.form.get("y_values", y_values)
            n_estimators = int(request.form.get("n_estimators", n_estimators))
            x = np.array(parse_input_data(x_values)).reshape(-1, 1)
            y = np.array(parse_input_data(y_values))
            if len(x) == len(y) and len(x) > 1:
                model = RandomForestRegressor(n_estimators=n_estimators)
                model.fit(x, y)
                y_pred = model.predict(x)
                mse = mean_squared_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                result = {
                    "mse": mse,
                    "r2": r2,
                    "n_estimators": n_estimators,
                    "feature_importance": model.feature_importances_[0]
                }
                x_range = np.linspace(min(x), max(x), 100).reshape(-1, 1)
                y_range = model.predict(x_range)
                plots = {"regression_plot": create_regression_plot(x.flatten(), y, x_range.flatten(), y_range, f"Случайный лес из {n_estimators}")}
                if r2 > 0.7:
                    conclusion = "модель хорошо описывает данные. Наблюдается сильная зависимость между X и Y. " \
                                 f"При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}." # Пример предсказания
                elif r2 > 0.5:
                    conclusion = "модель умеренно описывает данные. Зависимость между X и Y прослеживается, но есть значительная дисперсия. " \
                                 f"При X={max(x)[0]} предсказывается Y={y_range[-1]:.2f}." # Пример предсказания
                else:
                    conclusion = "модель плохо описывает данные. Зависимость между X и Y слабая или отсутствует."

                conclusion += f" Важность признака: {result['feature_importance']:.2f}. " # Добавляем важность признака к выводу

    return render_template(
        "regression/random_forest.html",
        active_page="random_forest_regression",
        title="Регрессия случайного леса",
        regression_type="random_forest",
        x_values=x_values,
        y_values=y_values,
        n_estimators=n_estimators,
        result=result,
        plots=plots,
        conclusion=conclusion
    )


# --- Описательная статистика / Распределения (Descriptive Stats / Distributions) ---

@app.route("/shapiro_wilk", methods=["GET", "POST"])
def shapiro_wilk():
    data_input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10"
    result = None
    plots = None
    conclusion = ""

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, [df.columns[0]])
                if data is not None:
                    # Извлекаем данные из первой колонки
                    values = data[df.columns[0]]
                    data_input = ', '.join(map(str, values))  # Форматируем для отображения
                    if len(values) >= 3:  # Минимальное количество значений для теста
                        stat, p = stats.shapiro(values)
                        result = {"statistic": stat, "pvalue": p}
                        conclusion = (
                            "данные выглядят нормально распределенными (не отвергаем H0)"
                            if p > 0.05
                            else "данные не выглядят нормально распределенными (отвергаем H0)"
                        )
                        plots = create_shapiro_plots(values)  # Генерируем графики
                else:
                    # Обработка ошибки извлечения данных
                    pass
            else:
                # Обработка ошибки загрузки файла
                pass
        else:
            data_input = request.form.get("data", data_input)
            values = parse_input_data(data_input)
            if len(values) >= 3:  # Минимальное количество значений для теста
                stat, p = stats.shapiro(values)
                result = {"statistic": stat, "pvalue": p}
                conclusion = (
                    "данные выглядят нормально распределенными (не отвергаем H0)"
                    if p > 0.05
                    else "данные не выглядят нормально распределенными (отвергаем H0)"
                )
                plots = create_shapiro_plots(values)

    return render_template(
        "descriptive_stats/shapiro_wilk.html",
        active_page="shapiro_wilk",
        data_input=data_input,
        result=result,
        conclusion=conclusion,
        plots=plots,
    )


@app.route("/iqr", methods=["GET", "POST"])
def iqr():
    data_input = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100"
    result = None
    plots = None
    conclusion = None

    if request.method == "POST":
        if 'data_file' in request.files and request.files['data_file'].filename != '':
            file = request.files['data_file']
            df, error = process_uploaded_file(file)
            if df is not None:
                data, error = extract_data_from_dataframe(df, [df.columns[0]])
                if data is not None:
                    values = data[df.columns[0]]
                    data_input = ', '.join(map(str, values))  # Форматируем для отображения
                    if len(values) > 0:
                        q75, q25 = np.percentile(values, [75, 25])
                        iqr = q75 - q25
                        lower_bound = q25 - 1.5 * iqr
                        upper_bound = q75 + 1.5 * iqr
                        outliers = [x for x in values if x < lower_bound or x > upper_bound]
                        median = np.median(values)
                        result = {
                            "q25": q25,
                            "q75": q75,
                            "iqr": iqr,
                            "lower_bound": lower_bound,
                            "upper_bound": upper_bound,
                            "outliers": outliers,
                            "median": median,
                        }
                        plots = create_iqr_plots(values)
                        outlier_note = (
                            f"обнаружены выбросы {outliers}." if outliers else "Выбросы не обнаружены согласно критерию IQR."
                        )
                        symmetry_note = get_symmetry_note(median, q25, q75)
                        conclusion = f"{outlier_note}  {symmetry_note}"
                else:
                    # Обработка ошибки извлечения данных
                    pass
            else:
                # Обработка ошибки загрузки файла
                pass
        else:
            data_input = request.form.get("data", data_input)
            values = parse_input_data(data_input)
            if len(values) > 0:
                q75, q25 = np.percentile(values, [75, 25])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr
                outliers = [x for x in values if x < lower_bound or x > upper_bound]
                median = np.median(values)
                result = {
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outliers": outliers,
                    "median": median,
                }
                plots = create_iqr_plots(values)
                outlier_note = (
                    f"обнаружены выбросы {outliers}." if outliers else "Выбросы не обнаружены согласно критерию IQR."
                )
                symmetry_note = get_symmetry_note(median, q25, q75)
                conclusion = f"{outlier_note}  {symmetry_note}"

    return render_template(
        "descriptive_stats/iqr.html",
        active_page="iqr",
        data_input=data_input,
        result=result,
        plots=plots,
        conclusion=conclusion,
    )

def get_symmetry_note(median, q25, q75):
    if median == q25 + (q75 - q25) / 2:
        return "Распределение данных выглядит симметричным, так как медиана находится примерно посередине между Q1 и Q3."
    elif median < q25 + (q75 - q25) / 2:
        return "Распределение данных имеет правостороннюю асимметрию (положительная скошенность), медиана смещена к Q1."
    else:
        return "Распределение данных имеет левостороннюю асимметрию (отрицательная скошенность), медиана смещена к Q3."


if __name__ == "__main__":
    app.run(debug=True)
