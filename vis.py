import plotly.express as px
from sklearn.metrics import auc
import plotly.graph_objects as go


def plot_auc_roc(fpr, tpr):
    fig = px.area(
        x=fpr, y=tpr,
        title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
        labels=dict(x='False Positive Rate', y='True Positive Rate'),
        width=700, height=500
    )
    fig.add_shape(
        type='line', line=dict(dash='dash'),
        x0=0, x1=1, y0=0, y1=1
    )

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain='domain')
    fig.show()


def create_histogram(dfm, title):
    fig = px.histogram(dfm, x="click")
    fig.update_layout(title=title)
    fig.show()


def trends_graph(trend):
    fig = go.Figure(go.Scatter(name=trend["name"],
                               x=trend["x"],
                               y=trend["y"],
                               hovertemplate=trend["hovers"],
                               marker_color=trend["marker_color"]
                               ))

    fig.update_layout(
        title=trend["title"],
        xaxis_tickformat=trend["xtick"],
        xaxis_title=trend["xtitle"],
        yaxis_title=trend["ytitle"]
    )

    fig.show()


def grouped_bars(grouped):
    fig = go.Figure(data=[
        go.Bar(name=grouped["bar1_name"], x=grouped["x"], y=grouped["y1"],
               hovertemplate=grouped["hover"], marker_color=grouped["color1"]),
        go.Bar(name=grouped["bar2_name"], x=grouped["x"], y=grouped["y2"],
               hovertemplate=grouped["hover"], marker_color=grouped["color2"])
    ])

    # Change the bar mode
    fig.update_layout(
        title=grouped["title"],
        xaxis_title=grouped["xtitle"],
        yaxis_title=grouped["ytitle"],
        barmode='group',
        xaxis_type=grouped["xaxis_type"]
    )
    fig.show()


def single_bars(df, single):
    fig = px.bar(df, x=single["x"], y=single["y"],
                 labels=single["labels"],
                 color=single["color"],
                 height=400)

    fig.update_layout(
        title=single["title"],
        xaxis_title=single["xtitle"],
        yaxis_title=single["ytitle"],
        xaxis_type=single["xaxis_type"]
    )
    fig.show()
