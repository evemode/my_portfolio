import joblib
import pandas as pd

from dash import Dash, html, dcc, Input, Output
import plotly.express as px


MODEL_PATH = "models/churn_rf_bundle.pkl"
bundle = joblib.load(MODEL_PATH)

model = bundle["model"] 
FEATURE_NAMES = bundle["feature_names"]
FEATURE_DTYPES = bundle.get("feature_dtypes", {})
THRESHOLD = float(bundle.get("threshold", 0.5))


def _set_onehot_group(X: pd.DataFrame, prefix: str, selected_value: str, baseline_value: str | None = None):
   
    cols = [c for c in X.columns if c.startswith(prefix + "_")]
    for c in cols:
        X.at[0, c] = 0

    if baseline_value is not None and selected_value == baseline_value:
        return

    target_col = f"{prefix}_{selected_value}"
    if target_col in X.columns:
        X.at[0, target_col] = 1


def _set_binary_yesno(X: pd.DataFrame, col_name: str, value_yes: str, value_no: str, selected_value: str):
   
    if col_name in X.columns:
        X.at[0, col_name] = 1 if selected_value == value_yes else 0
        return

    # Case B: one-hot
    _set_onehot_group(X, col_name, selected_value, baseline_value=None)


def build_processed_X(
    tenure: float | None,
    monthly: float | None,
    total: float | None,
    contract: str,
    internet_service: str,
    payment_method: str,
    paperless: str,
    online_security: str,
    tech_support: str,
    device_protection: str,
    streaming_tv: str,
    streaming_movies: str,
    dependents: str,
) -> pd.DataFrame:
    
    X = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES)

    if "tenure" in X.columns and tenure is not None:
        X.at[0, "tenure"] = float(tenure)

    if "MonthlyCharges" in X.columns and monthly is not None:
        X.at[0, "MonthlyCharges"] = float(monthly)

    if "TotalCharges" in X.columns and total is not None:
        X.at[0, "TotalCharges"] = float(total)

    _set_onehot_group(X, "Contract", contract, baseline_value="Month-to-month")

    _set_onehot_group(X, "InternetService", internet_service, baseline_value=None)

    _set_onehot_group(X, "PaymentMethod", payment_method, baseline_value=None)

    _set_binary_yesno(X, "PaperlessBilling", value_yes="Yes", value_no="No", selected_value=paperless)

    _set_binary_yesno(X, "Dependents", value_yes="Yes", value_no="No", selected_value=dependents)

    _set_onehot_group(X, "OnlineSecurity", online_security, baseline_value=None)

    _set_onehot_group(X, "TechSupport", tech_support, baseline_value=None)
    
    _set_onehot_group(X, "DeviceProtection", device_protection, baseline_value=None)
    
    _set_onehot_group(X, "StreamingTV", streaming_tv, baseline_value=None)
    
    _set_onehot_group(X, "StreamingMovies", streaming_movies, baseline_value=None)

    for col, dt in FEATURE_DTYPES.items():
        if col in X.columns:
            try:
                X[col] = X[col].astype(dt)
            except Exception:
                pass

    return X




app = Dash(__name__)
server = app.server

CONTRACT_OPTIONS = ["Month-to-month", "One year", "Two year"]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"] 
PAYMENT_OPTIONS = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
YES_NO = ["Yes", "No"]
YES_NO_NOINT = ["Yes", "No", "No internet service"]

def section_title(text: str):
    return html.H3(text, style={"marginTop": "20px", "marginBottom": "8px"})

app.layout = html.Div(
    style={"maxWidth": "980px", "margin": "40px auto", "fontFamily": "Arial"},
    children=[
        html.H1("Customer Churn Prediction"),

        section_title("Numeric"),
        html.Label("Tenure (months)"),
        dcc.Input(id="tenure", type="number", value=12),
        html.Br(),
        html.Label("Monthly Charges"),
        dcc.Input(id="monthly", type="number", value=70),
        html.Br(),
        html.Label("Total Charges"),
        dcc.Input(id="total", type="number", value=800),

        section_title("Contract & Billing"),
        html.Label("Contract"),
        dcc.RadioItems(
            id="contract",
            options=[{"label": x, "value": x} for x in CONTRACT_OPTIONS],
            value="Month-to-month",
            inline=True,
        ),
        html.Br(),
        html.Label("Paperless Billing"),
        dcc.RadioItems(
            id="paperless",
            options=[{"label": x, "value": x} for x in YES_NO],
            value="Yes",
            inline=True,
        ),
        html.Br(),
        html.Label("Payment Method"),
        dcc.Dropdown(
            id="payment_method",
            options=[{"label": x, "value": x} for x in PAYMENT_OPTIONS],
            value="Electronic check",
            clearable=False,
        ),

        section_title("Internet & Add-ons"),
        html.Label("Internet Service"),
        dcc.Dropdown(
            id="internet_service",
            options=[{"label": x, "value": x} for x in INTERNET_OPTIONS],
            value="Fiber optic",
            clearable=False,
        ),
        html.Br(),
        html.Label("Online Security"),
        dcc.Dropdown(
            id="online_security",
            options=[{"label": x, "value": x} for x in YES_NO_NOINT],
            value="No",
            clearable=False,
        ),
        html.Br(),
        html.Label("Tech Support"),
        dcc.Dropdown(
            id="tech_support",
            options=[{"label": x, "value": x} for x in YES_NO_NOINT],
            value="No",
            clearable=False,
        ),
        html.Br(),
        html.Label("Device Protection"),
        dcc.Dropdown(
            id="device_protection",
            options=[{"label": x, "value": x} for x in YES_NO_NOINT],
            value="No",
            clearable=False,
        ),

        section_title("Streaming"),
        html.Label("Streaming TV"),
        dcc.Dropdown(
            id="streaming_tv",
            options=[{"label": x, "value": x} for x in YES_NO_NOINT],
            value="No",
            clearable=False,
        ),
        html.Br(),
        html.Label("Streaming Movies"),
        dcc.Dropdown(
            id="streaming_movies",
            options=[{"label": x, "value": x} for x in YES_NO_NOINT],
            value="No",
            clearable=False,
        ),

        section_title("Household"),
        html.Label("Dependents"),
        dcc.RadioItems(
            id="dependents",
            options=[{"label": x, "value": x} for x in YES_NO],
            value="No",
            inline=True,
        ),

        html.Br(),
        html.Button("Predict churn", id="predict-btn"),

        html.H2(id="prediction-output", style={"marginTop": "24px"}),
        dcc.Graph(id="prob-gauge"),
        html.Div(id="debug-output", style={"whiteSpace": "pre-wrap", "fontSize": "12px", "opacity": 0.7}),
    ],
)

@app.callback(
    Output("prediction-output", "children"),
    Output("prob-gauge", "figure"),
    Output("debug-output", "children"),
    Input("predict-btn", "n_clicks"),
    Input("tenure", "value"),
    Input("monthly", "value"),
    Input("total", "value"),
    Input("contract", "value"),
    Input("internet_service", "value"),
    Input("payment_method", "value"),
    Input("paperless", "value"),
    Input("online_security", "value"),
    Input("tech_support", "value"),
    Input("device_protection", "value"),
    Input("streaming_tv", "value"),
    Input("streaming_movies", "value"),
    Input("dependents", "value"),
)
def predict(
    n_clicks,
    tenure,
    monthly,
    total,
    contract,
    internet_service,
    payment_method,
    paperless,
    online_security,
    tech_support,
    device_protection,
    streaming_tv,
    streaming_movies,
    dependents,
):
    if not n_clicks:
        empty_fig = px.bar(pd.DataFrame({"x": [0], "y": [0]}), x="x", y="y")
        empty_fig.update_layout(title="Churn probability", xaxis_visible=False, yaxis_visible=False)
        empty_fig.update_traces(showlegend=False)
        return "", empty_fig, ""

    X = build_processed_X(
        tenure=tenure,
        monthly=monthly,
        total=total,
        contract=contract,
        internet_service=internet_service,
        payment_method=payment_method,
        paperless=paperless,
        online_security=online_security,
        tech_support=tech_support,
        device_protection=device_protection,
        streaming_tv=streaming_tv,
        streaming_movies=streaming_movies,
        dependents=dependents,
    )

    proba = float(model.predict_proba(X)[0, 1])
    label = "HIGH RISK" if proba >= THRESHOLD else "LOW RISK"
    result = f"Churn probability: {proba:.2%} — {label} (threshold={THRESHOLD:.2f})"

    fig_df = pd.DataFrame({"metric": ["Churn probability", "Threshold"], "value": [proba, THRESHOLD]})
    fig = px.bar(fig_df, x="metric", y="value")
    fig.update_layout(title="Churn probability vs threshold", yaxis_tickformat=".0%", yaxis_range=[0, 1])

    non_zero = int((X.iloc[0] != 0).sum())
    dbg = (
        f"Input row columns: {X.shape[1]} (must match fit features)\n"
        f"Non-zero features: {non_zero}\n"
        f"Contract={contract}, InternetService={internet_service}, PaymentMethod={payment_method}, Paperless={paperless}\n"
        f"OnlineSecurity={online_security}, TechSupport={tech_support}, DeviceProtection={device_protection}\n"
        f"StreamingTV={streaming_tv}, StreamingMovies={streaming_movies}, Dependents={dependents}"
    )

    return result, fig, dbg


if __name__ == "__main__":
    app.run(debug=True)