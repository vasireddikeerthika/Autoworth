# AutoWorth: Used Car Price Predictor
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AutoWorth", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;600&family=Fraunces:wght@600&display=swap');
.stApp { background: #FBF6E6; color: #2B3310; font-family: 'Instrument Sans', sans-serif; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }
h1, h2, h3 { font-family: 'Fraunces', serif; color: #2B3310; }
#MainMenu, header, footer { visibility: hidden; }
.stButton > button { background: #D6452F; border: none; border-radius: 4px; padding: .65rem 1.4rem; }
.stButton > button p { color: #FBF6E6; font-weight: 600; }
.stButton > button:hover { background: #B93625; }
[data-testid="stMetric"] { background: #F2EBD0; border: 1px solid #D8D0B5; border-radius: 4px; padding: 1rem; }
.stTabs [aria-selected="true"] p { color: #D6452F; }
.result { background: #F2EBD0; border: 2px solid #D6452F; border-radius: 4px; padding: 1.4rem 1.6rem; margin-top: 1rem; }
.result h1 { margin: 0; }
</style>
""", unsafe_allow_html=True)

px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = ["#6F7F1E", "#D6452F"]


def show(fig):
    """Apply the cream theme to a chart and display it."""
    fig.update_layout(paper_bgcolor="#FBF6E6", plot_bgcolor="#FBF6E6", font_color="#2B3310")
    st.plotly_chart(fig, width="stretch")


# ---------- Load files saved by the notebook ----------
@st.cache_resource
def load():
    model = joblib.load("car_price_model.joblib")
    meta = json.load(open("meta.json"))
    metrics = pd.read_csv("model_metrics.csv")
    df = pd.read_csv("cleaned_cars.csv")
    df["price_lakh"] = df["selling_price"] / 1e5     # 1 lakh = 100,000 rupees
    return model, meta, metrics, df


model, meta, metrics, df = load()
best = metrics.sort_values("R2 (test)", ascending=False).iloc[0]

st.title("AutoWorth")
st.caption("Used car price estimation using machine learning. An academic data analytics project.")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Data Analysis", "Model Comparison", "Price Prediction", "Terms & Privacy"])

# ---------- Overview ----------
with tab1:
    st.header("Understanding the dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cars analysed", f"{len(df):,}")
    c2.metric("Brands", df["brand"].nunique())
    c3.metric("Median price", f"Rs {df['price_lakh'].median():.1f} L")
    c4.metric("Best R²", f"{best['R2 (test)']:.2f}")

    st.subheader("Project workflow")
    st.markdown("""
1. **Data preparation:** fix duplicates, inconsistent units and extreme values.
2. **Exploratory analysis:** use charts to see how vehicle features relate to price.
3. **Machine learning:** train Linear Regression and Random Forest and test both on unseen cars.
4. **Prediction:** the better model estimates the price of the car you enter.
""")
    st.dataframe(df.drop(columns=["price_lakh"]).head(15), hide_index=True, width="stretch")

# ---------- Data analysis ----------
with tab2:
    st.header("Data Analysis")
    top = df.groupby("brand")["price_lakh"].mean().sort_values().tail(10).reset_index()
    sample = df.sample(min(1500, len(df)), random_state=1)

    a, b = st.columns(2)
    with a:
        show(px.histogram(df, x="price_lakh", nbins=40, title="Price distribution",
                          labels={"price_lakh": "Price (lakh)"}))
    with b:
        show(px.bar(top, x="price_lakh", y="brand", orientation="h", title="Top 10 brands by average price",
                    labels={"price_lakh": "Average price (lakh)", "brand": "Brand"}))
    c, d = st.columns(2)
    with c:
        show(px.box(df, x="fuel", y="price_lakh", title="Price by fuel type",
                    labels={"price_lakh": "Price (lakh)", "fuel": "Fuel"}))
    with d:
        show(px.scatter(sample, x="car_age", y="price_lakh", title="Price against car age",
                        labels={"car_age": "Car age (years)", "price_lakh": "Price (lakh)"}))

# ---------- Model comparison ----------
with tab3:
    st.header("Model Comparison")
    st.markdown(f"**{best['Model']}** performed better on the test data with an R² of **{best['R2 (test)']:.2f}**.")
    st.dataframe(metrics, hide_index=True, width="stretch")

    a, b = st.columns(2)
    with a:
        show(px.bar(metrics, x="Model", y="R2 (test)", color="Model", title="R² (higher is better)"))
    with b:
        show(px.bar(metrics, x="Model", y="MAE", color="Model", title="MAE in rupees (lower is better)"))

    st.markdown("""
* **R²:** how much of the price variation the model explains. Closer to 1 is better.
* **MAE:** the average rupee gap between predicted and actual price.
* **RMSE:** like MAE, but punishes large errors more.
* **CV R²:** R² over 5 different data splits, to check the model is stable.
""")

# ---------- Price prediction ----------
with tab4:
    st.header("Estimate a car's price")
    now = meta["current_year"]
    inputs = {}
    left, right = st.columns(2)

    with left:
        for c in meta["cat_cols"]:
            inputs[c] = st.selectbox(c.replace("_", " ").title(), meta["cat_options"][c])

    with right:
        for c in meta["num_cols"]:
            s = meta["num_stats"][c]
            if c == "car_age":      # the user picks a year, the model needs the age
                year = st.slider("Manufacturing year", int(now - s["max"]), int(now - s["min"]), int(now - s["median"]))
                inputs[c] = now - year
            else:
                inputs[c] = st.number_input(c.replace("_", " ").title(),
                                            float(s["min"]), float(s["max"]), float(s["median"]))

    if st.button("Estimate price"):
        price = model.predict(pd.DataFrame([inputs])[meta["features"]])[0]
        low, high = max(price - best["MAE"], 0), price + best["MAE"]
        st.markdown(f'<div class="result"><p>Estimated selling price</p><h1>Rs {price:,.0f}</h1>'
                    f'<p>Expected range: Rs {low:,.0f} to Rs {high:,.0f}</p></div>', unsafe_allow_html=True)
        st.caption("The range is the prediction plus or minus the model's Mean Absolute Error (MAE).")

# ---------- Terms and privacy ----------
with tab5:
    st.header("Terms of Service")
    st.markdown("""
* This is an academic mini project, not a commercial service.
* Prices are statistical estimates, not official valuations, and come without any warranty.
* Data comes from the public CarDekho vehicle dataset on Kaggle. Brand names belong to their owners.
""")
    st.header("Privacy Policy")
    st.markdown("""
* No account is needed, and the app does not ask for your name, email or phone number.
* Car details you enter are used only to make the estimate and are not intentionally stored.
* The app uses no advertising or analytics cookies. Google Fonts is loaded, and the hosting provider may log technical data under its own policy.
""")
