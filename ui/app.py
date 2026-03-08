"""Streamlit demo UI."""
import streamlit as st
import httpx

API_URL = "http://localhost:8000"

st.title("Financial Report Analyst")
st.caption("Powered by SEC EDGAR & Companies House + LangGraph")

company = st.text_input("Company name or ticker", placeholder="Apple / AAPL")
filing_type = st.selectbox("Filing type", ["10-K", "10-Q"])
query = st.text_area("Question", placeholder="What are the key risk factors?")

if st.button("Analyze") and query:
    with st.spinner("Analysing..."):
        try:
            resp = httpx.post(
                f"{API_URL}/analyze",
                json={"query": query, "company": company, "filing_type": filing_type},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            st.markdown(data["report"])
            if data.get("sources"):
                with st.expander("Sources"):
                    for s in data["sources"]:
                        st.write(s)
        except Exception as e:
            st.error(f"Error: {e}")
