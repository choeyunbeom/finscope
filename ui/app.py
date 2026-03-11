"""Streamlit demo UI."""

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Financial Report Analyst", page_icon="📊", layout="wide")

st.title("📊 Financial Report Analyst")
st.caption("Multi-agent analysis powered by SEC EDGAR & LangGraph")

with st.sidebar:
    st.header("Settings")
    source = st.selectbox("Data source", ["sec", "ch"], format_func=lambda x: "SEC EDGAR" if x == "sec" else "Companies House")
    filing_type = st.selectbox("Filing type", ["10-K", "10-Q"])
    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Enter a company name or ticker\n"
        "2. Ask any question about the filing\n"
        "3. 3 agents (Retriever → Analyzer → Critic) collaborate to answer"
    )

col1, col2 = st.columns([1, 2])

with col1:
    company = st.text_input("Company name or ticker", placeholder="Apple / AAPL")
    query = st.text_area(
        "Question",
        placeholder="What are the key risk factors?",
        height=120,
    )
    analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

with col2:
    if analyze_btn:
        if not query:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Running multi-agent analysis... (this may take 30–60s)"):
                try:
                    resp = httpx.post(
                        f"{API_URL}/analyze",
                        json={
                            "query": query,
                            "company": company or None,
                            "filing_type": filing_type,
                            "source": source,
                        },
                        timeout=180,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    st.success(f"Analysis complete (retries: {data.get('retry_count', 0)})")
                    st.markdown("### Report")
                    st.markdown(data["report"])

                    if data.get("sources"):
                        with st.expander(f"Sources ({len(data['sources'])} chunks)"):
                            for s in data["sources"]:
                                st.markdown(f"- {s}")

                except httpx.ConnectError:
                    st.error("Cannot connect to API. Run: `uvicorn src.api.main:app --reload`")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Enter a company and question, then click **Analyze**.")
