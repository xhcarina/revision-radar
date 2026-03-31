import sys
sys.path.insert(0, ".")

import streamlit as st

st.set_page_config(page_title="Diag")

st.title("BOOT OK")
st.write(f"Python: {sys.version}")

try:
    import pandas as pd
    st.success("pandas ok")
except Exception as e:
    st.error(f"pandas FAILED: {e}"); st.stop()

try:
    import numpy as np
    st.success("numpy ok")
except Exception as e:
    st.error(f"numpy FAILED: {e}"); st.stop()

try:
    import plotly
    st.success("plotly ok")
except Exception as e:
    st.error(f"plotly FAILED: {e}"); st.stop()

try:
    import yfinance
    st.success("yfinance ok")
except Exception as e:
    st.error(f"yfinance FAILED: {e}"); st.stop()

try:
    import gdeltdoc
    st.success("gdeltdoc ok")
except Exception as e:
    st.error(f"gdeltdoc FAILED: {e}"); st.stop()

try:
    import src.data.sec_client
    st.success("sec_client ok")
except Exception as e:
    st.error(f"sec_client FAILED: {e}"); st.stop()

try:
    import src.extraction.llm_extractor
    st.success("llm_extractor ok")
except Exception as e:
    st.error(f"llm_extractor FAILED: {e}"); st.stop()

st.balloons()
st.success("All imports OK — ready to run main app")
