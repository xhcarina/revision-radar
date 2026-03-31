import sys
sys.path.insert(0, ".")

import streamlit as st

st.write("BOOT")

try:
    import pandas as pd
    st.write("pandas ok")

    import numpy as np
    st.write("numpy ok")

    import plotly
    st.write("plotly ok")

    import yfinance
    st.write("yfinance ok")

    import gdeltdoc
    st.write("gdeltdoc ok")

    import src.data.sec_client
    st.write("sec_client ok")

    import src.extraction.llm_extractor
    st.write("llm_extractor ok")

    st.success("All imports OK")

except Exception as e:
    import traceback
    st.error(f"ERROR: {e}")
    st.code(traceback.format_exc())
