import streamlit as st
import pandas as pd
from databricks.sdk import WorkspaceClient
import os
import time

#TODO 
# Page & UI Setup
st.set_page_config(page_title="GAE AI Agent Portal", layout="wide", page_icon="🗺️")
st.title("🗺️ Geoanalytics AI Data Agent")
st.caption("Powered by Databricks Apps, Genie, and the Geoanalytics Engine (GAE)")

GENIE_SPACE_ID = '01f16b0f00271be69d67170685241974'
# Initialize Databricks SDK Client safely
@st.cache_resource
def get_workspace_client():a
    # Databricks Apps handles OAuth/OIDC automatically; no tokens or hosts needed here.
    return WorkspaceClient()

try:
    w = get_workspace_client()
except Exception as e:
    st.error(f"Failed to connect to Databricks Workspace: {e}")
    st.stop()

#  Retrieve Genie Space ID from Environment Variables (set via app.yaml resource binding)
GENIE_SPACE_ID = os.getenv("GENIE_SPACE_ID")
if not GENIE_SPACE_ID:
    st.error(
        " GENIE_SPACE_ID is not set. Add the Genie Space as a resource in your "
        "Databricks App config (app.yaml) so it's injected automatically."
    )
    st.stop()

#  Handle Conversation Session State
# Note: Databricks recommends starting a fresh conversation per session rather than
# reusing threads across sessions, to avoid unintended context bleed.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = None

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sql" in message:
            with st.expander("View Executed Spatial SQL"):
                st.code(message["sql"], language="sql")
        if "data" in message:
            st.dataframe(message["data"])
            geo_cols = {c for c in message["data"].columns if c.lower() in ("lat", "latitude", "lon", "lng", "longitude")}
            if {"lat", "lon"}.issubset({c.lower() for c in message["data"].columns}):
                st.map(message["data"])


def get_attachment_query_result(space_id, conversation_id, message_id, attachment_id):
    """Fetch the tabular result for a SQL attachment Genie generated."""
    result = w.genie.execute_message_attachment_query(
        space_id=space_id,
        conversation_id=conversation_id,
        message_id=message_id,
        attachment_id=attachment_id,
    )
    statement_response = result.statement_response
    if not statement_response or not statement_response.result:
        return None
    columns = [c.name for c in statement_response.manifest.schema.columns]
    rows = statement_response.result.data_array or []
    return pd.DataFrame(rows, columns=columns)


# 5. Capture User Proximity / Geospatial Query
if prompt := st.chat_input(
    "Ask a geoanalytics question (e.g., 'Find all store anomalies within a 10km buffer of New York')"
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_text = st.empty()
        status_text.markdown(" *Sending question to Genie (space is backed by Unity Catalog tables using GAE ST_/H3_ functions)...*")

        try:
            if st.session_state.conversation_id is None:
                response = w.genie.start_conversation_and_wait(
                    space_id=GENIE_SPACE_ID,
                    content=prompt,
                )
                st.session_state.conversation_id = response.conversation_id
            else:
                response = w.genie.create_message_and_wait(
                    space_id=GENIE_SPACE_ID,
                    conversation_id=st.session_state.conversation_id,
                    content=prompt,
                )

            status_text.markdown("*Retrieving spatial query results...*")

            assistant_text_parts = []
            generated_sql = None
            df = None

            for attachment in (response.attachments or []):
                if attachment.text and attachment.text.content:
                    assistant_text_parts.append(attachment.text.content)
                if attachment.query:
                    generated_sql = attachment.query.query
                    df = get_attachment_query_result(
                        GENIE_SPACE_ID,
                        response.conversation_id,
                        response.id,
                        attachment.attachment_id,
                    )

            assistant_md = "\n\n".join(assistant_text_parts) or "Here are the results of your spatial query:"
            status_text.markdown(assistant_md)

            if generated_sql:
                with st.expander("View Executed Spatial SQL"):
                    st.code(generated_sql, language="sql")

            if df is not None and not df.empty:
                st.dataframe(df)
                cols_lower = {c.lower(): c for c in df.columns}
                if "lat" in cols_lower and "lon" in cols_lower:
                    st.map(df.rename(columns={cols_lower["lat"]: "lat", cols_lower["lon"]: "lon"}))
                elif "latitude" in cols_lower and "longitude" in cols_lower:
                    st.map(df.rename(columns={cols_lower["latitude"]: "lat", cols_lower["longitude"]: "lon"}))
                else:
                    st.caption(
                        "No lat/lon columns detected for map rendering. If your spatial SQL returns "
                        "geometry (e.g. ST_AsGeoJSON), consider rendering it with pydeck/folium."
                    )
            elif generated_sql:
                st.info("Query ran successfully but returned no rows.")

            history_entry = {"role": "assistant", "content": assistant_md}
            if generated_sql:
                history_entry["sql"] = generated_sql
            if df is not None:
                history_entry["data"] = df
            st.session_state.messages.append(history_entry)

        except Exception as e:
            error_message = f"An error occurred processing the geoanalytics task: `{str(e)}`"
            status_text.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})

