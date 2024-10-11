import os
import shutil

import pandas as pd
import streamlit as st

from agent import AgentExecutor
from tools_set.Rag_tool import RAGService
from version_base.until.table_data_preprocess import preprocess_table, get_all_file_paths

st.set_page_config(layout="wide")

common_bubble_style = """
<style>
.bubble-container {
    display: flex;
    justify-content: flex-end;
}

.bubble {
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    position: relative;
    max-width: 65%;
    word-wrap: break-word;
    font-size: 16px;
}

.user-bubble {
    background-color: #d9edf7;
    border-radius: 15px 0 15px 15px;
    text-align: right;
}

.bot-bubble {
    background-color: #5bc0de;
    color: white;
    border-radius: 0 15px 15px 15px;
    text-align: left;
}

@media (max-width: 600px) {
    .bubble {
        max-width: 90%;
    }
}

.bubble-icon {
    font-size: 24px;
}
</style>
"""


def display_processed_data(markdown, des):
    show_full = st.checkbox(des)
    if show_full:
        st.dataframe(markdown, use_container_width=True)
    else:
        st.dataframe(markdown.head(10), use_container_width=True)
    num_rows, num_cols = markdown.shape
    st.markdown(
        f"<p style='font-size:16px;'>数据表包含100万行和100列</p>",
        unsafe_allow_html=True
    )


def initialize_session_state():
    initial_data = {
        "messages": [],
        "embed": RAGService(),
        "tools": None,
        "model": AgentExecutor(),
        "table": None,
        "table_des": None,
    }
    for key, value in initial_data.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_chat_message(user_role, message_content):
    bubble_class = 'user-bubble' if user_role == 'user' else 'bot-bubble'
    container_align = 'flex-end' if user_role == 'user' else 'flex-start'

    col_icon, col_message, col_user_icon = st.columns([1, 18, 1])

    if user_role == 'user':
        with col_user_icon:
            st.markdown(f"<p class='bubble-icon' style='text-align:{container_align};'>{'👤'}</p>",
                        unsafe_allow_html=True)
    else:
        with col_icon:
            st.markdown(f"<p class='bubble-icon' style='text-align:{container_align};'>{'🤖'}</p>",
                        unsafe_allow_html=True)

    with col_message:
        st.markdown(common_bubble_style, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="bubble-container" style="justify-content: {container_align};">
                <div class="bubble {bubble_class}">
                    {message_content}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


if __name__ == '__main__':
    initialize_session_state()
    st.markdown("<h1 style='color:#007BFF; text-align:center;'>TableAgent 💬</h1>", unsafe_allow_html=True)

    with st.sidebar:
        uploaded_files = st.file_uploader("请上传您的表格：", type=['csv', 'xlsx'], accept_multiple_files=True)
        if uploaded_files:
            save_directory = 'cache/data/'
            os.makedirs(save_directory, exist_ok=True)
            index = 0
            for uploaded_file in uploaded_files:
                original_filename = uploaded_file.name
                index += 1
                save_path = os.path.join(save_directory, original_filename)
                with open(save_path, 'wb') as f:
                    f.write(uploaded_file.read())

                file_extension = original_filename.split('.')[-1]
                if file_extension == 'csv':
                    st.session_state.table = pd.read_csv(save_path)
                elif file_extension == 'xlsx':
                    st.session_state.table = pd.read_excel(save_path)
                else:
                    st.error("不支持的文件格式，请上传 .csv 或 .xlsx 文件。")
                with st.expander(f"{original_filename}"):
                    display_processed_data(st.session_state.table, f"显示全部{original_filename}数据")

            file_list = get_all_file_paths(save_directory)
            st.session_state.embed.initialize_vector_store(file_list)

            st.session_state.table_des = preprocess_table(save_directory)

            shutil.rmtree(save_directory)

    with st.form(key="query_form"):
        user_input = st.text_area("输入您要查询的问题:", value="", key='user_input', help='在这里输入你的问题')
        submit_button = st.form_submit_button(label="🔍 提交查询", use_container_width=True)

    if submit_button and user_input:
        output = st.session_state.model.agent_execute(user_input, st.session_state.table_des)

        st.session_state.messages.append({"role": "user", "content": user_input.strip()})
        st.session_state.messages.append({"role": "assistant", "content": output})

        user_input = ""
    with st.expander(expanded=True, label='对话记录'):
        for msg in st.session_state.messages:
            display_chat_message(msg["role"], msg["content"])
