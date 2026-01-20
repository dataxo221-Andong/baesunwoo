# OpenAI Embeddings를 이용한 벡터화
# FAISS 벡터 데이터베이스 구축

import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.messages.chat import ChatMessage
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# 기본 설정
load_dotenv()

if not os.path.exists(".cache"):
    os.mkdir(".cache")
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

st.title("PDF 기반 QA 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "retriever" not in st.session_state:
    st.session_state["retriever"] = None

# 벡터 데이터베이스 생성 함수
@st.cache_resource(show_spinner="벡터 데이터베이스를 생성 중입니다...")
def create_vectorstore(file):
    """PDF에서 벡터 데이터베이스를 생성하는 함수"""
    
    # 1. 파일 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    
    # 2. 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    
    # 3. 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50
    )
    split_documents = text_splitter.split_documents(docs)
    
    # 4. 임베딩 생성
    embeddings = OpenAIEmbeddings()
    
    # 5. 벡터 데이터베이스 생성
    vectorstore = FAISS.from_documents(
        documents=split_documents, 
        embedding=embeddings
    )
    
    # 6. 검색기 생성
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}  # 상위 3개 관련 문서 검색
    )
    
    return retriever, len(split_documents), len(docs)

# 검색 테스트 함수
def test_retrieval(retriever, query):
    """검색 기능을 테스트하는 함수"""
    if retriever is None:
        return []
    
    try:
        relevant_docs = retriever.get_relevant_documents(query)
        return relevant_docs
    except Exception as e:
        st.error(f"검색 중 오류: {e}")
        return []

with st.sidebar:
    st.header("📁 문서 업로드")
    clear_btn = st.button("대화 초기화")
    uploaded_file = st.file_uploader("PDF 파일 업로드", type=["pdf"])
    selected_model = st.selectbox("LLM 선택", ["gpt-4o-mini", "gpt-4.1-nano"], index=0)
    
    # 검색 설정
    st.subheader("🔍 검색 설정")
    search_k = st.slider("검색할 문서 수", 1, 10, 3)

def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

if clear_btn:
    st.session_state["messages"] = []

# 벡터 데이터베이스 생성 및 표시
if uploaded_file:
    try:
        retriever, num_chunks, num_pages = create_vectorstore(uploaded_file)
        st.session_state["retriever"] = retriever
        
        # 성공 메시지
        st.success("✅ 벡터 데이터베이스 생성 완료!")
        
        # 통계 정보
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📄 페이지", num_pages)
        with col2:
            st.metric("📝 청크", num_chunks)
        with col3:
            st.metric("🔍 검색 준비", "완료")
        
        # 검색 테스트 영역
        with st.expander("🔍 검색 시스템 테스트"):
            test_query = st.text_input(
                "검색할 내용을 입력하세요:", 
                placeholder="예: 주요 내용, 핵심 키워드 등"
            )
            
            if test_query:
                with st.spinner("검색 중..."):
                    relevant_docs = test_retrieval(retriever, test_query)
                
                if relevant_docs:
                    st.write(f"**'{test_query}'에 대한 검색 결과:**")
                    for i, doc in enumerate(relevant_docs):
                        st.write(f"**결과 {i+1}:**")
                        st.text_area(
                            f"내용 {i+1}", 
                            doc.page_content, 
                            height=100, 
                            key=f"search_result_{i}"
                        )
                        st.divider()
                else:
                    st.warning("관련 문서를 찾을 수 없습니다.")
        
    except Exception as e:
        st.error(f"벡터 데이터베이스 생성 중 오류: {e}")

print_messages()

user_input = st.chat_input("문서에 대해 궁금한 내용을 물어보세요!")

if user_input:
    if st.session_state["retriever"] is not None:
        add_message("user", user_input)
        st.chat_message("user").write(user_input)
        
        # 관련 문서 검색
        relevant_docs = test_retrieval(st.session_state["retriever"], user_input)
        
        if relevant_docs:
            # 검색된 문서를 기반으로 한 임시 응답
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            response = f"""
**검색된 관련 문서 ({len(relevant_docs)}개):**

{context[:500]}...

*AI 생성 답변은 다음 단계에서 구현됩니다.*
            """
            add_message("assistant", response)
            st.chat_message("assistant").write(response)
        else:
            st.warning("관련 문서를 찾을 수 없습니다.")
    else:
        st.error("먼저 PDF 파일을 업로드해주세요!")