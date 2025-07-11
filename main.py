import os
import speech_recognition as sr
import pyttsx3
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# 📄 Dokumentumok betöltése
docs = []
for filename in os.listdir("documents"):
    if filename.endswith(".txt"):
        loader = TextLoader(os.path.join("documents", filename))
        docs.extend(loader.load())
    elif filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join("documents", filename))
        docs.extend(loader.load())

# 📄 Feldarabolás & embedding
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
all_texts = text_splitter.split_documents(docs)

# 🧠 Vektoradatbázis
embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(all_texts, embeddings)
retriever = db.as_retriever()

# 🧠 LLM + RAG
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
)

# 🔈 TTS setup
engine = pyttsx3.init()

def speak(text):
    print(f"AI: {text}")
    engine.say(text)
    engine.runAndWait()

# 🎤 STT setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen():
    with mic as source:
        print("🎤 Kérlek beszélj...")
        audio = recognizer.listen(source)
    try:
        query = recognizer.recognize_google(audio, language="hu-HU")
        print(f"Te: {query}")
        return query
    except sr.UnknownValueError:
        print("Nem értettem. Próbáld újra.")
        return None

# 🚀 Fő loop
while True:
    question = listen()
    if question:
        answer = qa_chain.run(question)
        speak(answer)
