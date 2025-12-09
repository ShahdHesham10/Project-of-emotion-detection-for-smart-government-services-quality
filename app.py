import streamlit as st
import torch
import pandas as pd
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="GovCare", page_icon="💎", layout="centered")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; }
    .stApp { background-image: linear-gradient(180deg, #ffffff 0%, #f0f8ff 100%); }
    .main-title { color: #0077b6; text-align: center; font-size: 2.5rem; font-weight: bold; }
    div.stButton > button { background: linear-gradient(90deg, #48cae4 0%, #0077b6 100%); color: white; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)

MODEL_PATH = "./my_model"
DATA_FILE = "complaints_data.csv" 

@st.cache_resource
def load_local_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        return tokenizer, model
    except Exception:
        return None, None

tokenizer, model = load_local_model()

def analyze_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
    
    neg_score = probs[0]
    pos_score = probs[1]
    
    if pos_score > neg_score:
        return "إيجابي 😍", pos_score
    else:
        if neg_score < 0.65:
            return "محايد 😐", neg_score
        else:
            return "سلبي 💔", neg_score

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2362/2362057.png", width=100)
st.sidebar.title("نظام GovCare")
role = st.sidebar.radio("تسجيل الدخول بصفتك:", ["مواطن 👤", "موظف / مسؤول 👮‍♂️"])


if role == "مواطن 👤":
    st.markdown('<div class="main-title">أهلاً بك عزيزي المواطن</div>', unsafe_allow_html=True)
    st.write("نحن هنا لسماع صوتك. سيتم تسجيل شكواك ومراجعتها من قبل المسؤولين.")
    
    citizen_text = st.text_area("اكتب شكواك أو مقترحك:", height=150)
    
    if st.button("📤 إرسال الشكوى"):
        if not citizen_text.strip():
            st.warning("⚠️ يرجى كتابة نص الشكوى")
        else:
            if tokenizer:
                label, score = analyze_text(citizen_text)
                
                new_data = pd.DataFrame({
                    "التاريخ": [datetime.now().strftime("%Y-%m-%d %H:%M")],
                    "نص الشكوى": [citizen_text],
                    "تصنيف الذكاء الاصطناعي": [label],
                    "درجة الثقة": [f"{score:.2f}"]
                })
                
                if os.path.exists(DATA_FILE):
                    new_data.to_csv(DATA_FILE, mode='a', header=False, index=False, encoding='utf-8-sig')
                else:
                    new_data.to_csv(DATA_FILE, mode='w', header=True, index=False, encoding='utf-8-sig')
                
                st.success("✅ تم استلام شكواك بنجاح! شكراً لتواصلك معنا.")
                st.balloons()
            else:
                st.error("الموديل غير جاهز")


else:
    st.markdown('<div class="main-title">لوحة تحكم المسؤولين</div>', unsafe_allow_html=True)
    
    password = st.sidebar.text_input("كلمة المرور للموظفين", type="password")
    
    if password == "admin123":  
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي الشكاوى", len(df))
            col2.metric("عدد الشكاوى السلبية", len(df[df['تصنيف الذكاء الاصطناعي'].str.contains("سلبي")]))
            col3.metric("عدد الشكاوى الإيجابية", len(df[df['تصنيف الذكاء الاصطناعي'].str.contains("إيجابي")]))
            
            st.markdown("---")
            st.subheader("📋 سجل الشكاوى الواردة")
            st.dataframe(df, use_container_width=True)
            
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("تحميل التقرير (Excel)", csv, "report.csv", "text/csv")
            
        else:
            st.info("📭 لا توجد شكاوى مسجلة حتى الآن.")
    else:
        if password:
            st.error("كلمة المرور خاطئة!")
        else:
            st.warning("🔒 يرجى إدخال كلمة المرور لرؤية البيانات")