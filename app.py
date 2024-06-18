import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly
import plotly.express as px
import platform

# plt = platform.system()
# if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

temp = pathlib.PosixPath
pathlib.PosixPath = temp


#title
st.title("Qushlarni rasmiga ko'ra klassifikatsiya qiluvchi model (Eagle, Magpie, Woodpecker, Sparrow)")
st.markdown("## Ushbu model Sunnatillo Xayrullayev tomonidan yaratilgan va quyidagi 4 ta turdagi qushlarni klassifikatsiya qilish uchun mo'ljallangan")

# rasmni joylash
file = st.file_uploader("Rasm yuklash", type=['png', 'jpg', 'gif', 'svg'])
if file:
    st.image(file)
    # PIL convert
    img = PILImage.create(file)
    
    # modelni yuklash
    model = load_learner('bird_model.pkl')
    
    # bashorat qilish prediction
    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")
    
    # plotting
    
    fig = px.bar(x= probs*100, y = model.dls.vocab)
    st.plotly_chart(fig)
