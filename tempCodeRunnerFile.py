import streamlit as st
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
warnings.filterwarnings("ignore",category=UserWarning)
import subprocess

def run_delhi():
    try:
        result=subprocess.run(["streamlit","run","delhi.py"], check=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running:{e}")

def run_guwahati():
    try:
        result=subprocess.run(["streamlit","run","guwahati.py"], check=True, text=True)
        st.write(result.stdout)
    except subprocess.CalledProcessError as e:
        st.error(f"Error running:{e}")

def main():
    st.write("<h1 style='text-align: center;'>Safe or Unsafe Earthquake Zone Predictor</h1>", unsafe_allow_html=True)
    if st.button("Delhi"):
        run_delhi()

    elif st.button("Guwahati"):
        run_guwahati()      
    
    else:
        st.write("Choose a City")
          
if __name__ == "__main__":
    main()
