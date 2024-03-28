# Libraries
import streamlit as st  # type: ignore
import base64

# This is the home page which is displayed when the app is first launched
st.set_page_config(page_title="Home page 🏠", page_icon=":bar_chart:", layout="wide")
st.image("web_app/images/instadeep_cropped_logo.png", width=600)
st.title("Reinforcement Learning environment for component placement")
st.write(
    """
            Printed circuit boards (PCBs) are
            the core foundation which supports the majority
            of electronic products produced in the economy.
            Due to the rapid progress in artificial intelligence,
            there has been significant potential for remarkable advances
            in computer systems and hardware. While the PCB industry has advanced
            significantly over the decades by decreasing the size while simultaneously
            increasing speed and capabilities of electronic components, the design cycle
            for manufacturing PCB's has achieved minimal development and optimization.
            Despite decades of research on this problem, the current industry
            procedures still require human experts to manually modify PCB's using placement
            tools inorder to produce solutions that meet the multi-faceted design criterion's.
            In this project, we apply a learning based approach, by training a reinforcement learning agent
            to optimize the component placement task in a grid environment.
        """
)

# Reads video and displays it
video_file = open("web_app/images/Instadeep_video.webm", "rb")
video_bytes = video_file.read()
video_base64 = base64.b64encode(video_bytes).decode("utf-8")
video_html = f"""
<video controls width="600" autoplay="true" muted="true" loop="true">
    <source src="data:video/webm;base64,{video_base64}" type="video/webm">
</video>
"""
st.markdown(video_html, unsafe_allow_html=True)

st.header("Project overview")
st.write(
    """The project is divided into 3 main parts:
1. Reinforcement Learning environment: We create a custom environment for the component placement task.
2. Reinforcement Learning agent: We train a reinforcement learning agent to optimize the component placement task.
3. Research anlysis: We study which variables are most influencial to training the agent on the environment."""
)

st.subheader("Trained agents")
st.write(
    """This page allows you to view the statistics of past trained agents on specific environments."""
)

st.subheader("Train agents")
st.write("""This page allows you to train agents on specific environments.""")

st.subheader("Comparison analysis")
st.write(
    """This page allows you to compare the performance of different trained agents on specific environments."""
)
