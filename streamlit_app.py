import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration

from openai import OpenAI

import cv2
import av
import os

import base64

openai_key = st.secrets["openai_key"]
openai = OpenAI(api_key=openai_key)

# Setting page layout
st.set_page_config(
    page_title="Video Frame Analysis",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Sidebar
st.sidebar.header("About App")
st.sidebar.markdown('This is an app that utilizes computer vision to analyze video frames and describe what is happening in natural language using OpenAI GPT 4o-mini model created by <a href="https://ai.jdavis.xyz" target="_blank">0xjdavis</a>.', unsafe_allow_html=True)

# Calendly
st.sidebar.markdown("""
    <hr />
    <center>
    <div style="border-radius:8px;padding:8px;background:#fff";width:100%;">
    <img src="https://avatars.githubusercontent.com/u/98430977" alt="Oxjdavis" height="100" width="100" border="0" style="border-radius:50%"/>
    <br />
    <span style="height:12px;width:12px;background-color:#77e0b5;border-radius:50%;display:inline-block;"></span> <b>I'm available for new projects!</b><br />
    <a href="https://calendly.com/0xjdavis" target="_blank"><button style="background:#126ff3;color:#fff;border: 1px #126ff3 solid;border-radius:8px;padding:8px 16px;margin:10px 0">Schedule a call</button></a><br />
    </div>
    </center>
    <br />
""", unsafe_allow_html=True)

# Copyright
st.sidebar.caption("©️ Copyright 2024 J. Davis")

class VideoProcessor:
  def __init__(self):
    self.capture_frame = True
    self.image_saved = False
    self.image_path = ''

  def recv(self, frame):
    img = frame.to_ndarray(format="bgr24")

    if self.capture_frame:
      # Save the image
      self.image_path = 'captured_frame.png'
      cv2.imwrite(self.image_path, img)
      # Reset Flags
      self.capture_frame = False
      self.image_saved = True

    return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title('Video Frame Analysis')
st.write("Instructions")
st.caption(
  """
  1. Select your video input device. Then press the Start button to initiate the video feed. 
  2. Click the Capture Frame button to analyze what is going on.
  3. Click the Stop button to end the session.
  """
)

# VIDEO PROCESSING
RTC_CONFIGURATION = RTCConfiguration(
  {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

video_processor = VideoProcessor()
webrtc_ctx = webrtc_streamer(
  key="Video", 
  video_processor_factory=VideoProcessor, 
  rtc_configuration=RTC_CONFIGURATION
)

# CAPTURE VIDEO FRAME
if webrtc_ctx.video_processor:
  st.caption(
    """Click the Capture Frame button to take a screen shot of the video and analyze what is going on."""
  )
  if st.button("Capture Frame"):
    # Set a flag when the button is pressed
    webrtc_ctx.video_processor.capture_frame = True

  # Display the image if it is saved
  if webrtc_ctx.video_processor.image_saved:
    if os.path.exists(webrtc_ctx.video_processor.image_path):
      st.image(webrtc_ctx.video_processor.image_path)
      st.write(webrtc_ctx.video_processor.image_path)
      # Read the image and convert it to base64
      with open(webrtc_ctx.video_processor.image_path, "rb") as img_file:
        base64_image = base64.b64encode(img_file.read()).decode("utf-8")

      # ANALYSIS CAN BE DONE WITH VARIOUS MODELS (GROQ, OPENAI, ...)
      # OpenAI API Send request to...
      with st.spinner('OpenAI API Sending Request...'):
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        "Please explain specifically what you see.",
                        {"image": base64_image, "resize": 768},
                    ],
                },
            ],
            max_tokens=500,
        )
      # View Response
      st.write(response.choices[0].message.content)

      webrtc_ctx.video_processor.image_saved = False
    else:
      st.write(f"Image file does not exist: {webrtc_ctx.video_processor.image_path}")
