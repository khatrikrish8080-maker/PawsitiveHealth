import streamlit as st
import cv2
import numpy as np
import joblib
from PIL import Image
from vision_engine import get_analysed_results

st.title("Bira's Vision : Smart Health Analytics For Dog")
model_brain = joblib.load('dog_health_10_level_brain.pkl')

name=st.sidebar.text_input("Please enter your dog's name")


breed=st.sidebar.selectbox("choose your dog's breed",["Golden retriever","Pug","Shih-tzu",
                                                "Labrador","German Shepherd","Indie",
                                                "Doberman","Great Dane","Rottweiler",
                                                "Beagle","Cocker spaniel","Husky",
                                                "Chihuahua","Dachshund","Poodle",
                                                "Indian Spitz","Pomerian","ChowChow",
                                                "Bulldog","DachShund","PitBull","American Bully","Others"
                                                ])

dob=st.sidebar.date_input("Enter your dog's DATE OF BIRTH")

with st.sidebar.expander ("--- EMERGENCY RESOURCES SECTION ---"):
 st.divider()
 st.header("📍 Emergency Pet Care Network")
 st.info("Verified resources from our community research to assist you in urgent situations.")


 col1, col2 = st.columns(2)

 with col1:
    st.markdown("### 🩺 Paravets")
    st.caption("First-aid & Home visits")
    
    st.write("DR Manoj : 9871064380")
    st.write("DR Sanjay : 9599336762")
    st.write("DR Suresh : 9716585857")
    st.write("DR Rocky : 8512880968 ")

    st.markdown("### 🏥 Hospitals")
    st.caption("24/7 Veterinary Care")

    st.write("Faridabad Pet Clinic and Surgery centre  : 9582129385")
    st.write("VETIC Pet Clinic : 7042755007")
    st.write("Zigly For Pets : 807193681")
    st.write("Sarvodya Foundation : 9696534859 ")

 with col2:
    st.markdown("### 🚑 Ambulance")
    st.caption("Emergency Transport")

    st.write("Gopalji Ambulance : 9319419475")
    st.write("Guneet Ambulance : 9205567144")
    st.write("Gao Maa Ambulance: 9990142744")

    st.markdown("### 🩸 Blood Banks")
    st.caption("Canine Donors")
    st.write("Vets N Pets : 8010309629")
    st.write("MR Abhimanyu : 9810718318")
    st.write("CGS Hospital : 9311014490")



 


if st.button("Submit Your Details"):
 st.success(f"CONGRAGULATIONs !!! {name}'s Profile has been created and ready for health scan now")
 st.balloons()
 st.write(f"NAME : {name} "  )
 st.write(f"BREED : {breed} "  )
 st.write(f"DATE-OF-BIRTH : {dob} "  )
 st.warning("NOTE : FOR A 99.99% ACCURATE SCAN , ENSURE YOUR FURRY FRIEND IS NOT WEARING ANY CLOTHES , COLLAR OR HARNESS IN IMAGE YOU WILL PROVIDE")

uploaded_file=st.file_uploader("Pick a photo of your dog to upload",type=["jpg","jpeg","png"])
camera_photo=st.camera_input("Or take a live photo")

if uploaded_file is not None:
 st.image(uploaded_file,caption="Photo selected")
elif camera_photo is not None:
  st.image(camera_photo,caption=" Live photo selected")

final_image=None
if uploaded_file is not None:
 final_image=uploaded_file
elif camera_photo is not None:
 final_image=camera_photo

if final_image is not None:
 st.image(final_image,caption="Target image for AI scanner",width=300)
 
if st.button("START HEALTH ANALYSIS"):
 if final_image is not None:
       st.success(f"AI is now analysing {name}'s health markers......")
       st.balloons()
       file_bytes = np.asarray(bytearray(final_image.read()), dtype=np.uint8)
       opencv_bgr = cv2.imdecode(file_bytes, 1)

       marked_bgr, shine, redness, shape, detected = get_analysed_results(opencv_bgr)

       if detected:
            marked_rgb = cv2.cvtColor(marked_bgr, cv2.COLOR_BGR2RGB)
            st.image(marked_rgb, caption=f"Diagnostic Overlay for {name}", use_container_width=True)

            health_level = int(model_brain.predict([[shine, redness, shape]])[0])

            is_sitting=(shape<=1.2)
            if is_sitting:
              body_status = "INCONCLUSIVE (pose-dependant)"
              body_desc = "Dog appears sitting . Please provide standing side-profile for analysis"
              body_color = "gray"
            elif shape > 6.5:
             body_status = "Underweight"
             body_desc = "🔵 High Rib Visibility / Angular Frame"
             body_color = "blue"
            elif 5.2 < shape <= 6.5:
             body_status = "Slightly Underweight"
             body_desc = "🟡 Deep Waist Tuck / Pronounced Ribcage"
             body_color = "cyan"
            elif 3.4 < shape <= 5.2:
             body_status = "Fit"
             body_desc = "🟢 Ideal Anatomical Contour"
             body_color = "green"
            elif 1.7 < shape <= 3.4:
             body_status = "Overweight"
             body_desc = "🟠 Smooth Convex Outline / Minimal Waist Tuck"
             body_color = "orange"
            elif 1.2 < shape <=1.7 :
             body_status = "Obese"
             body_desc = "🔴 Circular Body Envelope / Distended Midsection"
             body_color = "red"

            st.markdown(f"### Body Composition: :{body_color}[{body_status}]")
            st.info(f"**Diagnostic Observation:** {body_desc}")

            if shine>75 and redness<2.0:
               health_level=0
            elif redness > 8.0:
               health_level=max(health_level,7)
            elif redness >= 6.0:
               health_level=max(health_level,6)
            elif redness >= 5.0:
                if breed == "Golden retriever" and shine>45 :
                 health_level=1
                else :
                  health_level=max(health_level,5)
            elif redness >= 4.0:
               health_level=max(health_level,4)
            elif redness>=3.0:
               health_level=3
            elif redness>=2.0:
               health_level=2
            else :
               health_level=health_level

            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Fur Shine", f"{shine}%")
            col2.metric("Skin Redness", f"{redness}%")
            col3.metric("Body Shape", f"{int(shape)}%")

            st.subheader(f"Final Assessment: Level {health_level}")
            
            if health_level == 0:
             if body_status == "Fit":
              st.success("🌟 LEVEL 0: THE BEST BEST. Perfect athletic frame and glowing coat!")
             else:
              st.warning(f"🟡 LEVEL 0: GLOWING BUT {body_status.upper()}. Skin is perfect, but body condition needs a diet adjustment.")

            elif health_level == 1:
             if body_status == "Fit":
              st.success("✅ LEVEL 1: EXCELLENT. Very healthy skin and ideal body weight.")
             else:
              st.warning(f"🟠 LEVEL 1: GOOD COAT / {body_status.upper()}. Great dermatology, but weight management is required.")

            elif health_level == 2:
             st.info(f"👍 LEVEL 2: GOOD. Body is {body_status}. Minimal dryness detected in fur.")

            elif health_level == 3:
             st.info(f"📄 LEVEL 3: FAIR. Body is {body_status}. Normal variations, monitor hydration.")

            elif health_level == 4:
             st.warning(f"⚠️ LEVEL 4: MILD IRRITATION. Body is {body_status}. Early signs of skin redness.")

            elif health_level == 5:
             st.warning(f"🟠 LEVEL 5: WARNING. Body is {body_status}. Visible inflammation found on skin.")

            elif health_level == 6:
             st.warning(f"🔴 LEVEL 6: POOR. Body is {body_status}. Significant redness; monitor closely.")

            elif health_level == 7:
             st.error(f"🎀 LEVEL 7: DISTRESS. Body is {body_status}. Possible infection detected.")

            elif health_level == 8:
             st.error(f"🆘 LEVEL 8: SEVERE. Body is {body_status}. High redness and moderate luster")
            elif health_level == 9:
             st.error(f"⚠️ LEVEL 9: STAY CALM. Body is {body_status}. It's better to have a check-up with vet!")

            st.progress(int((health_level / 9) * 100))

            with st.expander("🔍 Clinical Methodology: How the AI Analyzes Your Pet"):
             st.markdown("### 🧪 Automated Diagnostic Parameters")
    
             col_x, col_y, col_z = st.columns(3)
    
             with col_x:
              st.markdown("**✨ Fur Luster (Shine)**")
              st.write("🟢 **HIGH IS BETTER**")
              st.caption("Target: >75%. High luster indicates optimal sebaceous gland activity and nutrient absorption.")
        
             with col_y:
              st.markdown("**🔴 Dermal Erythema (Redness)**")
              st.write("🟢 **LOW IS BETTER**")
              st.caption("Target: <1.0%. Low values confirm an absence of skin irritation, allergies, or inflammation.")
        
             with col_z:
              st.markdown("**🐕 Somatic Morphology (Shape)**")
              st.write("🟢 **MID-RANGE IS BEST**")
              st.caption("Target: 4.0% - 5.2%. This 'Golden Ratio' represents a fit, athletic frame with a visible waist tuck.")

             st.divider()
             st.info("📊 **Summary:** Our AI prioritizes **High Shine**, **Low Redness**, and **Balanced Shape** to determine the final Health Level.")
           

       else:
            st.error("❌ No Dog Detected! Please upload a clearer photo of your dog.")
 else:
        st.error("Please upload a photo first!")
 

      