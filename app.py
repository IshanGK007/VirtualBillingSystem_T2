
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from fpdf import FPDF
import tempfile
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Disable GPU

# Load class labels from labels.txt
def load_labels(label_file):
    with open(label_file, 'r') as file:
        labels = file.read().splitlines()
    return labels

# Path to your labels.txt file
label_file = "labels.txt"
class_labels = load_labels(label_file)

# Load the YOLOv8 model
model = YOLO("best (1).pt")

# Define item prices in Indian Rupees
item_prices = {
    '4D_medical_face-mask': 10.0,
    'Let-green_alcohol_wipes': 15.0,
    'X-men': 20.0,
    'aquafina': 5.0,
    'cart': 12.0,
    'life-buoy': 8.0,
    'luong_kho': 7.0,
    'milo': 9.0,
    'teppy_orange_juice': 6.0
}

# Streamlit App Header with a clean design
st.markdown("""
    <style>
        .title {
            color: white;
            font-size: 40px;
            font-family: 'Roboto', sans-serif;
            font-weight: bold;
        }
        .subheader {
            color: #3f51b5;
            font-size: 20px;
            font-family: 'Roboto', sans-serif;
        }
        .description {
            font-size: 16px;
            font-family: 'Roboto', sans-serif;
            color: #424242;
        }
        .container {
            background-color: #f3f3f3;
            border-radius: 10px;
            padding: 20px;
        }
        .bill-table {
            width: 100%;
            border-collapse: collapse;
        }
        .bill-table th, .bill-table td {
            padding: 12px;
            text-align: left;
            border: 1px solid #ddd;
        }
        .bill-table th {
            background-color: #1a1a1a;
            color: white;
        }
        .bill-table tr:hover {
            background-color: #f5f5f5;
            color : black;
        }
        .total-row {
            background-color: #E1E1E1;
            font-weight: bold;
            font-size: 18px;
            color: black;
        }
        .button {
            background-color: #3f51b5;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 18px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .button:hover {
            background-color: #303f9f;
        }
        .image-container {
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        }
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Title and Description
st.markdown('<div class="title">YOLOv11 Object Detection and Billing App</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Upload or capture an image of items to generate a bill</div>', unsafe_allow_html=True)

# Add option to either upload an image or use the camera
image_option = st.radio("Select Image Input Option", ["Upload Image", "Take Picture"])

# File uploader or Camera input
if image_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

elif image_option == "Take Picture":
    captured_image = st.camera_input("Take a picture of the items...")
    image = None
    if captured_image is not None:
        image = Image.open(captured_image)
        st.image(image, caption="Captured Image", use_container_width=True)

# If an image is available for processing
if image is not None:
    # Display a progress bar while processing
    with st.spinner("Processing image..."):
        # Convert image to numpy array and perform inference
        image_np = np.array(image)
        results = model.predict(image_np, conf=0.7, iou=0.4)

        # Annotate the image with bounding boxes and labels
        annotated_frame = results[0].plot()

    # Display the annotated image
    st.image(annotated_frame, caption="Detected Objects", use_container_width=True)

    # Initialize a dictionary to count detected items
    item_counts = {name: 0 for name in item_prices.keys()}

    # Process results to count items
    detected_classes = []  # List to track detected classes for debugging

    for detection in results[0].boxes.data.tolist():
        _, _, _, _, _, class_id = detection
        class_name = class_labels[int(class_id)] if int(class_id) < len(class_labels) else "Unknown"
        class_name = class_name.strip()  # Remove leading/trailing spaces
        
        # Log detected class name for debugging
        detected_classes.append(class_name)
        
        # Count the items in the item list
        if class_name in item_counts:
            item_counts[class_name] += 1

    # Generate the bill based on counts and prices
    total_bill = 0
    bill_details = []
    for item, count in item_counts.items():
        if count > 0:
            item_total = count * item_prices[item]
            total_bill += item_total
            bill_details.append(f"{item}: {count} x Rs{item_prices[item]:.2f} = Rs{item_total:.2f}")

    # Beautiful Bill Display
    st.markdown('<div class="subheader">Bill Summary</div>', unsafe_allow_html=True)
    bill_html = """
    <table class="bill-table">
        <thead>
            <tr><th>Item</th><th>Quantity</th><th>Price</th><th>Total</th></tr>
        </thead>
        <tbody>
    """
    for item, count in item_counts.items():
        if count > 0:
            item_total = count * item_prices[item]
            bill_html += f"<tr><td>{item}</td><td>{count}</td><td>Rs{item_prices[item]:.2f}</td><td>Rs{item_total:.2f}</td></tr>"
    bill_html += f"<tr class='total-row'><td colspan='3'>Total Bill</td><td>Rs{total_bill:.2f}</td></tr>"
    bill_html += "</tbody></table>"
    st.markdown(bill_html, unsafe_allow_html=True)

    # PDF Generation with improved layout
    def generate_pdf(bill_details, total_bill):
        pdf = FPDF()
        pdf.add_page()
        
        # Header with Store Name
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, txt="Visual Based Generated Bill Summary", ln=True, align='C')
        pdf.ln(10)

        # Table Header
        pdf.set_font("Arial", "B", 12)
        pdf.cell(80, 10, "Item", 1, 0, 'C')
        pdf.cell(30, 10, "Quantity", 1, 0, 'C')
        pdf.cell(40, 10, "Price (Rs)", 1, 0, 'C')
        pdf.cell(40, 10, "Total (Rs)", 1, 1, 'C')

        # Table Rows
        pdf.set_font("Arial", size=12)
        for item, count in item_counts.items():
            if count > 0:
                item_total = count * item_prices[item]
                pdf.cell(80, 10, item, 1)
                pdf.cell(30, 10, str(count), 1)
                pdf.cell(40, 10, f"Rs{item_prices[item]:.2f}", 1)
                pdf.cell(40, 10, f"Rs{item_total:.2f}", 1)
                pdf.ln()

        # Total Bill
        pdf.set_font("Arial", "B", 12)
        pdf.cell(150, 10, "Total Bill", 1)
        pdf.cell(40, 10, f"Rs{total_bill:.2f}", 1)
        return pdf

    # Generate and offer the PDF for download
    pdf = generate_pdf(bill_details, total_bill)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        pdf.output(tmpfile.name)
        tmpfile.close()

        # Generate download button for the PDF
        with open(tmpfile.name, "rb") as pdf_file:
            st.download_button(
                label="Download Bill as PDF",
                data=pdf_file.read(),
                file_name="bill_summary.pdf",
                mime="application/pdf",
                key="download_pdf"
            )

        # Delay file deletion to ensure Streamlit is done using the file
        time.sleep(5)  # Wait for 5 seconds
        try:
            os.remove(tmpfile.name)
        except Exception as e:
            st.error(f"Error removing the file: {e}")
