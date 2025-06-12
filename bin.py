import streamlit as st
import pandas as pd
import os
from reportlab.lib.pagesizes import landscape
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph, PageBreak, Image
from reportlab.lib.units import cm, inch
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.utils import ImageReader
from io import BytesIO
import subprocess
import sys
import re
import tempfile

# Define sticker dimensions
STICKER_WIDTH = 10 * cm
STICKER_HEIGHT = 15 * cm
STICKER_PAGESIZE = (STICKER_WIDTH, STICKER_HEIGHT)

# Define content box dimensions
CONTENT_BOX_WIDTH = 10 * cm  # Same width as page
CONTENT_BOX_HEIGHT = 7.2 * cm  # Half the page height

# Check for PIL and install if needed
try:
    from PIL import Image as PILImage
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    st.write("PIL not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pillow'])
    from PIL import Image as PILImage
    PIL_AVAILABLE = True

# Check for QR code library and install if needed
try:
    import qrcode
    QR_AVAILABLE = True
except ImportError:
    QR_AVAILABLE = False
    st.write("qrcode not available. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'qrcode'])
    import qrcode
    QR_AVAILABLE = True

# Define paragraph styles
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=12, alignment=TA_LEFT, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=8, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

def get_dynamic_part_no_style(text):
    """
    Dynamically adjust font size for Part No based on text length
    """
    text_length = len(str(text))
    
    if text_length <= 12:
        font_size = 16
        leading = 18
    elif text_length <= 14:
        font_size = 16
        leading = 18
    elif text_length <= 18:
        font_size = 14
        leading = 16
    elif text_length <= 22:
        font_size = 12
        leading = 14
    elif text_length <= 25:
        font_size = 12
        leading = 12
    else:
        font_size = 11
        leading = 10
    
    return ParagraphStyle(
        name='DynamicPartNo',
        fontName='Helvetica-Bold',
        fontSize=font_size,
        alignment=TA_LEFT,
        leading=leading
    )

def get_dynamic_description_style(text):
    """
    Dynamically adjust font size for Description based on text length
    """
    text_length = len(str(text))
    
    if text_length <= 30:
        font_size = 10
        leading = 12
    elif text_length <= 50:
        font_size = 9
        leading = 11
    elif text_length <= 70:
        font_size = 8
        leading = 10
    elif text_length <= 90:
        font_size = 7
        leading = 9
    elif text_length <= 120:
        font_size = 6
        leading = 8
    else:
        font_size = 5
        leading = 7
    
    return ParagraphStyle(
        name='DynamicDescription',
        fontName='Helvetica',
        fontSize=font_size,
        alignment=TA_CENTER,
        leading=leading
    )

def find_bus_model_column(df_columns):
    """
    Enhanced function to find the bus model column with better detection
    """
    cols = [str(col).upper() for col in df_columns]
    
    # Priority order for bus model column detection
    patterns = [
        # Exact matches (highest priority)
        lambda col: col == 'BUS_MODEL',
        lambda col: col == 'BUSMODEL',
        lambda col: col == 'BUS MODEL',
        lambda col: col == 'MODEL',
        lambda col: col == 'BUS_TYPE',
        lambda col: col == 'BUSTYPE',
        lambda col: col == 'BUS TYPE',
        lambda col: col == 'VEHICLE_TYPE',
        lambda col: col == 'VEHICLETYPE',
        lambda col: col == 'VEHICLE TYPE',
        # Partial matches (lower priority)
        lambda col: 'BUS' in col and 'MODEL' in col,
        lambda col: 'BUS' in col and 'TYPE' in col,
        lambda col: 'VEHICLE' in col and 'MODEL' in col,
        lambda col: 'VEHICLE' in col and 'TYPE' in col,
        lambda col: 'MODEL' in col,
        lambda col: 'BUS' in col,
        lambda col: 'VEHICLE' in col,
    ]
    
    for pattern in patterns:
        for i, col in enumerate(cols):
            if pattern(col):
                return df_columns[i]  # Return original column name
    
    return None

def detect_bus_model_and_qty(row, qty_veh_col, bus_model_col=None):
    """
    Improved bus model detection that properly matches bus model to MTM box
    Returns a dictionary with keys '7M', '9M', '12M' and their respective quantities
    """
    # Initialize result dictionary
    result = {'7M': '', '9M': '', '12M': ''}
    
    # Get quantity value
    qty_veh = ""
    if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
        qty_veh = str(row[qty_veh_col]).strip()
    
    if not qty_veh:
        return result
    
    # Method 1: Check if quantity already contains model info (e.g., "9M:2", "7M-3", "12M 5")
    qty_pattern = r'(\d+M)[:\-\s]*(\d+)'
    matches = re.findall(qty_pattern, qty_veh.upper())
    
    if matches:
        # If we found model-quantity pairs in the qty_veh field itself
        for model, quantity in matches:
            if model in result:
                result[model] = quantity
        return result
    
    # Method 2: Look for bus model in dedicated bus model column first
    detected_model = None
    if bus_model_col and bus_model_col in row and pd.notna(row[bus_model_col]):
        bus_model_value = str(row[bus_model_col]).strip().upper()
        
        # Check for exact matches first
        if bus_model_value in ['7M', '7']:
            detected_model = '7M'
        elif bus_model_value in ['9M', '9']:
            detected_model = '9M'
        elif bus_model_value in ['12M', '12']:
            detected_model = '12M'
        # Check for patterns within the text
        elif re.search(r'\b7M\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9M\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12M\b', bus_model_value):
            detected_model = '12M'
        # Check for standalone numbers
        elif re.search(r'\b7\b', bus_model_value):
            detected_model = '7M'
        elif re.search(r'\b9\b', bus_model_value):
            detected_model = '9M'
        elif re.search(r'\b12\b', bus_model_value):
            detected_model = '12M'
    
    # If we found a model in the dedicated column, use it
    if detected_model:
        result[detected_model] = qty_veh
        return result
    
    # Method 3: Search through all columns systematically with priority
    # First, search in columns that are most likely to contain bus model info
    priority_columns = []
    other_columns = []
    
    for col in row.index:
        if pd.notna(row[col]):
            col_upper = str(col).upper()
            # High priority columns
            if any(keyword in col_upper for keyword in ['MODEL', 'BUS', 'VEHICLE', 'TYPE']):
                priority_columns.append(col)
            else:
                other_columns.append(col)
    
    # Search priority columns first
    for col in priority_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Look for exact matches first
            if re.search(r'\b7M\b', value_str):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9M\b', value_str):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12M\b', value_str):
                result['12M'] = qty_veh
                return result
            # Then look for standalone numbers in context
            elif re.search(r'\b7\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['7M'] = qty_veh
                return result
            elif re.search(r'\b9\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['9M'] = qty_veh
                return result
            elif re.search(r'\b12\b', value_str) and any(keyword in value_str for keyword in ['BUS', 'METER', 'M']):
                result['12M'] = qty_veh
                return result
    
    # Method 4: Search in other columns as fallback
    detected_models = []
    for col in other_columns:
        if pd.notna(row[col]):
            value_str = str(row[col]).upper()
            
            # Use word boundaries to avoid false matches
            if re.search(r'\b7M\b', value_str):
                detected_models.append('7M')
            elif re.search(r'\b9M\b', value_str):
                detected_models.append('9M')
            elif re.search(r'\b12M\b', value_str):
                detected_models.append('12M')
    
    # Remove duplicates while preserving order
    detected_models = list(dict.fromkeys(detected_models))
    
    if detected_models:
        # Use the first detected model
        result[detected_models[0]] = qty_veh
        return result
    
    # Method 5: Last resort - look for standalone numbers that might indicate bus length
    for col in row.index:
        if pd.notna(row[col]):
            value_str = str(row[col]).strip()
            
            # Look for exact matches of just the number
            if value_str == '7':
                result['7M'] = qty_veh
                return result
            elif value_str == '9':
                result['9M'] = qty_veh
                return result
            elif value_str == '12':
                result['12M'] = qty_veh
                return result
    
    # Method 6: If still no model detected, return empty (no boxes filled)
    return result

def generate_qr_code(data_string):
    """
    Generate a QR code from the given data string
    """
    try:
        # Create QR code instance
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        
        # Add data
        qr.add_data(data_string)
        qr.make(fit=True)
        
        # Create QR code image
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert PIL image to bytes that reportlab can use
        img_buffer = BytesIO()
        qr_img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        # Create a QR code image with specified size
        return Image(img_buffer, width=2.5*cm, height=2.5*cm)
    except Exception as e:
        st.error(f"Error generating QR code: {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_location_string(location_str):
    """Parse a location string into components for table display"""
    # Initialize with empty values
    location_parts = [''] * 7

    if not location_str or not isinstance(location_str, str):
        return location_parts

    # Remove any extra spaces
    location_str = location_str.strip()

    # Try to parse location components
    import re
    pattern = r'([^_\s]+)'
    matches = re.findall(pattern, location_str)

    # Fill the available parts
    for i, match in enumerate(matches[:7]):
        location_parts[i] = match

    return location_parts

def generate_sticker_labels(excel_file_path, output_pdf_path, status_callback=None):
    """Generate sticker labels with QR code from Excel data"""
    if status_callback:
        status_callback(f"Processing file: {excel_file_path}")
    else:
        st.write(f"Processing file: {excel_file_path}")

    # Create a function to draw the border box around content
    def draw_border(canvas, doc):
        canvas.saveState()
        # Draw border box around the content area (10cm x 7.5cm)
        # Position it at the top of the page with minimal margin
        x_offset = (STICKER_WIDTH - CONTENT_BOX_WIDTH) / 2
        y_offset = STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm  # Position at top with minimal margin
        canvas.setStrokeColor(colors.Color(0, 0, 0, alpha=0.95))  # Slightly darker black (95% opacity)
        canvas.setLineWidth(1.8)  # Slightly thicker border
        canvas.rect(
            x_offset + doc.leftMargin,
            y_offset,
            CONTENT_BOX_WIDTH - 0.2*cm,  # Account for margins
            CONTENT_BOX_HEIGHT
        )
        canvas.restoreState()

    # Load the Excel data
    try:
        if excel_file_path.lower().endswith('.csv'):
            df = pd.read_csv(excel_file_path)
        else:
            try:
                df = pd.read_excel(excel_file_path)
            except Exception as e:
                try:
                    df = pd.read_excel(excel_file_path, engine='openpyxl')
                except Exception as e2:
                    df = pd.read_csv(excel_file_path, encoding='latin1')

        if status_callback:
            status_callback(f"Successfully read file with {len(df)} rows")
            status_callback(f"Columns found: {df.columns.tolist()}")
        else:
            st.write(f"Successfully read file with {len(df)} rows")
            st.write("Columns found:", df.columns.tolist())
    except Exception as e:
        error_msg = f"Error reading file: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        return None

    # Identify columns (case-insensitive)
    original_columns = df.columns.tolist()
    df.columns = [col.upper() if isinstance(col, str) else col for col in df.columns]
    cols = df.columns.tolist()

    # Find relevant columns
    part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                   next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))

    desc_col = next((col for col in cols if 'DESC' in col),
                   next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))

    # Look specifically for "QTY/BIN" column first, then fall back to general QTY column
    qty_bin_col = next((col for col in cols if 'QTY/BIN' in col or 'QTY_BIN' in col or 'QTYBIN' in col), 
                  next((col for col in cols if 'QTY' in col and 'BIN' in col), None))
    
    # If no specific QTY/BIN column is found, fall back to general QTY column
    if not qty_bin_col:
        qty_bin_col = next((col for col in cols if 'QTY' in col),
                      next((col for col in cols if 'QUANTITY' in col), None))
  
    loc_col = next((col for col in cols if 'LOC' in col or 'POS' in col or 'LOCATION' in col),
                   cols[2] if len(cols) > 2 else desc_col)

    # Improved detection of QTY/VEH column
    qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)

    # Look for store location column
    store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                      next((col for col in cols if 'STORELOCATION' in col), None))

    # Find bus model column using the enhanced detection function
    bus_model_col = find_bus_model_column(original_columns)

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            status_callback(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            st.write(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")

    # Create document with minimal margins
    doc = SimpleDocTemplate(output_pdf_path, pagesize=STICKER_PAGESIZE,
                          topMargin=0.2*cm,  # Minimal top margin
                          bottomMargin=(STICKER_HEIGHT - CONTENT_BOX_HEIGHT - 0.2*cm),  # Adjust bottom margin accordingly
                          leftMargin=0.1*cm, rightMargin=0.1*cm)

    content_width = CONTENT_BOX_WIDTH - 0.2*cm
    all_elements = []

    # Process each row as a single sticker
    total_rows = len(df)
    for index, row in df.iterrows():
        # Update progress
        if status_callback:
            status_callback(f"Creating sticker {index+1} of {total_rows} ({int((index+1)/total_rows*100)}%)")
        
        elements = []

        # Extract data
        part_no = str(row[part_no_col])
        desc = str(row[desc_col])
        
        # Extract QTY/BIN properly
        qty_bin = ""
        if qty_bin_col and qty_bin_col in row and pd.notna(row[qty_bin_col]):
            qty_bin = str(row[qty_bin_col])
            
        # Extract QTY/VEH properly
        qty_veh = ""
        if qty_veh_col and qty_veh_col in row and pd.notna(row[qty_veh_col]):
            qty_veh = str(row[qty_veh_col])
        
        location_str = str(row[loc_col]) if loc_col and loc_col in row else ""
        store_location = str(row[store_loc_col]) if store_loc_col and store_loc_col in row else ""
        location_parts = parse_location_string(location_str)

        # Use enhanced bus model detection
        mtm_quantities = detect_bus_model_and_qty(row, qty_veh_col, bus_model_col)

        # Generate QR code with part information
        qr_data = f"Part No: {part_no}\nDescription: {desc}\nLocation: {location_str}\n"
        qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}"
        
        qr_image = generate_qr_code(qr_data)
        if status_callback and qr_image:
            status_callback(f"QR code generated for part: {part_no}")
        
        # Define row heights
        header_row_height = 0.9*cm
        desc_row_height = 1.0*cm
        qty_row_height = 0.5*cm
        location_row_height = 0.5*cm

        # Create dynamic styles for Part No and Description
        part_no_style = get_dynamic_part_no_style(part_no)
        desc_style_dynamic = get_dynamic_description_style(desc)

        # Main table data with dynamic font sizing
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", part_no_style)],
            ["Description", Paragraph(desc, desc_style_dynamic)],
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style)]
        ]

        # Create main table
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height, qty_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(main_table)

        # Store Location section
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
            name='StoreLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        # Total width for the 7 inner columns (2/3 of full content width)
        inner_table_width = content_width * 2 / 3
        
        # Define proportional widths - same as Line Location for consistency
        col_proportions = [1.5, 1.8, 0.9, 0.8, 1, 1, 0.9]
        total_proportion = sum(col_proportions)
        
        # Calculate column widths based on proportions 
        inner_col_widths = [w * inner_table_width / total_proportion for w in col_proportions]

        # Use store_location if available, otherwise use empty values
        store_loc_values = parse_location_string(store_location) if store_location else ["", "", "", "", "", "", ""]

        store_loc_inner_table = Table(
            [store_loc_values],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )

        store_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make store location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        store_loc_table = Table(
            [[store_loc_label, store_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        store_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(store_loc_table)

        # Line Location section
        line_loc_label = Paragraph("Line Location", ParagraphStyle(
            name='LineLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))
        
        # The inner table width is already calculated above
        
        # Create the inner table
        line_loc_inner_table = Table(
            [location_parts],
            colWidths=inner_col_widths,
            rowHeights=[location_row_height]
        )
        
        line_loc_inner_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),  # Make line location values bold
            ('FONTSIZE', (0, 0), (-1, -1), 9)
        ]))
        
        # Wrap the label and the inner table in a containing table
        line_loc_table = Table(
            [[line_loc_label, line_loc_inner_table]],
            colWidths=[content_width/3, inner_table_width],
            rowHeights=[location_row_height]
        )

        line_loc_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(line_loc_table)

        # Add smaller spacer between line location and bottom section
        elements.append(Spacer(1, 0.3*cm))

        # Bottom section - Enhanced with intelligent bus model detection
        mtm_box_width = 1.2*cm
        mtm_row_height = 1.5*cm

        # Create MTM boxes with detected quantities
        position_matrix_data = [
            ["7M", "9M", "12M"],
            [
                Paragraph(f"<b>{mtm_quantities['7M']}</b>", ParagraphStyle(
                    name='Bold7M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['7M'] else "",
                Paragraph(f"<b>{mtm_quantities['9M']}</b>", ParagraphStyle(
                    name='Bold9M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['9M'] else "",
                Paragraph(f"<b>{mtm_quantities['12M']}</b>", ParagraphStyle(
                    name='Bold12M', fontName='Helvetica-Bold', fontSize=10, alignment=TA_CENTER
                )) if mtm_quantities['12M'] else ""
            ]
        ]

        mtm_table = Table(
            position_matrix_data,
            colWidths=[mtm_box_width, mtm_box_width, mtm_box_width],
            rowHeights=[mtm_row_height/2, mtm_row_height/2]
        )

        mtm_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
        ]))

        # QR code with preserved size
        qr_width = 2.5*cm
        qr_height = 2.5*cm

        if qr_image:
            qr_table = Table(
                [[qr_image]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )
        else:
            qr_table = Table(
                [[Paragraph("QR", ParagraphStyle(
                    name='QRPlaceholder', fontName='Helvetica-Bold', fontSize=12, alignment=TA_CENTER
                ))]],
                colWidths=[qr_width],
                rowHeights=[qr_height]
            )

        qr_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        # Bottom section layout
        bottom_section_data = [
            [mtm_table, qr_table]
        ]

        # Calculate remaining width for spacing
        remaining_width = content_width - (mtm_box_width * 3) - qr_width
        spacing_width = remaining_width / 2

        bottom_table = Table(
            bottom_section_data,
            colWidths=[mtm_box_width * 3, qr_width],
            rowHeights=[max(mtm_row_height, qr_height)]
        )

        bottom_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        elements.append(bottom_table)

        # Add all elements for this sticker
        all_elements.extend(elements)
        
        # Add page break between stickers (except for the last one)
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build the PDF with border
    try:
        if status_callback:
            status_callback("Building PDF document...")
        else:
            st.write("Building PDF document...")
            
        doc.build(all_elements, onFirstPage=draw_border, onLaterPages=draw_border)
        
        if status_callback:
            status_callback(f"PDF generated successfully: {output_pdf_path}")
        else:
            st.success(f"PDF generated successfully: {output_pdf_path}")
        
        return output_pdf_path
    except Exception as e:
        error_msg = f"Error building PDF: {e}"
        if status_callback:
            status_callback(error_msg)
        else:
            st.error(error_msg)
        import traceback
        traceback.print_exc()
        return None

def main():
    st.title("🏷️ Sticker Label Generator")
    st.write("Upload an Excel file to generate professional sticker labels with QR codes")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an Excel or CSV file", 
        type=['xlsx', 'xls', 'csv'],
        help="Upload your Excel or CSV file containing part information"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        st.success("File uploaded successfully!")
        
        # Show file info
        st.info(f"File: {uploaded_file.name} ({uploaded_file.size} bytes)")

        # Generate labels button
        if st.button("🚀 Generate Sticker Labels", type="primary"):
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_status(message):
                status_text.text(message)
                # Simple progress estimation based on message content
                if "Creating sticker" in message and "of" in message:
                    try:
                        # Extract current/total from message
                        parts = message.split()
                        current = int(parts[2])
                        total = int(parts[4])
                        progress = current / total
                        progress_bar.progress(progress)
                    except:
                        pass
            
            # Generate output filename
            output_filename = f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            
            try:
                # Generate the PDF
                result_path = generate_sticker_labels(temp_file_path, output_path, update_status)
                
                if result_path:
                    progress_bar.progress(1.0)
                    status_text.text("✅ PDF generation completed!")
                    
                    # Read the generated PDF
                    with open(result_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download Sticker Labels PDF",
                        data=pdf_data,
                        file_name=output_filename,
                        mime="application/pdf",
                        type="primary"
                    )
                    
                    st.success(f"✨ Successfully generated {output_filename}!")
                    
                    # Clean up temporary files
                    try:
                        os.unlink(temp_file_path)
                        os.unlink(result_path)
                    except:
                        pass
                        
                else:
                    st.error("Failed to generate PDF. Please check your file format and try again.")
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
    
    # Instructions
    with st.expander("📋 Instructions"):
        st.markdown("""
        ### How to use this tool:
        
        1. **Upload your file**: Choose an Excel (.xlsx, .xls) or CSV file containing your part data
        2. **Required columns**: The tool will automatically detect columns for:
           - Part Number (looks for columns containing 'PART', 'NO', 'NUM')
           - Description (looks for 'DESC', 'NAME')
           - Location (looks for 'LOC', 'POS', 'LOCATION')
           - Quantity per Bin (looks for 'QTY/BIN', 'QTY')
           - Quantity per Vehicle (looks for 'QTY/VEH', 'QTY_VEH')
           - Bus Model (looks for 'BUS_MODEL', 'MODEL', 'BUS_TYPE')
           - Store Location (looks for 'STORE' and 'LOC')
        
        3. **Bus Model Detection**: The tool intelligently detects bus models (7M, 9M, 12M) from:
           - Dedicated bus model columns
           - Quantity fields with model info (e.g., "7M:2", "9M-3")
           - Any column containing model information
        
        4. **Output**: Each row generates one sticker label with:
           - Part number and description
           - QR code with part information
           - Location details (store and line locations)
           - MTM boxes (7M, 9M, 12M) with detected quantities
        
        ### Features:
        - **Smart column detection** - automatically finds relevant columns
        - **Dynamic font sizing** - adjusts text size based on content length
        - **QR code generation** - includes part details for easy scanning
        - **Professional layout** - bordered stickers ready for printing
        - **Bus model intelligence** - automatically fills MTM boxes based on detected models
        """)

if __name__ == "__main__":
    main()
