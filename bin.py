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
bold_style = ParagraphStyle(name='Bold', fontName='Helvetica-Bold', fontSize=16, alignment=TA_LEFT, leading=14)
desc_style = ParagraphStyle(name='Description', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)
qty_style = ParagraphStyle(name='Quantity', fontName='Helvetica', fontSize=11, alignment=TA_CENTER, leading=12)

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

def find_bin_type_column(df_columns):
    """
    Function to find the bin type/container column
    """
    cols = [str(col).upper() for col in df_columns]
    
    # Priority order for bin type column detection
    patterns = [
        # Exact matches (highest priority)
        lambda col: col == 'BIN_TYPE',
        lambda col: col == 'BINTYPE',
        lambda col: col == 'BIN TYPE',
        lambda col: col == 'CONTAINER',
        lambda col: col == 'CONTAINER_TYPE',
        lambda col: col == 'CONTAINERTYPE',
        lambda col: col == 'CONTAINER TYPE',
        lambda col: col == 'BIN_CONTAINER',
        lambda col: col == 'BINCONTAINER',
        lambda col: col == 'BIN CONTAINER',
        # Partial matches (lower priority)
        lambda col: 'BIN' in col and 'TYPE' in col,
        lambda col: 'CONTAINER' in col and 'TYPE' in col,
        lambda col: 'BIN' in col and 'CONTAINER' in col,
        lambda col: 'TYPE' in col and ('BIN' in col or 'CONTAINER' in col),
        lambda col: 'BIN' in col,
        lambda col: 'CONTAINER' in col,
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

    # Find bin type/container column using the new detection function
    bin_type_col = find_bin_type_column(original_columns)

    if status_callback:
        status_callback(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            status_callback(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            status_callback(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            status_callback(f"Bus Model Column: {bus_model_col}")
        if bin_type_col:
            status_callback(f"Bin Type/Container Column: {bin_type_col}")
    else:
        st.write(f"Using columns: Part No: {part_no_col}, Description: {desc_col}, Location: {loc_col}, Qty/Bin: {qty_bin_col}")
        if qty_veh_col:
            st.write(f"Qty/Veh Column: {qty_veh_col}")
        if store_loc_col:
            st.write(f"Store Location Column: {store_loc_col}")
        if bus_model_col:
            st.write(f"Bus Model Column: {bus_model_col}")
        if bin_type_col:
            st.write(f"Bin Type/Container Column: {bin_type_col}")

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

        # Extract Bin Type/Container properly
        bin_type = ""
        if bin_type_col and bin_type_col in row and pd.notna(row[bin_type_col]):
            bin_type = str(row[bin_type_col])
            
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
        qr_data += f"Store Location: {store_location}\nQTY/VEH: {qty_veh}\nQTY/BIN: {qty_bin}\nBin Type: {bin_type}"
        
        qr_image = generate_qr_code(qr_data)
        if status_callback and qr_image:
            status_callback(f"QR code generated for part: {part_no}")
        
        # Define row heights
        header_row_height = 0.9*cm
        desc_row_height = 1.0*cm
        qty_row_height = 0.5*cm
        location_row_height = 0.5*cm

        # Main table data - Modified to include 3-column qty/bin row
        main_table_data = [
            ["Part No", Paragraph(f"{part_no}", bold_style)],
            ["Description", Paragraph(desc[:47] + "..." if len(desc) > 50 else desc, desc_style)]
        ]

        # Create main table for Part No and Description
        main_table = Table(main_table_data,
                         colWidths=[content_width/3, content_width*2/3],
                         rowHeights=[header_row_height, desc_row_height])

        main_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, -1), 11),
        ]))

        elements.append(main_table)

        # Create separate qty/bin table with 3 columns
        qty_bin_table_data = [
            ["Qty/Bin", Paragraph(str(qty_bin), qty_style), Paragraph(str(bin_type), qty_style)]
        ]

        # Calculate column widths for the qty/bin row
        # 1st column: 1/3 of content width (same as header column)
        # 2nd and 3rd columns: split the remaining 2/3 equally
        qty_bin_col_widths = [content_width/3, content_width/3, content_width/3]

        qty_bin_table = Table(qty_bin_table_data,
                            colWidths=qty_bin_col_widths,
                            rowHeights=[qty_row_height])

        qty_bin_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),  # Darker grid lines
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),  # Make header bold
            ('FONTSIZE', (0, 0), (-1, -1), 11),
        ]))

        elements.append(qty_bin_table)

        # Store Location section
        store_loc_label = Paragraph("Store Location", ParagraphStyle(
            name='StoreLoc', fontName='Helvetica-Bold', fontSize=11, alignment=TA_CENTER
        ))

        # Total width for the 7 inner columns (2/3 of full content width)
        inner_table_width = content_width * 2 / 3
        
        # Define proportional widths - same as Line Location for consistency
        col_proportions = [1.5, 2, 0.7, 0.8, 1, 1, 0.9]
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
        position_matrix = Table(position_matrix_data,
                               colWidths=[mtm_box_width] * 3,
                               rowHeights=[0.5*cm, mtm_row_height])

        position_matrix.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),  # Header row bold
            ('FONTSIZE', (0, 0), (-1, 0), 9),  # Header row font size
            ('FONTSIZE', (0, 1), (-1, 1), 12), # Data row font size
        ]))

        # Create the bottom row table
        qr_and_matrix_table_data = [["MTM", position_matrix]]
        
        # Add QR code if available
        if qr_image:
            qr_and_matrix_table_data[0].append(qr_image)
            bottom_col_widths = [0.8*cm, mtm_box_width * 3, 2.5*cm]
        else:
            bottom_col_widths = [0.8*cm, mtm_box_width * 3]

        qr_and_matrix_table = Table(qr_and_matrix_table_data,
                                   colWidths=bottom_col_widths,
                                   rowHeights=[mtm_row_height])

        qr_and_matrix_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1.2, colors.Color(0, 0, 0, alpha=0.95)),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (0, 0), 11),
        ]))

        elements.append(qr_and_matrix_table)

        # Add all elements for this sticker
        all_elements.extend(elements)
        
        # Add page break after each sticker except the last one
        if index < total_rows - 1:
            all_elements.append(PageBreak())

    # Build the PDF
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

def main():
    st.title("Sticker Label Generator with QR Codes")
    
    st.markdown("""
    ### Instructions:
    1. Upload an Excel (.xlsx) or CSV (.csv) file containing your data
    2. The system will automatically detect columns for:
       - Part Number (columns containing 'PART' and 'NO'/'NUM'/'#')
       - Description (columns containing 'DESC' or 'NAME')
       - Location (columns containing 'LOC' or 'LOCATION')
       - Quantity per Bin (columns containing 'QTY/BIN' or 'QTY')
       - Quantity per Vehicle (columns containing 'QTY/VEH' or similar)
       - Bus Model (columns containing 'BUS', 'MODEL', 'VEHICLE', 'TYPE')
       - Bin Type/Container (columns containing 'BIN', 'CONTAINER', 'TYPE')
       - Store Location (columns containing 'STORE' and 'LOC')
    3. Click "Generate Sticker Labels" to create the PDF
    4. Download the generated PDF
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose an Excel or CSV file", 
                                   type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, 
                                       suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
        
        # Display file info
        st.success(f"File uploaded: {uploaded_file.name}")
        
        # Preview data
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                preview_df = pd.read_csv(temp_file_path)
            else:
                preview_df = pd.read_excel(temp_file_path)
            
            st.subheader("Data Preview (first 5 rows):")
            st.dataframe(preview_df.head())
            
            st.subheader("Column Detection:")
            cols = [col.upper() if isinstance(col, str) else col for col in preview_df.columns]
            
            # Show detected columns
            part_no_col = next((col for col in cols if 'PART' in col and ('NO' in col or 'NUM' in col or '#' in col)),
                              next((col for col in cols if col in ['PARTNO', 'PART']), cols[0]))
            
            desc_col = next((col for col in cols if 'DESC' in col),
                           next((col for col in cols if 'NAME' in col), cols[1] if len(cols) > 1 else part_no_col))
            
            qty_bin_col = next((col for col in cols if 'QTY/BIN' in col or 'QTY_BIN' in col or 'QTYBIN' in col), 
                              next((col for col in cols if 'QTY' in col and 'BIN' in col), None))
            
            if not qty_bin_col:
                qty_bin_col = next((col for col in cols if 'QTY' in col),
                                  next((col for col in cols if 'QUANTITY' in col), None))
            
            loc_col = next((col for col in cols if 'LOC' in col or 'POS' in col or 'LOCATION' in col),
                          cols[2] if len(cols) > 2 else desc_col)
            
            qty_veh_col = next((col for col in cols if any(term in col for term in ['QTY/VEH', 'QTY_VEH', 'QTY PER VEH', 'QTYVEH', 'QTYPERCAR', 'QTYCAR', 'QTY/CAR'])), None)
            
            store_loc_col = next((col for col in cols if 'STORE' in col and 'LOC' in col),
                                next((col for col in cols if 'STORELOCATION' in col), None))
            
            bus_model_col = find_bus_model_column(preview_df.columns.tolist())
            bin_type_col = find_bin_type_column(preview_df.columns.tolist())
            
            detection_info = {
                "Part Number": part_no_col,
                "Description": desc_col,
                "Location": loc_col,
                "Qty/Bin": qty_bin_col,
                "Qty/Vehicle": qty_veh_col or "Not detected",
                "Store Location": store_loc_col or "Not detected",
                "Bus Model": bus_model_col or "Not detected",
                "Bin Type/Container": bin_type_col or "Not detected"
            }
            
            for label, col in detection_info.items():
                st.write(f"**{label}:** {col}")
            
        except Exception as e:
            st.error(f"Error previewing file: {e}")
        
        # Generate button
        if st.button("Generate Sticker Labels", type="primary"):
            # Create progress placeholder
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            def update_status(message):
                status_placeholder.text(message)
            
            try:
                # Generate output filename
                output_filename = f"sticker_labels_{uploaded_file.name.split('.')[0]}.pdf"
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_pdf:
                    output_path = tmp_pdf.name
                
                # Generate stickers
                result_path = generate_sticker_labels(temp_file_path, output_path, update_status)
                
                if result_path and os.path.exists(result_path):
                    # Read the generated PDF
                    with open(result_path, 'rb') as pdf_file:
                        pdf_data = pdf_file.read()
                    
                    # Provide download button
                    st.success("Sticker labels generated successfully!")
                    st.download_button(
                        label="Download PDF",
                        data=pdf_data,
                        file_name=output_filename,
                        mime="application/pdf"
                    )
                    
                    # Clean up temporary files
                    try:
                        os.unlink(result_path)
                        os.unlink(temp_file_path)
                    except:
                        pass
                        
                else:
                    st.error("Failed to generate PDF")
                    
            except Exception as e:
                st.error(f"Error generating stickers: {e}")
                import traceback
                st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
